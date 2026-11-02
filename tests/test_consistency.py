from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from training.run import run_training
from training.checkpoint import read_checkpoint
from training.consistency import verify_commit, reconcile
from training.storage import sync_results
from test_training import fixture_data, tiny_model


@pytest.mark.parametrize('economical', [False, True])
@pytest.mark.parametrize('failure', ['staged', 'committed', 'drive_weights', 'drive_metrics'])
def test_crash_recovery_matches_reference(tmp_path, economical, failure):
    import training.run as runner
    import training.consistency as consistency
    import training.storage as storage
    config = replace(fixture_data(tmp_path), epochs=4, economical=economical,
                     work_dir=str(tmp_path / 'local') if economical else None)
    def validation(*args, **kwargs):
        score = [.8, .9, .7, .9][torch.initial_seed() - config.seed]
        return {'loss': 1., 'macro_f1': score, 'accuracy': score}
    with patch('training.run.build_model', side_effect=tiny_model), patch('training.run.validate', side_effect=validation):
        reference = run_training(config)
        saved = runner.save_checkpoint
        atomic = consistency.atomic_save
        copy = storage.shutil.copyfile
        sync = runner.sync_results
        crashed = False
        def fail():
            nonlocal crashed
            crashed = True
            raise OSError('synthetic crash')
        def save(path, *args, **kwargs):
            if not crashed and failure == 'staged' and path.name == 'last.pt' and args[4] == 1:
                fail()
            return saved(path, *args, **kwargs)
        def publish(path, writer):
            if not crashed and failure == 'committed' and path.name == 'best_weights.pt' and read_checkpoint(path.parent / 'last.pt')['epoch'] == 1:
                fail()
            return atomic(path, writer)
        def copying(source, dest):
            if not crashed and failure == 'drive_weights' and source.name == 'best_weights.pt' and read_checkpoint(source.parent / 'last.pt')['epoch'] == 1:
                Path(dest).write_bytes(b'partial')
                fail()
            return copy(source, dest)
        def syncing(run, durable, **kwargs):
            if not crashed and (failure == 'drive_metrics' or not economical and failure == 'drive_weights') and (run / 'last.pt').exists() and read_checkpoint(run / 'last.pt')['epoch'] == 1:
                fail()
            return sync(run, durable, **kwargs)
        with patch('training.run.save_checkpoint', side_effect=save), patch('training.consistency.atomic_save', side_effect=publish), patch('training.storage.shutil.copyfile', side_effect=copying), patch('training.run.sync_results', side_effect=syncing):
            with pytest.raises(OSError, match='synthetic crash'):
                run_training(config)
        base = Path(config.work_dir if economical else config.output) / config.model
        interrupted = next(p for p in base.iterdir() if p != reference)
        run_training(config, interrupted)
    for name in ('best_weights.pt', 'last.pt'):
        a = torch.load(reference / name, weights_only=True) if name == 'best_weights.pt' else read_checkpoint(reference / name)['model']
        b = torch.load(interrupted / name, weights_only=True) if name == 'best_weights.pt' else read_checkpoint(interrupted / name)['model']
        for key in a:
            assert torch.equal(a[key], b[key])
    a, b = read_checkpoint(reference / 'last.pt'), read_checkpoint(interrupted / 'last.pt')
    assert a['best_selection'] == b['best_selection']
    assert b['best_selection']['epoch'] == 2  # first .9, not the final tie
    assert [{k:v for k,v in row.items() if k != 'seconds'} for row in a['history']] == [{k:v for k,v in row.items() if k != 'seconds'} for row in b['history']]
    verify_commit(interrupted)
    if economical:
        durable = Path(json.loads((interrupted / 'storage.json').read_text())['durable_directory'])
        verify_commit(durable)
        for name in ('best_weights.pt', 'metrics.json', 'summary.json', 'commit.json'):
            assert (durable / name).read_bytes() == (interrupted / name).read_bytes()


def test_missing_best_fails_and_legacy_recovers(tmp_path):
    config = fixture_data(tmp_path)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(config)
    state = read_checkpoint(run / 'last.pt')
    # Legacy best.pt proves provenance even when best is not last.
    state.pop('best_selection')
    (run / 'best_weights.pt').write_bytes(b'corrupt')
    selection = reconcile(run, state, False)
    verify_commit(run)
    assert selection['epoch'] == max(state['history'], key=lambda r:r['validation']['macro_f1'])['epoch']
    # Construct an earlier selected epoch with all recovery sources lost.
    state['best_selection'] = selection
    state['epoch'] = 20
    (run / 'best.pt').unlink()
    (run / 'best_weights.pt').write_bytes(b'corrupt')
    (run / 'best_pending.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='Cannot recover'):
        reconcile(run, state, True)


def test_partial_publication_rejected_and_ahead_weights_repaired(tmp_path):
    from training.checkpoint import load_weights, export_weights
    config = fixture_data(tmp_path)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(config)
    saved = read_checkpoint(run / 'last.pt')
    expected = torch.load(run / 'best_weights.pt', weights_only=True)
    # Simulate publishing unrelated/ahead weights without committing history.
    unrelated, _ = tiny_model()
    export_weights(run / 'best_weights.pt', unrelated)
    with pytest.raises(RuntimeError, match='Incomplete or corrupt'):
        load_weights(run / 'best_weights.pt', unrelated)
    reconcile(run, saved, False)
    verify_commit(run)
    actual = torch.load(run / 'best_weights.pt', weights_only=True)
    for key in expected:
        assert torch.equal(actual[key], expected[key])


def test_first_drive_publication_requires_commit(tmp_path):
    from training.checkpoint import load_weights, export_weights
    model, _ = tiny_model()
    export_weights(tmp_path / 'best_weights.pt', model)
    (tmp_path / 'storage.json').write_text('{}')
    with pytest.raises(RuntimeError, match='missing commit.json'):
        load_weights(tmp_path / 'best_weights.pt', model)
