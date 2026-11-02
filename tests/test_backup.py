"""Full-backup crash/recovery tests on small synthetic models only."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import torch
from training.backup import BackupError, restore_run, sync_checkpoint, verify_backup
from training.checkpoint import read_checkpoint
from training.consistency import file_digest, verify_commit
from training.run import run_training, main
from test_training import fixture_data, tiny_model


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def config_for(tmp_path):
    return replace(fixture_data(tmp_path), economical=True,
                   work_dir=str(tmp_path / 'local'), epochs=4, device='cpu')


def validation_for(config):
    def validate(*args, **kwargs):
        score = [.6, .9, .7, .9][torch.initial_seed() - config.seed]
        return {'loss': 1., 'macro_f1': score, 'accuracy': score}
    return validate


def durable_for(run):
    return Path(json.loads((run / 'storage.json').read_text())['durable_directory'])


@pytest.mark.parametrize('phase,fail_epoch', [('manifest', 1), ('metadata', 1), ('metadata', 2)])
def test_destroy_runtime_and_restore_matches_reference(tmp_path, phase, fail_epoch):
    import training.backup as backup
    import training.run as runner
    config = config_for(tmp_path)
    recorded = []
    def observed(run, destination):
        sync_checkpoint(run, destination)
        state = verify_backup(destination)
        recorded.append(state['epoch'])
        assert file_digest(run / 'last.pt') == file_digest(destination / 'last.pt')
    with patch('training.run.build_model', side_effect=tiny_model), patch('training.run.validate', side_effect=validation_for(config)):
        with patch('training.run.sync_checkpoint', side_effect=observed):
            reference = run_training(config)
        assert recorded == [0, 1, 2, 3]
        write = backup.write_json
        sync = runner.sync_results
        def interrupted_write(path, record):
            if phase == 'manifest' and path.name == 'backup.json' and record['completed_epoch'] == fail_epoch + 1:
                raise OSError('runtime destroyed')
            return write(path, record)
        def interrupted_sync(run, destination, **kwargs):
            if phase == 'metadata' and (run / 'last.pt').exists() and read_checkpoint(run / 'last.pt')['epoch'] == fail_epoch:
                raise OSError('runtime destroyed')
            return sync(run, destination, **kwargs)
        with patch('training.backup.write_json', side_effect=interrupted_write), patch('training.run.sync_results', side_effect=interrupted_sync):
            with pytest.raises((OSError, BackupError), match='runtime destroyed'):
                run_training(config)
        local = next(p for p in (Path(config.work_dir) / config.model).iterdir() if p != reference)
        durable = durable_for(local)
        original = read_checkpoint(local / 'last.pt')
        shutil.rmtree(local)  # Simulate loss of Colab /content, not production cleanup.
        restored = restore_run(durable)
        equal(read_checkpoint(restored / 'last.pt'), original)
        assert verify_backup(durable)['epoch'] == fail_epoch
        verify_commit(restored)
        run_training(config, restored)
    expected, actual = read_checkpoint(reference / 'last.pt'), read_checkpoint(restored / 'last.pt')
    for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'best_selection'):
        equal(expected[key], actual[key])
    equal([{k:v for k,v in r.items() if k != 'seconds'} for r in expected['history']],
          [{k:v for k,v in r.items() if k != 'seconds'} for r in actual['history']])
    equal(torch.load(reference / 'best_weights.pt', weights_only=True),
          torch.load(restored / 'best_weights.pt', weights_only=True))
    assert actual['best_selection']['epoch'] == 2
    verify_commit(durable)
    assert len(list(durable.glob('*.pt'))) == 2  # last + compact best, no epoch archive
    assert not list(durable.glob('.*.tmp'))
    with pytest.raises(FileExistsError):
        restore_run(durable)


@pytest.mark.parametrize('failure', ['disk_full', 'unavailable', 'bad_sha', 'bad_epoch'])
def test_failed_staging_preserves_previous_backup(tmp_path, failure):
    import training.backup as backup
    config = config_for(tmp_path)
    with patch('training.run.build_model', side_effect=tiny_model), patch('training.run.validate', side_effect=validation_for(config)):
        # Stop after the next local checkpoint commits, keeping first backup intact.
        def stopped(run, destination):
            if read_checkpoint(run / 'last.pt')['epoch'] > 0:
                raise BackupError('stop')
            sync_checkpoint(run, destination)
        with patch('training.run.sync_checkpoint', side_effect=stopped):
            with pytest.raises(BackupError, match='stop'):
                run_training(config)
        run = next((Path(config.work_dir) / config.model).iterdir())
        durable = durable_for(run)
        previous = (durable / 'last.pt').read_bytes()
        manifest = (durable / 'backup.json').read_bytes()
        copying = backup.shutil.copyfile
        reading = backup.read_checkpoint
        def broken_copy(source, target):
            if failure in ('disk_full', 'unavailable'):
                Path(target).write_bytes(b'partial')
                raise OSError('No space left on device')
            copying(source, target)
            if failure == 'bad_sha':
                with Path(target).open('ab') as stream:
                    stream.write(b'corruption')
        def wrong_epoch(path):
            state = reading(path)
            if failure == 'bad_epoch' and path.name.endswith('.tmp'):
                state['epoch'] += 1
            return state
        with patch('training.backup.shutil.copyfile', side_effect=broken_copy), patch('training.backup.read_checkpoint', side_effect=wrong_epoch):
            with pytest.raises(BackupError):
                sync_checkpoint(run, durable)
        assert (durable / 'last.pt').read_bytes() == previous
        assert (durable / 'backup.json').read_bytes() == manifest
        assert (run / 'last.pt').is_file()
        recovered_previous = restore_run(durable, tmp_path / 'previous-backup')
        assert read_checkpoint(recovered_previous / 'last.pt')['epoch'] == 0
        assert not list(durable.glob('.*.tmp'))
        run_training(config, run)
        assert verify_backup(durable)['epoch'] == 3


def test_corruption_foreign_run_and_legacy_without_manifest(tmp_path):
    config = replace(config_for(tmp_path), smoke=True)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(config)
    durable = durable_for(run)
    # Existing economical checkpoint before backup feature: no manifest/run_id.
    state = read_checkpoint(run / 'last.pt')
    state.pop('run_id')
    torch.save(state, run / 'last.pt')
    (durable / 'backup.json').unlink()
    (durable / 'backup_pending.json').unlink()
    sync_checkpoint(run, durable)
    verify_backup(durable)
    assert (durable / 'backup.json').exists()
    original = (durable / 'last.pt').read_bytes()
    (durable / 'last.pt').write_bytes(b'broken checkpoint')
    with pytest.raises(BackupError, match='SHA-256'):
        restore_run(durable, tmp_path / 'recovery')
    (durable / 'last.pt').write_bytes(original)
    record = json.loads((durable / 'backup.json').read_text())
    record['run_id'] = 'foreign'
    (durable / 'backup.json').write_text(json.dumps(record))
    with pytest.raises(BackupError, match='different run'):
        restore_run(durable, tmp_path / 'recovery')


def test_restore_cli_prints_resume_command(tmp_path, capsys):
    config = replace(config_for(tmp_path), smoke=True)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(config)
    durable = durable_for(run)
    with patch('sys.argv', ['training.run', '--restore-from', str(durable), '--work-dir', str(tmp_path / 'recovered')]):
        main()
    assert 'Resume: python -m training.run --resume' in capsys.readouterr().out


def test_parent_source_compatibility_and_foreign_checkpoint(tmp_path):
    from training.run import compatible_sources, provenance
    current = provenance()['source_sha256']
    previous = {**current,
        'training/run.py': '3fe4afdba0d3692e53cd76c2a6555ab71d44c33e9354b17058e02b8f62d774c8',
        'training/checkpoint.py': 'ed12f2d077b2fd3c42f7bc887a72d4f6e091c2d096bd5e7d32a4f31d9f30596c'}
    previous.pop('training/backup.py')
    assert compatible_sources(previous, current)
    assert not compatible_sources({**previous, 'data/visdrone.py': 'unknown'}, current)
    config = replace(config_for(tmp_path), smoke=True)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(config)
    durable = durable_for(run)
    state = read_checkpoint(durable / 'last.pt')
    state['run_id'] = 'foreign-checkpoint'
    torch.save(state, durable / 'last.pt')
    record = json.loads((durable / 'backup.json').read_text())
    record['sha256'] = file_digest(durable / 'last.pt')
    record['file_size_bytes'] = (durable / 'last.pt').stat().st_size
    (durable / 'backup.json').write_text(json.dumps(record))
    with pytest.raises(BackupError, match='different run'):
        verify_backup(durable)
