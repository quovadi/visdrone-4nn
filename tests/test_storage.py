"""Synthetic storage tests; no real data, downloads, or long training."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from training.checkpoint import export_weights, load_weights, read_checkpoint
from training.run import run_training, compatible_sources, provenance
from training.storage import durable_directory, sync_results
from test_training import fixture_data, tiny_model, train_epoch


def test_fp32_export_safe_load_and_buffers(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).half().eval()
    path = tmp_path / 'weights.pt'
    export_weights(path, model)
    state = torch.load(path, weights_only=True)
    assert all(t.dtype == torch.float32 for t in state.values() if t.is_floating_point())
    assert state['1.num_batches_tracked'].dtype == torch.int64
    restored = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).eval()
    load_weights(path, restored)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, model.state_dict()[key].to(value.dtype))
    assert not any(k in state for k in ('optimizer', 'scheduler', 'rng', 'config'))


def test_failed_weight_export_preserves_previous(tmp_path):
    path = tmp_path / 'best_weights.pt'
    model, _ = tiny_model()
    export_weights(path, model)
    before = path.read_bytes()
    def broken_save(state, temporary):
        Path(temporary).write_bytes(b'incomplete')
        raise OSError('disk full')
    with patch('training.checkpoint.torch.save', side_effect=broken_save):
        with pytest.raises(OSError, match='disk full'):
            export_weights(path, model)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def economical_config(tmp_path):
    config = fixture_data(tmp_path)
    return replace(config, economical=True, work_dir=str(tmp_path / 'local'))


def test_economical_resume_and_durable_allowlist(tmp_path):
    config = economical_config(tmp_path)
    before = {p.name: p.read_bytes() for p in Path(config.indices).iterdir()}
    with patch('training.run.build_model', side_effect=tiny_model):
        complete = run_training(config)
        default = run_training(replace(config, economical=False, work_dir=None,
                                       output=str(tmp_path / 'default-results')))
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('disconnect')
            return train_epoch(*args, **kwargs)
        with patch('training.run.train_epoch', side_effect=interrupted):
            with pytest.raises(RuntimeError, match='disconnect'):
                run_training(config)
        partial = next(p for p in (Path(config.work_dir) / config.model).iterdir() if p != complete)
        run_training(config, partial)
    a, b = read_checkpoint(complete / 'last.pt'), read_checkpoint(partial / 'last.pt')
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key])
    for run in (complete, partial):
        durable = Path(json.loads((run / 'storage.json').read_text())['durable_directory'])
        assert (run / 'last.pt').is_file()
        assert not (run / 'best.pt').exists()
        assert (durable / 'last.pt').read_bytes() == (run / 'last.pt').read_bytes()
        assert not list(durable.glob('best.pt'))
        for name in ('best_weights.pt', 'config.json', 'data.json', 'preprocessing.json',
                     'provenance.json', 'metrics.json', 'summary.json', 'source.zip', 'environment.txt'):
            assert (durable / name).read_bytes() == (run / name).read_bytes()
        expected = read_checkpoint(default / 'best.pt')['model']
        compact = torch.load(durable / 'best_weights.pt', weights_only=True)
        for key in compact:
            assert torch.equal(compact[key], expected[key])
        model, _ = tiny_model()
        load_weights(durable / 'best_weights.pt', model)
    assert before == {p.name: p.read_bytes() for p in Path(config.indices).iterdir()}


def test_owner_collision_and_failed_copy(tmp_path):
    config = economical_config(tmp_path)
    local = Path(config.work_dir) / config.model / 'fixed-run'
    local.mkdir(parents=True)
    durable = durable_directory(config, local)
    (local / 'best_weights.pt').write_bytes(b'new weights')
    (durable / 'best_weights.pt').write_bytes(b'previous weights')
    def failed_copy(source, temporary):
        Path(temporary).write_bytes(b'partial')
        raise OSError('Drive full')
    with patch('training.storage.shutil.copyfile', side_effect=failed_copy):
        with pytest.raises(OSError, match='Drive full'):
            sync_results(local, durable)
    assert (durable / 'best_weights.pt').read_bytes() == b'previous weights'
    assert not list(durable.glob('.*.tmp'))
    with pytest.raises(FileExistsError):
        durable_directory(config, local)
    (durable / 'storage.json').write_text('{"run_id":"another run"}')
    with pytest.raises(ValueError, match='different run'):
        sync_results(local, durable)
    assert (durable / 'best_weights.pt').read_bytes() == b'previous weights'


def test_legacy_full_checkpoint_and_default_mode(tmp_path):
    config = fixture_data(tmp_path)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(config)
        best = read_checkpoint(run / 'best.pt')
        # Simulate the old config schema and missing compact export.
        state = read_checkpoint(run / 'last.pt')
        state['config'].pop('economical')
        state['config'].pop('work_dir')
        torch.save(state, run / 'last.pt')
        (run / 'best_weights.pt').unlink()
        run_training(config, run)
    compact = torch.load(run / 'best_weights.pt', weights_only=True)
    for key in compact:
        assert torch.equal(compact[key], best['model'][key])
    assert (run / 'last.pt').is_file() and (run / 'best.pt').is_file()
    assert not (run / 'storage.json').exists()


def test_real_legacy_fingerprints_and_invalid_paths(tmp_path):
    current = provenance()['source_sha256']
    previous = {**current,
        'training/config.py': '513e052e0aa1c7fc96b8717ecb7944e7f9bb973ab5343ccb7b040b42e035f691',
        'training/checkpoint.py': 'a55f7c3ef91d6e2305ede6aa15289fe799763d64502c1fd688b78ff250a73a35',
        'training/run.py': 'a16b0554002caa15801bfbee694efbddcdeb217d939fdbdbca2860baf02b7acd'}
    previous.pop('training/storage.py')
    assert compatible_sources(previous, current)
    assert not compatible_sources({**previous, 'data/visdrone.py': 'unknown'}, current)
    config = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='work_dir'):
        replace(config, economical=True)
    with pytest.raises(ValueError, match='work_dir'):
        replace(config, work_dir='/content/runs')
    from training.run import new_run_directory
    with pytest.raises(ValueError, match='separate'):
        new_run_directory(replace(config, economical=True, work_dir=config.indices))


def test_cli_old_config_defaults(tmp_path):
    from training.run import main
    config = fixture_data(tmp_path).to_dict()
    config.pop('economical')
    config.pop('work_dir')
    saved = tmp_path / 'old-run'
    saved.mkdir()
    (saved / 'config.json').write_text(json.dumps(config))
    with patch('sys.argv', ['training.run', '--resume', str(saved)]), \
            patch('training.run.run_training') as train:
        main()
    restored, resume = train.call_args.args
    assert restored.economical is False and restored.work_dir is None and resume == saved


def test_drive_failure_keeps_local_resume_state(tmp_path):
    config = economical_config(tmp_path)
    with patch('training.run.build_model', side_effect=tiny_model):
        def interrupted_export(run, destination, **kwargs):
            if (run / 'last.pt').is_file():
                raise OSError('Drive unavailable')
            sync_results(run, destination, **kwargs)
        with patch('training.run.sync_results', side_effect=interrupted_export):
            with pytest.raises(OSError, match='Drive unavailable'):
                run_training(config)
        run = next((Path(config.work_dir) / config.model).iterdir())
        saved = read_checkpoint(run / 'last.pt')
        assert saved['epoch'] == 0
        assert saved['optimizer']['state'] and saved['scheduler'] and saved['rng']
        run_training(config, run)
    durable = Path(json.loads((run / 'storage.json').read_text())['durable_directory'])
    assert (durable / 'best_weights.pt').read_bytes() == (run / 'best_weights.pt').read_bytes()
    assert len(json.loads((durable / 'metrics.json').read_text())) == config.epochs
