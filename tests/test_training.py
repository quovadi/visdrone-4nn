"""Synthetic only: no VisDrone, pretrained download, GPU or long training."""
from dataclasses import replace
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.config import new_config
from training.models import build_model, check_model, make_transforms
from training.data import make_datasets, epoch_loaders, load_records
from training.engine import train_epoch
from training.validation import ClassificationMetrics, validate
from training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from training.run import run_training, seed_everything, new_run_directory

# Avoid oversubscribing small synthetic CPU work.
torch.set_num_threads(1)


@pytest.mark.parametrize('name', ['convnext_base', 'convnext_nano', 'vit_base', 'tinyvit'])
def test_real_model_registry_and_forward_without_weights(name):
    config = new_config(name)
    weights = check_model(config.model_id)
    assert weights['hf_hub_id'] and weights['num_classes'] == 1000
    model, data_config = build_model(config.model_id, pretrained=False)
    train, evaluation = make_transforms(data_config)
    image = Image.new('RGB', (11, 7), (255, 128, 0))
    x = evaluation(image)
    expected = (torch.tensor([1., 128/255, 0.]) - torch.tensor(data_config['mean'])) / torch.tensor(data_config['std'])
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert torch.allclose(x[:, 112, 112], expected, atol=1e-5)  # exactly one normalization
    model.eval()
    with torch.inference_mode():
        assert model(x.unsqueeze(0)).shape == (1, 10)
    assert torch.isfinite(train(image)).all()


def test_pretrained_flag_and_no_substitution():
    model = torch.nn.Linear(1, 10)
    with patch('training.models.timm.create_model', return_value=model) as create, \
            patch('training.models.resolve_model_data_config', return_value={'input_size': (3, 224, 224)}):
        build_model('convnext_nano.d1h_in1k')
        create.assert_called_once_with('convnext_nano.d1h_in1k', pretrained=True, num_classes=10)
    with pytest.raises(ValueError, match='No substitution'):
        check_model('tiny_vit_5m_224.nonexistent_weights')


def fixture_data(tmp_path):
    root, indices = tmp_path / 'images', tmp_path / 'data-v1'
    indices.mkdir()
    for split, name in [('train', 't'), ('validation', 'v')]:
        folder = root / split / 'images'
        folder.mkdir(parents=True)
        Image.new('RGB', (20, 10), 'red').save(folder / f'{name}.png')
        records = [{'image': f'{split}/images/{name}.png', 'box': [0, 0, 2, 2],
                    'label': i % 10, 'line': i+1} for i in range(11)]
        (indices / f'{split}.json').write_text(json.dumps(records))
    # holdout deliberately absent: training must not need it.
    return new_config('tinyvit', root=str(root), indices=str(indices),
                      output=str(tmp_path / 'results'), batch_size=4, workers=0, epochs=2)


def tiny_model(*args, **kwargs):
    return (torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(3*8*8, 10)),
            {'input_size': (3, 8, 8), 'mean': (0.5,)*3, 'std': (0.5,)*3,
             'interpolation': 'bicubic', 'crop_pct': 1.0})


def test_batch_subset_boxes_and_no_writes_to_indices(tmp_path):
    config = fixture_data(tmp_path)
    before = {p.name: p.read_bytes() for p in Path(config.indices).iterdir()}
    transforms = make_transforms(tiny_model()[1])
    train, validation, metadata = make_datasets(replace(config, train_size=5), *transforms)
    assert len(train) == 5 and len(validation) == 11
    assert train.records[0].box == (0, 0, 2, 2)  # small objects retained
    x, y = next(iter(epoch_loaders(config, train, validation, 0)[0]))
    assert x.shape == (4, 3, 8, 8) and y.dtype == torch.int64
    assert y.min() >= 0 and y.max() < 10
    again, _, other = make_datasets(replace(config, train_size=5), *transforms)
    assert train.records == again.records and metadata == other
    smoke, val_smoke, _ = make_datasets(replace(config, smoke=True), *transforms)
    assert len(smoke) == len(val_smoke) == 8
    assert before == {p.name: p.read_bytes() for p in Path(config.indices).iterdir()}
    raw = json.loads((Path(config.indices) / 'validation.json').read_text())
    raw[0]['image'] = 'validation/images/t.png'
    (Path(config.indices) / 'validation.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='overlap'):
        make_datasets(config, *transforms)


def test_metrics_fixed_ten_classes_and_weighted_loss():
    metrics = ClassificationMetrics()
    logits = torch.full((3, 10), -10.)
    logits[range(3), [0, 1, 0]] = 10
    metrics.update(logits[:2], torch.tensor([0, 1]), 2.0)
    metrics.update(logits[2:], torch.tensor([1]), 5.0)
    result = metrics.compute()
    assert result['loss'] == 3.0
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['macro_f1'] == pytest.approx((2/3 + 2/3) / 10)
    assert result['support'] == [1, 2] + [0]*8
    with pytest.raises(ValueError):
        ClassificationMetrics().compute()


def components():
    model, _ = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, optimizer, scheduler, scaler


def test_train_validate_and_checkpoint_resume_next_step(tmp_path):
    seed_everything(42)
    model, opt, scheduler, scaler = components()
    loader = DataLoader(TensorDataset(torch.randn(5, 3, 8, 8), torch.arange(5)), batch_size=3)
    before = model[-1].weight.detach().clone()
    result = train_epoch(model, loader, opt, torch.nn.CrossEntropyLoss(), scaler, torch.device('cpu'))
    assert result['samples'] == 5 and np.isfinite(result['loss'])
    assert not torch.equal(before, model[-1].weight)
    val = validate(model, loader, torch.nn.CrossEntropyLoss(), torch.device('cpu'))
    assert val['samples'] == 5
    scheduler.step()
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, opt, scheduler, scaler, 0, .2, {'model': 'test'}, [], {}, {})
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))
    state = read_checkpoint(path)
    model2, opt2, scheduler2, scaler2 = components()
    assert restore_checkpoint(state, model2, opt2, scheduler2, scaler2) == (1, .2)
    assert expected_rng[:2] == (random.random(), np.random.rand())
    assert torch.equal(expected_rng[2], torch.rand(1))
    assert opt2.param_groups[0]['lr'] == opt.param_groups[0]['lr']
    assert scheduler2.state_dict() == scheduler.state_dict()
    assert scaler2.state_dict() == scaler.state_dict()
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
    # Momentum/RNG restoration must reproduce a subsequent optimization epoch.
    seed_everything(77)
    train_epoch(model, loader, opt, torch.nn.CrossEntropyLoss(), scaler, torch.device('cpu'))
    seed_everything(77)
    train_epoch(model2, loader, opt2, torch.nn.CrossEntropyLoss(), scaler2, torch.device('cpu'))
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
    assert not list(tmp_path.glob('*.tmp'))


def test_end_to_end_interruption_and_immutable_run(tmp_path):
    config = fixture_data(tmp_path)
    with patch('training.run.build_model', side_effect=tiny_model):
        uninterrupted = run_training(config)
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('synthetic disconnect')
            return train_epoch(*args, **kwargs)
        with patch('training.run.train_epoch', side_effect=interrupt):
            with pytest.raises(RuntimeError, match='synthetic disconnect'):
                run_training(config)
        runs = [p for p in (Path(config.output) / config.model).iterdir() if p != uninterrupted]
        interrupted = runs[0]
        run_training(config, interrupted)
        a, b = read_checkpoint(uninterrupted / 'last.pt'), read_checkpoint(interrupted / 'last.pt')
        assert a['epoch'] == b['epoch'] == 1
        for key in a['model']:
            assert torch.equal(a['model'][key], b['model'][key])
        assert [r['validation'] for r in a['history']] == [r['validation'] for r in b['history']]
        history = json.loads((interrupted / 'metrics.json').read_text())
        best = read_checkpoint(interrupted / 'best.pt')
        assert best['best_f1'] == max(r['validation']['macro_f1'] for r in history)
        assert (interrupted / 'source.zip').is_file()
        with pytest.raises(ValueError, match='configuration differs'):
            run_training(replace(config, epochs=3), interrupted)
        index_path = Path(config.indices) / 'train.json'
        index_path.write_text(index_path.read_text() + '\n')
        with pytest.raises(ValueError, match='indices'):
            run_training(config, interrupted)


def test_guards_and_smoke(tmp_path):
    config = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        new_run_directory(replace(config, output=config.indices))
    with pytest.raises(ValueError):
        new_config('tinyvit', batch_size=0)
    with patch('training.run.build_model', side_effect=tiny_model):
        run = run_training(replace(config, smoke=True))
        assert len(json.loads((run / 'metrics.json').read_text())) == 1
        assert json.loads((run / 'summary.json').read_text())['smoke'] is True
    path = Path(config.indices) / 'train.json'
    raw = json.loads(path.read_text())
    raw[0]['box'] = [0, 0, 0, 1]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='Invalid'):
        load_records(path)


def test_atomic_failure_preserves_previous_file(tmp_path):
    from training.checkpoint import atomic_save
    path = tmp_path / 'last.pt'
    path.write_bytes(b'previous completed checkpoint')
    def failing_writer(temporary):
        temporary.write_bytes(b'incomplete')
        raise OSError('synthetic disk failure')
    with pytest.raises(OSError, match='disk failure'):
        atomic_save(path, failing_writer)
    assert path.read_bytes() == b'previous completed checkpoint'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP requires GPU (Colab check)')
def test_cuda_amp_scaler_checkpoint(tmp_path):
    seed_everything(42)
    model, _ = tiny_model()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler('cuda')
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 8, 8), torch.arange(4)), batch_size=2)
    result = train_epoch(model, loader, optimizer, torch.nn.CrossEntropyLoss(), scaler,
                         torch.device('cuda'), amp=True)
    assert np.isfinite(result['loss']) and scaler.state_dict()['scale'] > 0
    scheduler.step()
    save_checkpoint(tmp_path / 'last.pt', model, optimizer, scheduler, scaler,
                    0, 0., {}, [], {}, {})
    new_model, _ = tiny_model()
    new_model.cuda()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=2)
    new_scaler = torch.amp.GradScaler('cuda')
    restore_checkpoint(read_checkpoint(tmp_path / 'last.pt'), new_model, new_optimizer,
                       new_scheduler, new_scaler)
    assert new_scaler.state_dict() == scaler.state_dict()
    assert all(values['exp_avg'].is_cuda for values in new_optimizer.state.values())
    # This must also execute successfully with the restored GPU Adam buffers.
    train_epoch(new_model, loader, new_optimizer, torch.nn.CrossEntropyLoss(), new_scaler,
                torch.device('cuda'), amp=True)
