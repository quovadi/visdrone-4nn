"""CPU-only mathematical and end-to-end tests. No downloaded weights/data."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from timm.models import get_pretrained_cfg

from attacks.core import AttackConfig, generate, project
from attacks.data import selected_validation, dataset
from attacks.evaluation import evaluate, main, parse_epsilon
from attacks.metrics import PairedMetrics
from attacks.models import load_classifier
from attacks.preprocessing import PixelModel, rgb_transform
from attacks.results import create_run, read_predictions, summarize, write_chunk
from training.models import make_transforms
from training.consistency import file_digest


torch.set_num_threads(1)


def analytic_model():
    layer = nn.Linear(12, 2, bias=False)
    with torch.no_grad():
        layer.weight[0].fill_(1)
        layer.weight[1].zero_()
    return nn.Sequential(nn.Flatten(), layer)


@pytest.mark.parametrize('epsilon', [0., 1/255, 2/255, 4/255, 8/255, .1])
@pytest.mark.parametrize('name', ['fgsm', 'pgd'])
def test_attack_bounds_and_zero(name, epsilon):
    model = analytic_model()
    x = torch.rand(3, 3, 2, 2)
    x[0, :, 0, 0] = 0
    x[1, :, 0, 0] = 1
    original = x.clone()
    checkpoints = []
    output = generate(model, x, torch.zeros(3, dtype=torch.int64), AttackConfig(name, epsilon, steps=3),
                      observer=lambda step, value: checkpoints.append((step, value)))
    assert output.shape == x.shape and output.dtype == torch.float32
    assert not output.requires_grad and torch.equal(x, original)
    for value in [output] + [value for step, value in checkpoints]:
        assert torch.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        assert float((value - original).abs().max()) <= epsilon + 1e-6
    if epsilon == 0:
        assert torch.equal(output, x) and not checkpoints


def test_fgsm_analytic_direction_and_loss():
    model = analytic_model()
    x = torch.full((1, 3, 2, 2), .5)
    y = torch.tensor([0])
    output = generate(model, x, y, AttackConfig('fgsm', .1))
    assert torch.allclose(output, torch.full_like(x, .4))
    assert F.cross_entropy(model(output), y) > F.cross_entropy(model(x), y)


def test_pgd_iterations_random_start_projection_and_model_preservation():
    model = analytic_model()
    model.train()
    model[0].eval()  # Mixed per-module flags must survive.
    flags = [module.training for module in model.modules()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    random_state = torch.get_rng_state().clone()
    x = torch.full((1, 3, 2, 2), .5)
    labels = torch.tensor([0])
    stages = []
    first = generate(model, x, labels, AttackConfig('pgd', .2, steps=3), sample_seeds=[17],
                     observer=lambda step, value: stages.append((step, value)))
    second = generate(model, x, labels, AttackConfig('pgd', .2, steps=3), sample_seeds=[17])
    other = generate(model, x, labels, AttackConfig('pgd', .2, steps=3), sample_seeds=[19])
    assert torch.equal(first, second) and not torch.equal(first, other)
    assert [step for step, value in stages] == [0, 1, 2, 3]
    for step, value in stages[1:]:
        expected = (stages[step-1][1] - .05).clamp(.3, .7)
        assert torch.allclose(value, expected, atol=1e-7)
    assert not torch.equal(stages[1][1], stages[3][1])
    assert torch.equal(random_state, torch.get_rng_state())
    assert flags == [module.training for module in model.modules()]
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    for parameter in model.parameters():
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
    origin = torch.tensor([[[[.01, .99]], [[.2, .4]], [[.8, 0.]]]])
    candidate = torch.tensor([[[[-2., 5.]], [[.5, -.1]], [[4., 8.]]]])
    assert torch.allclose(project(candidate, origin, .1), torch.tensor([[[[0., 1.]], [[.3, .3]], [[.9, .1]]]]))


def test_diagnostics_and_flags_restored_on_failure():
    class Broken(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
        def forward(self, x):
            values = x.flatten(1).sum(1) * (float('nan') if self.kind == 'nan' else 0)
            return torch.stack((values, values), dim=1)
    events = []
    model = Broken('zero').train()
    x = torch.ones(1, 3, 2, 2) * .5
    y = torch.tensor([0])
    output = generate(model, x, y, AttackConfig('fgsm', .1), diagnostic=events.append)
    assert torch.equal(x, output) and events[0]['kind'] == 'zero_input_gradient'
    model.kind = 'nan'
    with pytest.raises(FloatingPointError):
        generate(model, x, y, AttackConfig('fgsm', .1), diagnostic=events.append)
    assert model.training and events[-1]['kind'] == 'nonfinite_logits'
    with pytest.raises(ValueError, match='RGB'):
        generate(model, x - 1, y, AttackConfig('fgsm', .1))


MODEL_IDS = ['convnext_base.fb_in1k', 'convnext_nano.d1h_in1k',
             'vit_base_patch16_224.augreg_in1k', 'tiny_vit_5m_224.dist_in22k_ft_in1k']


def data_config(model_id=MODEL_IDS[1]):
    weights = get_pretrained_cfg(model_id).to_dict()
    return {key: weights[key] for key in ('input_size', 'interpolation', 'mean', 'std', 'crop_pct', 'crop_mode')}


@pytest.mark.parametrize('model_id', MODEL_IDS)
def test_preprocessing_exactly_matches_timm_once_and_gradient(model_id):
    cfg = data_config(model_id)
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (31, 67, 3), dtype=np.uint8))
    rgb = rgb_transform(cfg)(image)
    _, baseline = make_transforms(cfg)
    normalized = baseline(image)
    assert rgb.shape == (3, 224, 224) and 0 <= rgb.min() <= rgb.max() <= 1
    pixel_identity = PixelModel(nn.Identity(), cfg)
    assert torch.equal(pixel_identity(rgb.unsqueeze(0))[0], normalized)
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 10))
    pixel_model = PixelModel(classifier, cfg)
    x = rgb.unsqueeze(0).requires_grad_()
    loss = F.cross_entropy(pixel_model(x), torch.tensor([0]))
    gradient = torch.autograd.grad(loss, x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    output = generate(pixel_model, x.detach(), torch.tensor([0]), AttackConfig('pgd', 0.))
    assert torch.equal(pixel_model(output), classifier(normalized.unsqueeze(0)))


def test_metrics_expected_values_and_undefined_asr():
    metrics = PairedMetrics()
    metrics.update([0, 1, 1, 0], [0, 1, 2, 3], [1, 1, 1, 0])
    result = metrics.compute()
    assert result['clean_accuracy'] == .5 and result['adversarial_accuracy'] == .75
    assert result['asr'] == .5 and result['attack_success_count'] == 1
    assert result['clean_correct_per_class'] == [1, 1] + [0]*8
    assert result['clean_macro_f1'] == pytest.approx((2/3 + 2/3)/10)
    assert result['adversarial_macro_f1'] == pytest.approx((2/3 + 4/5)/10)
    assert result['macro_f1_drop'] < 0  # Not artificially forced to be nonnegative.
    other = PairedMetrics()
    other.update([0, 1], [1, 0], [0, 1])
    assert other.compute()['asr'] is None and not other.compute()['asr_defined']
    assert other.compute()['adversarial_accuracy'] == 1.


def evaluation_fixture(tmp_path):
    root = tmp_path / 'images'
    index = tmp_path / 'data-v1' / 'validation.json'
    index.parent.mkdir()
    records = []
    for i in range(6):
        name = f'VisDrone2019-DET-val/images/{i}.png'
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (17, 23), (150 if i % 2 == 0 else 90,)*3).save(path)
        records.append({'image': name, 'box': [1, 2, 15, 20], 'label': i % 3, 'line': 1})
    index.write_text(json.dumps(records))
    cfg = data_config()
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 10))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.fill_(-10)
        model[-1].weight[0].fill_(1)
        model[-1].weight[1].fill_(-1)
        model[-1].bias[:2].zero_()
    checkpoint = tmp_path / 'weights.pt'
    torch.save(model.state_dict(), checkpoint)
    chosen, selected = selected_validation(index, 42, 6)
    ds = dataset(root, chosen, rgb_transform(cfg))
    info = {'model': 'convnext_nano', 'model_id': MODEL_IDS[1], 'variant': 'baseline-v1',
            'checkpoint': str(checkpoint), 'checkpoint_sha256': file_digest(checkpoint),
            'validation_index': str(index)}
    return root, index, model, cfg, selected, ds, info


def test_evaluation_roundtrip_zero_control_and_no_overwrite(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    untouched = index.read_bytes()
    pixels = PixelModel(model, cfg)
    pixels.train()
    output = tmp_path / 'results'
    run = evaluate(pixels, ds, selected, info, cfg, AttackConfig('fgsm', 0.), output,
                   batch_size=2, examples=1, run_id='fixed')
    assert pixels.training  # Evaluation also restores caller flags.
    rows = read_predictions(run)
    result = json.loads((run / 'metrics.json').read_text())
    assert summarize(rows) == result
    assert result['samples'] == 6 and result['asr'] == 0.
    assert all(row['clean_prediction'] == row['adversarial_prediction'] and row['linf'] == 0 for row in rows)
    assert all(row['attack_success'] is None for row in rows if not row['clean_correct'])
    assert (run / 'examples' / '0000000-delta.pt').is_file()
    assert torch.load(run / 'examples' / '0000000-delta.pt', weights_only=True).abs().sum() == 0
    saved = json.loads((run / 'config.json').read_text())
    assert saved['checkpoint_sha256'] == file_digest(Path(info['checkpoint']))
    assert saved['selection']['index_sha256'] == file_digest(index)
    assert saved['provenance']['git_commit'] and saved['provenance']['attack_source_sha256']
    assert json.loads((run / 'status.json').read_text())['state'] == 'completed'
    assert index.read_bytes() == untouched
    assert read_predictions(run) == rows
    with pytest.raises(FileExistsError):
        evaluate(pixels, ds, selected, info, cfg, AttackConfig('fgsm', 0.), output, run_id='fixed')
    assert read_predictions(run) == rows


def test_partial_results_and_diagnostics_survive_failure(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    import attacks.evaluation as module
    original = module.write_chunk
    def interrupted(run, number, rows):
        if number == 1:
            raise OSError('synthetic interruption')
        return original(run, number, rows)
    with patch('attacks.evaluation.write_chunk', side_effect=interrupted):
        with pytest.raises(OSError, match='synthetic interruption'):
            evaluate(PixelModel(model, cfg), ds, selected, info, cfg,
                     AttackConfig('pgd', 4/255, steps=2), tmp_path / 'results', batch_size=2, run_id='partial')
    run = tmp_path / 'results' / 'partial'
    rows = read_predictions(run)
    assert len(rows) == 2
    assert summarize(rows) == json.loads((run / 'metrics.json').read_text())
    assert json.loads((run / 'status.json').read_text())['state'] == 'failed'
    assert not list((run / 'predictions').glob('.*.tmp'))


def test_selection_stability_and_validation_only(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    records, a = selected_validation(index, 19, 3)
    _, b = selected_validation(index, 19, 3)
    assert a == b and len(a['objects']) == 3
    assert a['objects'] == sorted(a['objects'], key=lambda obj: obj['index_row'])
    assert len({obj['sample_id'] for obj in a['objects']}) == 3
    assert all(record.box == (1, 2, 15, 20) for record in records)
    for forbidden in ('train.json', 'holdout.json'):
        with pytest.raises(ValueError, match='validation.json'):
            selected_validation(index.with_name(forbidden), 42)


@pytest.mark.parametrize('kind', ['compact', 'full'])
def test_load_only_trained_weights_and_metadata(tmp_path, kind):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    run = tmp_path / 'training-run'
    run.mkdir()
    config = {'model': 'convnext_nano', 'model_id': MODEL_IDS[1], 'train_size': None, 'lr': 1e-4}
    metadata = {'config': config, 'preprocessing': {'data_config': cfg},
                'data': {'resolved_data_config': cfg}, 'provenance': {'git_commit': 'synthetic'}}
    for name, value in metadata.items():
        (run / f'{name}.json').write_text(json.dumps(value))
    path = Path(info['checkpoint'])
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    if kind == 'full':
        torch.save({'format_version': 1, 'model': expected, 'config': config, 'data_metadata': metadata['data']}, path)
        with pytest.raises(ValueError, match='trusted-full'):
            with patch('attacks.models.build_model', return_value=(model, cfg)):
                load_classifier(run, path, 'baseline-v1', kind)
    with torch.no_grad():
        model[-1].weight.fill_(99)
    with patch('attacks.models.build_model', return_value=(model, cfg)) as build:
        loaded, recorded, metadata_info = load_classifier(run, path, 'baseline-v1', kind, trusted_full=kind == 'full')
        build.assert_called_once_with(MODEL_IDS[1], pretrained=False)
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, expected[name])
    assert metadata_info['checkpoint_sha256'] == file_digest(path)
    config.update(model='convnext_base', train_size=50000, lr=3e-5)
    (run / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='convnext-base-v2'):
        load_classifier(run, path, 'baseline-v1', kind)


def test_cli_sequential_epsilon_synthetic_smoke(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    args = ['attacks.evaluation', '--model-run', str(tmp_path / 'model-run'), '--checkpoint', info['checkpoint'],
            '--variant', 'baseline-v1', '--root', str(root), '--validation-index', str(index),
            '--output', str(tmp_path / 'results'), '--attack', 'pgd', '--epsilon', '0', '1/255',
            '--subset-size', '3', '--batch-size', '2']
    with patch('sys.argv', args), patch('attacks.evaluation.load_classifier', return_value=(model, cfg, info)):
        main()
    runs = list((tmp_path / 'results').iterdir())
    assert len(runs) == 2
    configs = [json.loads((run / 'config.json').read_text()) for run in runs]
    assert {c['attack']['epsilon'] for c in configs} == {0., 1/255}
    assert all(c['attack']['iterations'] == 20 for c in configs)
    assert len({c['run_id'] for c in configs}) == 2
    assert len({c['selection']['subset_sha256'] for c in configs}) == 1
    for run in runs:
        rows = read_predictions(run)
        assert len(rows) == 3 and summarize(rows) == json.loads((run / 'metrics.json').read_text())
    with pytest.raises(Exception):
        parse_epsilon('3/255')


@pytest.mark.parametrize('name', ['fgsm', 'pgd'])
def test_known_classification_flip_not_just_loss_growth(name):
    classifier = nn.Linear(12, 2)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.weight[0].fill_(1)
        classifier.bias.zero_()
        classifier.bias[0] = -5.4
    model = nn.Sequential(nn.Flatten(), classifier)
    x = torch.full((1, 3, 2, 2), .5)
    labels = torch.tensor([0])
    assert model(x).argmax(1).item() == 0
    output = generate(model, x, labels, AttackConfig(name, .1))
    assert model(output).argmax(1).item() == 1
    assert float((output - x).abs().max()) <= .1 + 1e-6
    metrics = PairedMetrics()
    metrics.update(labels, model(x).argmax(1), model(output).argmax(1))
    assert metrics.compute()['asr'] == 1.


def test_mixed_result_chunks_are_rejected(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    run = evaluate(PixelModel(model, cfg), ds, selected, info, cfg,
                   AttackConfig('fgsm', 0.), tmp_path / 'results', batch_size=2)
    path = sorted((run / 'predictions').glob('*.csv'))[0]
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]['checkpoint_sha256'] = 'foreign-checkpoint'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='another run/model/checkpoint'):
        read_predictions(run)
