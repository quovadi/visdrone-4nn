"""Square algorithm, accounting and artifact tests; synthetic CPU only."""
import json
from unittest.mock import patch
import numpy as np
import pytest
import torch
from attacks.square import SquareConfig, generate_square, probability, proposal
from attacks.square_evaluation import evaluate_square, main
from attacks.square_results import read_square_predictions, query_summary
from attacks.preprocessing import PixelModel
from test_attacks import evaluation_fixture


def constant_scores(x):
    assert not torch.is_grad_enabled()
    return torch.stack((x.flatten(1).mean(1) + 2, torch.zeros(len(x))), 1)


@pytest.mark.parametrize('budget', [1, 2, 3, 20])
@pytest.mark.parametrize('epsilon', [0., 1/255, 8/255])
def test_budget_bounds_no_autograd(budget, epsilon):
    x = torch.full((1, 3, 8, 8), .5, requires_grad=True)
    calls = []
    def scores(candidate):
        assert not torch.is_grad_enabled()
        assert candidate.dtype == torch.float32
        assert candidate.min() >= 0 and candidate.max() <= 1
        assert (candidate-x.detach()).abs().max() <= epsilon+1e-6
        calls.append(candidate.detach().clone())
        return constant_scores(candidate)
    with patch('torch.autograd.grad', side_effect=AssertionError('must not use autograd')):
        result = generate_square(scores, x, torch.tensor([0]), SquareConfig(epsilon, budget), sample_seeds=[19])
    assert result['queries'].item() == len(calls) == (1 if epsilon == 0 else budget)
    assert result['stop_reason'] == (['epsilon_zero'] if epsilon == 0 else ['budget'])
    assert x.grad is None and not result['images'].requires_grad
    assert (result['images']-x.detach()).abs().max() <= epsilon+1e-6
    if epsilon == 0:
        assert torch.equal(result['images'], x)
    if len(calls) > 1:
        delta = calls[1] - x.detach()
        assert torch.equal(delta[:, :, :1, :].expand_as(delta), delta)  # vertical stripes
        assert torch.allclose(delta.abs(), torch.full_like(delta, epsilon), atol=1e-7)


def test_schedule_and_reference_square_proposal():
    points = [(0, 1), (10, 1), (11, 2), (50, 2), (51, 4), (200, 4), (201, 8),
              (501, 16), (1001, 32), (2001, 64), (4001, 128), (6001, 256), (8001, 512)]
    for t, divisor in points:
        assert probability(.05, t, 10000) == .05/divisor
    class FixedRNG:
        def integers(self, limit):
            return 0
        def choice(self, values, size):
            return -np.ones(size)
    origin = torch.full((3, 4, 4), .5)
    best = origin + .2
    candidate, side = proposal(origin, best, .2, .25, FixedRNG())
    expected = best.clone()
    expected[:, :2, :2] = .3
    assert side == 2 and torch.allclose(candidate, expected)


def test_success_stops_per_sample_and_margin_rejection():
    # Three identifiable brightness levels; oracle query history per sample.
    x = torch.stack([torch.full((3, 8, 8), value) for value in (.2, .5, .8)])
    calls = [0, 0, 0]
    def scores(batch):
        values = []
        for image in batch:
            index = int(round((float(image.mean())-.2)/.3))
            q = calls[index]
            calls[index] += 1
            # First succeeds on stripe initialization; second exhausts;
            # third is clean-incorrect and must not be attacked.
            margin = (-1 if q >= 1 else 10) if index == 0 else 10-q*.01
            values.append([margin, 0.])
        return torch.tensor(values, dtype=torch.float32)
    result = generate_square(scores, x, torch.tensor([0, 0, 1]), SquareConfig(.01, 7), sample_seeds=[1, 2, 3])
    assert calls == [2, 7, 1] == result['queries'].tolist()
    assert result['success'].tolist() == [True, False, False]
    assert result['stop_reason'] == ['success', 'budget', 'clean_incorrect']
    assert torch.equal(result['images'][2], x[2])
    # Greedy margin minimization must reject a worse candidate.
    seen = []
    def scripted(batch):
        seen.append(batch.clone())
        return torch.tensor([[10., 0.], [9., 0.], [11., 0.], [8., 0.], [-1., 0.]][len(seen)-1:len(seen)])
    r = generate_square(scripted, x[:1], torch.tensor([0]), SquareConfig(.1, 5), sample_seeds=[100])
    assert r['queries'].item() == 5 and r['accepted_updates'] == [2]
    assert r['success'].item() and torch.equal(r['images'], seen[-1])


def test_per_sample_determinism_order_batch_and_global_rng():
    x = torch.linspace(.2, .8, 3*3*8*8).reshape(3, 3, 8, 8)
    labels = torch.zeros(3, dtype=torch.int64)
    seeds = [111, 222, 333]
    config = SquareConfig(.1, 15)
    before = torch.get_rng_state().clone()
    batch = generate_square(constant_scores, x, labels, config, sample_seeds=seeds)
    for i in range(3):
        single = generate_square(constant_scores, x[i:i+1], labels[i:i+1], config, sample_seeds=seeds[i:i+1])
        assert torch.equal(single['images'][0], batch['images'][i])
        assert single['queries'][0] == batch['queries'][i]
    order = [2, 0, 1]
    shuffled = generate_square(constant_scores, x[order], labels[order], config, sample_seeds=[seeds[i] for i in order])
    assert torch.equal(shuffled['images'], batch['images'][order])
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match='per-sample'):
        generate_square(constant_scores, x, labels, config)


def test_no_effective_move_and_nonfinite_diagnostic():
    result = generate_square(constant_scores, torch.zeros(1, 3, 4, 4), torch.tensor([0]),
                             SquareConfig(1e-10, 10), sample_seeds=[7])
    assert result['queries'].item() <= 10
    events = []
    with pytest.raises(ValueError, match='finite FP32'):
        generate_square(lambda x: torch.full((len(x), 2), float('nan')), torch.zeros(1, 3, 4, 4),
                        torch.tensor([0]), SquareConfig(.1), sample_seeds=[1], diagnostic=events.append)
    assert events[0]['queries'] == [1]


def test_square_evaluation_and_query_integrity(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    original = {k:v.clone() for k,v in model.state_dict().items()}
    run = evaluate_square(PixelModel(model, cfg), ds, selected, info, cfg,
                          SquareConfig(4/255, 8), tmp_path / 'results', batch_size=2)
    rows = read_square_predictions(run)
    assert len(rows) == 6
    assert query_summary(rows) == json.loads((run / 'query_metrics.json').read_text())
    for row in rows:
        assert 1 <= row['queries'] <= 8
        if not row['clean_correct']:
            assert row['queries'] == 1
    for k,v in model.state_dict().items():
        assert torch.equal(original[k], v)
    path = sorted((run / 'queries').glob('*.json'))[0]
    records = json.loads(path.read_text())
    records[0]['queries'] = 999
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='budget'):
        read_square_predictions(run)


def test_cli_synthetic_cpu_smoke(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    args = ['square_evaluation', '--model-run', str(tmp_path / 'metadata'), '--checkpoint', info['checkpoint'],
            '--variant', 'baseline-v1', '--root', str(root), '--validation-index', str(index),
            '--output', str(tmp_path / 'results'), '--epsilon', '0', '4/255', '--max-queries', '5',
            '--subset-size', '3', '--batch-size', '2', '--device', 'cpu']
    with patch('sys.argv', args), patch('attacks.square_evaluation.load_classifier', return_value=(model, cfg, info)):
        main()
    for run in (tmp_path / 'results').iterdir():
        assert len(read_square_predictions(run)) == 3
        assert json.loads((run / 'status.json').read_text())['state'] == 'completed'


def test_partial_diagnostics_and_immutable_runs(tmp_path):
    import attacks.square_evaluation as module
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    writer = module.write_chunk
    def interrupted(run, number, rows):
        if number == 1:
            raise OSError('synthetic interruption')
        writer(run, number, rows)
    with patch('attacks.square_evaluation.write_chunk', side_effect=interrupted):
        with pytest.raises(OSError, match='synthetic interruption'):
            evaluate_square(PixelModel(model, cfg), ds, selected, info, cfg,
                SquareConfig(4/255, 4), tmp_path / 'results', batch_size=2, run_id='partial')
    run = tmp_path / 'results' / 'partial'
    rows = read_square_predictions(run)
    assert len(rows) == 2  # Orphan query chunk is not counted as completed prediction.
    assert len(list((run / 'queries').glob('*.json'))) == 2
    assert (run / 'diagnostics.jsonl').read_text()
    before = (run / 'config.json').read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_square(PixelModel(model, cfg), ds, selected, info, cfg,
            SquareConfig(4/255, 4), tmp_path / 'results', batch_size=2, run_id='partial')
    assert (run / 'config.json').read_bytes() == before
