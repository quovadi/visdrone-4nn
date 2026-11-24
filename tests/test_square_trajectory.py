"""Exact first-hit times, single-run curves, legacy and plotting isolation."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from attacks.square import SquareConfig, generate_square
from attacks.square_evaluation import evaluate_square
from attacks.square_results import read_square_predictions
from attacks.square_trajectory import compute_trajectory, export_trajectory
from attacks.preprocessing import PixelModel
from training.consistency import file_digest
from test_attacks import evaluation_fixture


def test_first_success_is_observed_including_rejected_queries():
    calls = []
    def scores(batch):
        calls.append(batch.clone())
        # clean, initialization, rejected candidate, improvement, success
        return torch.tensor([[10., 0.], [9., 0.], [11., 0.], [8., 0.], [-1., 0.]][len(calls)-1:len(calls)])
    result = generate_square(scores, torch.full((1, 3, 8, 8), .5), torch.tensor([0]),
                             SquareConfig(.1, 10), sample_seeds=[11])
    assert len(calls) == 5 and result['queries'].tolist() == [5]
    assert result['first_success_query'] == [5]
    assert result['accepted_updates'] == [2]  # The rejected third query still counts.


@pytest.mark.parametrize('budget', [1, 2, 5, 10])
def test_first_hits_batch_independence_and_clean_errors(budget):
    x = torch.stack([torch.full((3, 8, 8), value) for value in (.2, .5, .8)])
    labels = torch.tensor([0, 0, 1])
    seeds = [10, 20, 30]
    def oracle():
        counts = [0]*3
        def scores(batch):
            output = []
            for image in batch:
                i = int(round((float(image.mean())-.2)/.3))
                counts[i] += 1
                # Initialization success, never success, clean incorrect.
                output.append([-1. if i == 0 and counts[i] >= 2 else 10., 0.])
            return torch.tensor(output)
        return scores, counts
    scores, counts = oracle()
    batch = generate_square(scores, x, labels, SquareConfig(.01, budget), sample_seeds=seeds)
    assert batch['first_success_query'] == [2 if budget >= 2 else None, None, None]
    assert counts == [min(2, budget), budget, 1]
    for order in ([2, 0, 1], [1, 2, 0]):
        scores, _ = oracle()
        permuted = generate_square(scores, x[order], labels[order], SquareConfig(.01, budget),
                                   sample_seeds=[seeds[i] for i in order])
        assert permuted['first_success_query'] == [batch['first_success_query'][i] for i in order]
        assert torch.equal(permuted['images'], batch['images'][order])
    for i in range(3):
        scores, _ = oracle()
        single = generate_square(scores, x[i:i+1], labels[i:i+1], SquareConfig(.01, budget), sample_seeds=seeds[i:i+1])
        assert single['first_success_query'] == [batch['first_success_query'][i]]


def make_run(tmp_path, cap=8, *, all_wrong=False):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    if all_wrong:
        # Every fixture label is 0/1/2, so this fixed model always errs cleanly.
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.zero_()
            model[-1].bias[9] = 10.
    return evaluate_square(PixelModel(model, cfg), ds, selected, info, cfg,
                           SquareConfig(4/255, cap), tmp_path / 'results', batch_size=2)


def test_complete_curve_and_export_with_undefined_denominator(tmp_path):
    run = make_run(tmp_path, all_wrong=True)
    config, curve = compute_trajectory(run)
    assert [r['query_budget'] for r in curve] == list(range(1, 9))
    assert all(r['asr'] is None and r['clean_correct_samples'] == 0 for r in curve)
    assert curve[-1]['asr'] == json.loads((run / 'metrics.json').read_text())['asr']
    sources = {p: file_digest(p) for p in run.rglob('*') if p.is_file()}
    analysis = export_trajectory(run, tmp_path / 'analysis', checkpoints=[100, 250, 500, 750, 1000, 1500])
    assert (analysis / 'asr_vs_queries.png').read_bytes().startswith(b'\x89PNG')
    assert (analysis / 'asr_vs_queries.pdf').read_bytes().startswith(b'%PDF')
    with (analysis / 'trajectory.csv').open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 8 and all(row['asr'] == '' for row in exported)
    assert all(not r['available'] and r['asr'] is None for r in json.loads((analysis / 'checkpoints.json').read_text()))
    assert sources == {p: file_digest(p) for p in run.rglob('*') if p.is_file()}


def successful_run(tmp_path):
    root, index, model, cfg, selected, ds, info = evaluation_fixture(tmp_path)
    # PixelModel supplies normalized uniform-color images; sequential dataset
    # repeats colors, so use batch_size=1 and fixed clean-query boundaries.
    class Scripted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.))
            self.sample = 0
            self.query = 0
        def forward(self, images):
            self.query += 1
            label = selected['objects'][self.sample]['label']
            first_hits = [2, None, 5, None, None, 100]
            # Sample 3 is a clean mistake, excluded from the denominator.
            wrong = self.sample == 3 or first_hits[self.sample] is not None and self.query >= first_hits[self.sample]
            pred = (label+1) % 10 if wrong else label
            logits = torch.full((1, 10), -10.)
            logits[0, pred] = 10.
            if wrong or self.query == 100:
                self.sample += 1
                self.query = 0
            return logits
    run = evaluate_square(Scripted(), ds, selected, info, cfg, SquareConfig(4/255, 100),
                          tmp_path / 'results', batch_size=1)
    return run


def test_exact_cumulative_asr_and_display_checkpoints(tmp_path):
    run = successful_run(tmp_path)
    rows = read_square_predictions(run)
    assert [r['first_success_query'] for r in rows] == [2, None, 5, None, None, 100]
    _, curve = compute_trajectory(run)
    assert curve[0]['asr'] == 0 and curve[1]['asr'] == 1/5
    assert curve[3]['asr'] == 1/5 and curve[4]['asr'] == 2/5
    assert curve[98]['asr'] == 2/5 and curve[99]['asr'] == 3/5
    assert all(r['clean_correct_samples'] == 5 for r in curve)
    assert [r['asr'] for r in curve] == sorted(r['asr'] for r in curve)
    assert curve[-1]['asr'] == json.loads((run / 'metrics.json').read_text())['asr']
    analysis = export_trajectory(run, tmp_path / 'analysis', checkpoints=[100, 250])
    checkpoints = json.loads((analysis / 'checkpoints.json').read_text())
    assert checkpoints == [{'query_budget': 100, 'available': True, 'asr': 3/5},
                           {'query_budget': 250, 'available': False, 'asr': None}]


@pytest.mark.parametrize('value', [0, 1, -1, True, 999, 2.5])
def test_reader_rejects_invalid_first_success_values(tmp_path, value):
    run = make_run(tmp_path)
    path = sorted((run / 'queries').glob('*.json'))[0]
    records = json.loads(path.read_text())
    records[0]['first_success_query'] = value
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='first_success_query'):
        read_square_predictions(run)


def test_explicit_legacy_read_without_guessing_or_mutation(tmp_path):
    run = successful_run(tmp_path)
    for path in (run / 'queries').glob('*.json'):
        records = json.loads(path.read_text())
        for r in records:
            r.pop('first_success_query')
        path.write_text(json.dumps(records))
    sources = {p: file_digest(p) for p in run.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='Legacy'):
        read_square_predictions(run)
    with pytest.warns(RuntimeWarning, match='Legacy'):
        legacy = read_square_predictions(run, allow_legacy=True)
    assert any(r['success'] for r in legacy)
    assert all(r['first_success_query'] is None and not r['first_success_query_recorded'] for r in legacy)
    with pytest.raises(ValueError, match='Legacy'):
        compute_trajectory(run)
    assert sources == {p: file_digest(p) for p in run.rglob('*') if p.is_file()}


def test_plot_failure_never_changes_completed_attack(tmp_path):
    run = make_run(tmp_path)
    sources = {p: file_digest(p) for p in run.rglob('*') if p.is_file()}
    with patch('attacks.square_trajectory.plot_trajectory', side_effect=RuntimeError('plot backend failed')):
        analysis = export_trajectory(run, tmp_path / 'analysis')
    status = json.loads((analysis / 'postprocessing_status.json').read_text())
    assert status['state'] == 'plot_failed' and status['csv_complete']
    assert (analysis / 'trajectory.csv').is_file()
    assert json.loads((run / 'status.json').read_text())['state'] == 'completed'
    assert sources == {p: file_digest(p) for p in run.rglob('*') if p.is_file()}
    second = export_trajectory(run, tmp_path / 'analysis')
    assert second != analysis
    assert json.loads((second / 'postprocessing_status.json').read_text())['state'] == 'completed'
    with pytest.raises(ValueError, match='separate'):
        export_trajectory(run, run / 'analysis')


@pytest.mark.parametrize('row_index,value', [(0, None), (0, 1), (0, True), (0, 3), (0, 2.0), (2, 2)])
def test_successful_rows_require_exact_immediate_stop_index(tmp_path, row_index, value):
    run = successful_run(tmp_path)
    path = sorted((run / 'queries').glob('*.json'))[row_index]
    records = json.loads(path.read_text())
    records[0]['first_success_query'] = value
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='first_success_query'):
        read_square_predictions(run, allow_legacy=True)  # Null != missing legacy field.


@pytest.mark.parametrize('cap', [1, 8])
def test_failed_samples_keep_constant_denominator_to_cap(tmp_path, cap):
    run = make_run(tmp_path, cap=cap)
    _, curve = compute_trajectory(run)
    assert len(curve) == cap and curve[0]['clean_correct_samples'] > 0
    assert all(r['asr'] == 0. for r in curve)
    assert all(r['clean_correct_samples'] == curve[0]['clean_correct_samples'] for r in curve)
