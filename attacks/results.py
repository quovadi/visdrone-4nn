"""Immutable runs and atomic CSV chunks; no resume or reuse of old results."""
import csv
import json
from pathlib import Path
import re
import uuid
import torch
from training.checkpoint import atomic_save, write_json
from attacks.metrics import PairedMetrics

FIELDS = ('run_id', 'sample_id', 'index_row', 'image', 'box', 'line', 'true_class',
          'clean_prediction', 'adversarial_prediction', 'clean_correct', 'attack_success',
          'model', 'model_id', 'variant', 'checkpoint', 'checkpoint_sha256',
          'attack', 'epsilon', 'seed', 'attack_seed', 'attack_parameters', 'linf')


def create_run(output, config, run_id=None):
    run_id = run_id or uuid.uuid4().hex
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('Unsafe run identifier')
    run = Path(output) / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / 'predictions').mkdir()
    write_json(run / 'config.json', {**config, 'run_id': run_id})
    return run


def write_chunk(run, number, rows):
    path = run / 'predictions' / f'batch-{number:07d}.csv'
    if path.exists():
        raise FileExistsError(path)
    def writer(temporary):
        with temporary.open('w', newline='') as stream:
            output = csv.DictWriter(stream, fieldnames=FIELDS)
            output.writeheader()
            output.writerows(rows)
    atomic_save(path, writer)


def read_predictions(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    seen, rows = set(), []
    for path in sorted((run / 'predictions').glob('batch-*.csv')):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                for key in ('run_id', 'model', 'model_id', 'variant', 'checkpoint', 'checkpoint_sha256'):
                    if row[key] != str(config[key]):
                        raise ValueError('Prediction chunk belongs to another run/model/checkpoint')
                if (row['attack'] != config['attack']['name'] or float(row['epsilon']) != config['attack']['epsilon']
                        or int(row['seed']) != config['attack']['seed']
                        or json.loads(row['attack_parameters']) != config['attack']):
                    raise ValueError('Prediction attack configuration differs')
                if row['sample_id'] in seen:
                    raise ValueError('Duplicate sample_id')
                seen.add(row['sample_id'])
                for key in ('index_row', 'line', 'true_class', 'clean_prediction', 'adversarial_prediction', 'seed', 'attack_seed'):
                    row[key] = int(row[key])
                for key in ('epsilon', 'linf'):
                    row[key] = float(row[key])
                if not 0 <= row['linf'] <= row['epsilon'] + 1e-6:
                    raise ValueError('Invalid recorded perturbation budget')
                if row['clean_correct'] not in ('True', 'False') or row['attack_success'] not in ('True', 'False', ''):
                    raise ValueError('Invalid prediction boolean flags')
                row['clean_correct'] = row['clean_correct'] == 'True'
                row['attack_success'] = None if row['attack_success'] == '' else row['attack_success'] == 'True'
                correct = row['clean_prediction'] == row['true_class']
                success = row['adversarial_prediction'] != row['true_class'] if correct else None
                if row['clean_correct'] != correct or row['attack_success'] != success:
                    raise ValueError('Prediction flags disagree with labels')
                rows.append(row)
    return rows


def summarize(rows):
    metrics = PairedMetrics()
    for row in rows:
        metrics.update([row['true_class']], [row['clean_prediction']], [row['adversarial_prediction']])
    return metrics.compute() if rows else {'samples': 0, 'asr': None, 'asr_defined': False}


def save_examples(run, offset, images, adversarial, identities, count, epsilon):
    from PIL import Image
    import numpy as np
    folder = run / 'examples'
    folder.mkdir(exist_ok=True)
    for i in range(min(len(images), max(0, count - offset))):
        stem = f'{offset+i:07d}'
        for name, value in (('clean', images[i]), ('adversarial', adversarial[i])):
            pixels = (value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            atomic_save(folder / f'{stem}-{name}.png', lambda p: Image.fromarray(pixels).save(p, format='PNG'))
        delta = (adversarial[i] - images[i]).detach().cpu()
        atomic_save(folder / f'{stem}-delta.pt', lambda p: torch.save(delta, p))
        write_json(folder / f'{stem}.json', {'sample_id': identities[i]['sample_id'],
                   'epsilon': epsilon, 'delta': 'signed FP32 RGB difference, not display-scaled'})
