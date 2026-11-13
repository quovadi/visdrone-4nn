"""One model, one attack, one immutable run per epsilon. Validation only."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import warnings

import torch
from torch.utils.data import DataLoader

from attacks.core import AttackConfig, generate, check_bounds, evaluation_mode
from attacks.data import selected_validation, dataset
from attacks.metrics import PairedMetrics
from attacks.models import load_classifier
from attacks.preprocessing import PixelModel, rgb_transform
from attacks.results import create_run, write_chunk, save_examples
from training.checkpoint import write_json
from training.consistency import file_digest
from training.run import seed_everything
from training.storage import check_separate

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_EPSILONS = (0., 1/255, 2/255, 4/255, 8/255)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def code_provenance():
    def git(*args):
        result = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    diff = git('diff', 'HEAD') or ''
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'git_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
            'attack_source_sha256': {str(p.relative_to(REPO)): file_digest(p) for p in sorted((REPO / 'attacks').glob('*.py'))},
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm')},
            'python': sys.version, 'cuda_runtime': torch.version.cuda}


def evaluate(model, data, selection, model_info, data_config, attack, output, *,
             batch_size=8, workers=0, examples=0, run_id=None):
    if batch_size < 1 or workers < 0 or not 0 <= examples <= 16:
        raise ValueError('Positive batch size, nonnegative workers, and 0..16 examples required')
    if len(data) != selection['size'] or len(data) != len(selection['objects']):
        raise ValueError('Dataset and selected object identities disagree')
    started = utc_now()
    device = next(model.parameters()).device
    config = {**model_info, 'attack': attack.parameters(), 'selection': {k: v for k, v in selection.items() if k != 'objects'},
              'data_config': data_config, 'batch_size': batch_size, 'workers': workers,
              'save_examples': examples, 'device': str(device), 'started_utc': started,
              'provenance': code_provenance()}
    run = create_run(output, config, run_id)
    write_json(run / 'subset.json', selection)
    write_json(run / 'status.json', {'state': 'running', 'started_utc': started, 'recorded_samples': 0})
    aggregate = PairedMetrics()
    recorded = 0
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers)
    try:
        with (run / 'diagnostics.jsonl').open('x') as log, evaluation_mode(model), torch.autocast(device_type=device.type, enabled=False):
            for batch_number, (rgb, labels) in enumerate(loader):
                identities = selection['objects'][recorded:recorded+len(labels)]
                def diagnostic(event):
                    log.write(json.dumps({'batch': batch_number, 'sample_ids': [r['sample_id'] for r in identities], **event}) + '\n')
                    log.flush()
                rgb = rgb.to(device=device, dtype=torch.float32)
                labels = labels.to(device)
                if rgb.shape[1:] != (3, 224, 224):
                    raise ValueError('Evaluation requires post-geometry RGB [N,3,224,224]')
                # Independent output check is performed here even if the attack
                # implementation changes. Inputs are also validated for eps=0.
                check_bounds(rgb, rgb, 0.)
                with torch.no_grad():
                    logits = model(rgb)
                    if logits.shape != (len(labels), 10):
                        raise ValueError('Classifier must return [N,10] logits')
                    if not torch.isfinite(logits).all():
                        diagnostic({'kind': 'nonfinite_clean_logits'})
                        raise FloatingPointError('Non-finite clean inference')
                    clean = logits.argmax(1)
                adversarial = generate(model, rgb, labels, attack,
                    sample_seeds=[r['attack_seed'] for r in identities], diagnostic=diagnostic)
                linf = check_bounds(rgb, adversarial, attack.epsilon)
                with torch.no_grad():
                    logits = model(adversarial)
                    if logits.shape != (len(labels), 10):
                        raise ValueError('Classifier must return [N,10] logits')
                    if not torch.isfinite(logits).all():
                        diagnostic({'kind': 'nonfinite_adversarial_logits'})
                        raise FloatingPointError('Non-finite adversarial inference')
                    predictions = logits.argmax(1)
                rows = []
                for i, identity in enumerate(identities):
                    truth, before, after = int(labels[i]), int(clean[i]), int(predictions[i])
                    correct = before == truth
                    rows.append({'run_id': run.name, 'sample_id': identity['sample_id'],
                        'index_row': identity['index_row'], 'image': identity['image'],
                        'box': json.dumps(identity['box']), 'line': identity['line'],
                        'true_class': truth, 'clean_prediction': before, 'adversarial_prediction': after,
                        'clean_correct': correct, 'attack_success': after != truth if correct else None,
                        **{k: model_info[k] for k in ('model', 'model_id', 'variant', 'checkpoint', 'checkpoint_sha256')},
                        'attack': attack.name, 'epsilon': attack.epsilon, 'seed': attack.seed,
                        'attack_seed': identity['attack_seed'], 'attack_parameters': json.dumps(attack.parameters(), sort_keys=True),
                        'linf': float(linf[i])})
                write_chunk(run, batch_number, rows)  # Commit predictions first.
                aggregate.update(labels, clean, predictions)
                recorded += len(labels)
                write_json(run / 'metrics.json', aggregate.compute())
                write_json(run / 'status.json', {'state': 'running', 'started_utc': started, 'recorded_samples': recorded})
                if examples:
                    save_examples(run, recorded-len(labels), rgb, adversarial, identities, examples, attack.epsilon)
        # Refuse a clean completion if immutable inputs changed during the run.
        if file_digest(Path(model_info['checkpoint'])) != model_info['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during evaluation')
        if file_digest(Path(model_info['validation_index'])) != selection['index_sha256']:
            raise ValueError('Frozen validation index changed during evaluation')
        write_json(run / 'status.json', {'state': 'completed', 'started_utc': started,
                   'finished_utc': utc_now(), 'recorded_samples': recorded})
    except BaseException as error:
        try:
            write_json(run / 'status.json', {'state': 'failed', 'started_utc': started,
                       'finished_utc': utc_now(), 'recorded_samples': recorded,
                       'error': f'{type(error).__name__}: {error}'})
        except Exception as status_error:
            warnings.warn(f'Could not persist failure status: {status_error}', RuntimeWarning)
        raise
    return run


def parse_epsilon(value):
    try:
        epsilon = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError('Expected epsilon such as 4/255 or 0') from error
    if epsilon not in PROTOCOL_EPSILONS:
        raise argparse.ArgumentTypeError('Protocol epsilon must be 0, 1/255, 2/255, 4/255 or 8/255')
    return epsilon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-format', choices=('compact', 'full'), default='compact')
    parser.add_argument('--trusted-full-checkpoint', action='store_true')
    parser.add_argument('--variant', choices=('baseline-v1', 'convnext-base-v2'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--validation-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attack', choices=('fgsm', 'pgd'), required=True)
    parser.add_argument('--epsilon', type=parse_epsilon, nargs='+', required=True)
    parser.add_argument('--subset-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--save-examples', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    # Results cannot be written inside, over, or around protected inputs.
    for protected in (args.root, args.model_run, args.validation_index.parent, args.checkpoint):
        check_separate(args.output, protected)
    records, selection = selected_validation(args.validation_index, args.seed, args.subset_size)
    model, data_config, info = load_classifier(args.model_run, args.checkpoint, args.variant,
        args.checkpoint_format, args.trusted_full_checkpoint)
    info['validation_index'] = str(args.validation_index.resolve())
    pixels = PixelModel(model, data_config).to(args.device)
    images = dataset(args.root, records, rgb_transform(data_config))
    for epsilon in args.epsilon:
        seed_everything(args.seed)
        attack = AttackConfig(args.attack, epsilon, seed=args.seed)
        run = evaluate(pixels, images, selection, info, data_config, attack, args.output,
                       batch_size=args.batch_size, workers=args.workers, examples=args.save_examples)
        print(f'RUN_DIR={run}', flush=True)


if __name__ == '__main__':
    main()
