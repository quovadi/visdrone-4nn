"""CLI: one model per process, immutable runs, resume from last completed epoch."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid
import zipfile

import numpy as np
import torch

from training.config import TrainConfig, new_config
from training.models import build_model, make_transforms
from training.data import make_datasets, epoch_loaders
from training.engine import train_epoch
from training.validation import validate
from training.checkpoint import (read_checkpoint, restore_checkpoint, save_checkpoint,
                                 write_json, export_weights)
from training.consistency import reconcile, tensor_digest
from training.storage import check_separate, durable_directory, sync_results

REPO = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def source_files():
    files = [REPO / 'requirements.txt']
    for folder, pattern in (('training', '*.py'), ('data', '*.py'), ('configs', '*.json')):
        files.extend(sorted((REPO / folder).rglob(pattern)))
    return files


def provenance():
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_files()}
    def git(*args):
        result = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'torchvision', 'timm', 'numpy', 'Pillow')}
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'source_sha256': hashes, 'versions': versions,
            'python': os.sys.version, 'cuda_runtime': torch.version.cuda}


def compatible_sources(saved, current):
    if saved == current:
        return True
    # Narrow migration from the original stage-2 code. All other source and
    # configuration fingerprints must match; arbitrary code changes still fail.
    legacy = {
        'training/engine.py': '9d85f633ca38f161bd98a9686ed57b4b313a8a76b5b820aba9ba105e1a2f38df',
        'training/run.py': '548c68e9f57ef7014e1ad33a5fc1cf168da024baa401ba46a47fbf75d3b2bb7d',
    }
    candidates = [{**current, **legacy}]
    prior = {
        'training/config.py': '513e052e0aa1c7fc96b8717ecb7944e7f9bb973ab5343ccb7b040b42e035f691',
        'training/checkpoint.py': 'a55f7c3ef91d6e2305ede6aa15289fe799763d64502c1fd688b78ff250a73a35',
        'training/run.py': 'a16b0554002caa15801bfbee694efbddcdeb217d939fdbdbca2860baf02b7acd',
    }
    previous = {k: v for k, v in {**current, **prior}.items() if k != 'training/storage.py'}
    candidates.extend([previous, {**previous, **legacy}])
    pr2 = {**current,
        'training/run.py': '9afd8f9237d54784e56a3ca63f04b340a909698f2c4c183d4e636eb96f563d3d',
        'training/checkpoint.py': '5c77835464c723cf5529c6e9fd2b87498b0e2b3451aca7dfab18df2f9b65ae77'}
    pr2['training/storage.py'] = '13402ab2ca280fb022481a8a9ee6dcc63db72c7e11b06ff87219302e8c553745'
    pr2.pop('training/consistency.py', None)
    candidates.append(pr2)
    for candidate in list(candidates):
        legacy_candidate = dict(candidate)
        legacy_candidate.pop('training/consistency.py', None)
        candidates.append(legacy_candidate)
    return saved in candidates


def new_run_directory(config):
    output = Path(config.output).resolve()
    indices = Path(config.indices).resolve()
    if output == indices or indices in output.parents or output in indices.parents:
        raise ValueError('Results and frozen data-v1 must be separate directory trees')
    if config.economical:
        check_separate(config.work_dir, config.output)
        check_separate(config.work_dir, config.indices)
        output = Path(config.work_dir).resolve()
    # Model-specific parent; unique identifier and exclusive mkdir prevent overwrite.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = 'smoke' if config.smoke else 'full'
    run = output / config.model / f'{prefix}-{stamp}-{uuid.uuid4().hex[:8]}'
    run.mkdir(parents=True, exist_ok=False)
    return run


def run_training(config, resume=None):
    seed_everything(config.seed)
    device = torch.device('cuda' if config.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if config.device == 'auto' else config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    amp = config.amp and device.type == 'cuda'
    evidence = provenance()
    state = None
    if resume is not None:
        run = Path(resume)
        state = read_checkpoint(run / 'last.pt')
        expected = {k: v for k, v in config.to_dict().items() if k not in ('root', 'device')}
        saved = {k: v for k, v in TrainConfig(**state['config']).to_dict().items() if k not in ('root', 'device')}
        if expected != saved:
            raise ValueError('Resume configuration differs from checkpoint; create a new run')
        old = state['provenance']
        if (not compatible_sources(old['source_sha256'], evidence['source_sha256'])
                or old['versions'] != evidence['versions']):
            raise ValueError('Code/dependency versions changed; restore recorded environment for resume')
        if old['device_type'] != device.type or old['amp_enabled'] != amp:
            raise ValueError('Resume requires the same device type and AMP mode')
        evidence = {**old, **evidence}
        if old['source_sha256'] != evidence['source_sha256']:
            evidence['resumed_from_source_sha256'] = old['source_sha256']
            write_json(run / 'provenance.json', evidence)
    else:
        run = new_run_directory(config)
        evidence.update(device_type=device.type, amp_enabled=amp,
                        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
        write_json(run / 'config.json', config.to_dict())
        write_json(run / 'provenance.json', evidence)
        with zipfile.ZipFile(run / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in source_files():
                archive.write(path, path.relative_to(REPO))
        freeze = subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (run / 'environment.txt').write_text(freeze.stdout)
    durable = durable_directory(config, run, resume=resume is not None)
    print(f'RUN_DIR={run}', flush=True)
    if durable is not None:
        print(f'DURABLE_DIR={durable}', flush=True)
    # Exactly one architecture in memory. Pretrained is mandatory for a new run.
    model, data_config = build_model(config.model_id, pretrained=state is None)
    train_transform, val_transform = make_transforms(data_config)
    train, validation, data_metadata = make_datasets(config, train_transform, val_transform)
    data_metadata['resolved_data_config'] = data_config
    # JSON canonicalization removes tuple/list serialization differences on resume.
    data_metadata = json.loads(json.dumps(data_metadata))
    if state is not None and state['data_metadata'] != data_metadata:
        raise ValueError('Frozen indices, selected records or preprocessing changed')
    if state is None:
        write_json(run / 'data.json', data_metadata)
        write_json(run / 'preprocessing.json', {'data_config': data_config,
                   'train': str(train_transform), 'validation': str(val_transform)})
    selection = None
    if state is not None:
        selection = reconcile(run, state, config.economical, durable)
        sync_results(run, durable)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best_f1, history = 0, -1.0, []
    if state is not None:
        start, best_f1 = restore_checkpoint(state, model, optimizer, scheduler, scaler)
        history = state['history']
        del state  # Release CPU model/Adam buffers before the next epoch.
        write_json(run / 'metrics.json', history)  # last.pt is the source of truth
    epochs = 1 if config.smoke else config.epochs
    for epoch in range(start, epochs):
        # Also handles workers=0: augmentation and dropout restart identically.
        seed_everything(config.seed + epoch)
        train_loader, val_loader = epoch_loaders(config, train, validation, epoch)
        learning_rate = optimizer.param_groups[0]['lr']
        started = time.monotonic()
        training = train_epoch(model, train_loader, optimizer, criterion, scaler, device, amp)
        metrics = validate(model, val_loader, criterion, device, amp)
        if not np.isfinite(metrics['loss']):
            raise FloatingPointError('Non-finite validation loss')
        if training['optimizer_steps'] > 0:
            scheduler.step()  # Advance only after an actual optimizer update.
        row = {'epoch': epoch + 1, 'lr': learning_rate, 'train': training,
               'validation': metrics, 'seconds': time.monotonic() - started}
        history.append(row)
        improved = metrics['macro_f1'] > best_f1
        best_f1 = max(best_f1, metrics['macro_f1'])
        arguments = (model, optimizer, scheduler, scaler, epoch, best_f1,
                     config.to_dict(), history, data_metadata, evidence)
        if improved:
            export_weights(run / 'best_pending.pt', model)
            selection = {'epoch': epoch + 1, 'score': best_f1,
                         'sha256': tensor_digest(model.state_dict())}
        save_checkpoint(run / 'last.pt', *arguments, best_selection=selection)
        # The atomic last.pt replacement is the commit point. Publication is
        # idempotent and can be completed after interruption from this state.
        if improved and not config.economical:
            save_checkpoint(run / 'best.pt', *arguments, best_selection=selection)
        committed = {'history': history, 'epoch': epoch, 'best_f1': best_f1,
                     'best_selection': selection, 'config': config.to_dict(),
                     'model': model.state_dict()}
        selection = reconcile(run, committed, config.economical, durable)
        sync_results(run, durable, weights=improved)
        print(json.dumps(row, allow_nan=False), flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('convnext_base', 'convnext_nano', 'vit_base', 'tinyvit'))
    parser.add_argument('--resume', type=Path, help='Existing RUN_DIR containing config.json and last.pt')
    for option in ('root', 'indices', 'output', 'device', 'work-dir'):
        parser.add_argument(f'--{option}')
    for option in ('batch-size', 'epochs', 'workers', 'seed', 'train-size'):
        parser.add_argument(f'--{option}', type=int)
    for option in ('lr', 'weight-decay'):
        parser.add_argument(f'--{option}', type=float)
    parser.add_argument('--economical', action='store_true', default=None)
    parser.add_argument('--smoke', action='store_true', default=None)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args())
    resume, model = args.pop('resume'), args.pop('model')
    overrides = {key: value for key, value in args.items() if value is not None}
    if resume is None:
        if model is None:
            parser.error('--model is required for a new experiment')
        config = new_config(model, **overrides)
    else:
        config = TrainConfig(**json.loads((resume / 'config.json').read_text()))
        if model is not None and model != config.model:
            parser.error('--model differs from saved configuration')
        config = replace(config, **overrides)
    run_training(config, resume)


if __name__ == '__main__':
    main()
