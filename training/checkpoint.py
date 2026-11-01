"""Epoch-boundary checkpoints. Load only trusted local experiment files."""
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch


FORMAT_VERSION = 1


def atomic_save(path, writer):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_save(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_f1, config, history, data_metadata, provenance):
    payload = {'format_version': FORMAT_VERSION, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'scaler': scaler.state_dict(), 'epoch': epoch, 'best_f1': best_f1,
               'config': config, 'history': history, 'rng': rng_state(),
               'data_metadata': data_metadata, 'provenance': provenance}
    atomic_save(path, lambda p: torch.save(payload, p))


def read_checkpoint(path):
    # Contains Python/numpy RNG state; explicit weights_only=False is intentional.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format')
    return state


def restore_checkpoint(state, model, optimizer, scheduler, scaler):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    # Optimizer state follows its parameter device (checkpoint loaded on CPU).
    for parameter, values in optimizer.state.items():
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and key != 'step':
                values[key] = value.to(parameter.device)
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])
    return state['epoch'] + 1, state['best_f1']
