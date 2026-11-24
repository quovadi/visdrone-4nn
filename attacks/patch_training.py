"""Train one shared patch on official train crops; no validation selection."""
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from attacks.data import canonical_hash
from attacks.adversarial_patch import PatchConfig, frozen_classifier, optimize_step, patch_side, placements
from attacks.patch_io import new_run, finish, verify, write_csv, now
from training.checkpoint import atomic_save, write_json
from training.consistency import file_digest
from training.storage import check_separate

LOSS_FIELDS = ('step', 'loss', 'clean_correct', 'updated', 'gradient_l1')


def save_patch_image(path, patch):
    pixels = (patch.detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0,255).astype(np.uint8)
    atomic_save(path, lambda p: Image.fromarray(pixels).save(p, format='PNG'))


def loss_plot(run, history):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6.4,4.2), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    valid = [r for r in history if r['loss'] is not None]
    ax.plot([r['step'] for r in valid], [r['loss'] for r in valid], color='#2459a6')
    ax.set(xlabel='Optimization iteration', ylabel='Untargeted CE (maximize)', title='Patch train-only objective')
    ax.grid(alpha=.25)
    for extension in ('png','pdf'):
        atomic_save(run / f'training_loss.{extension}', lambda p: fig.savefig(p, format=extension, dpi=300))
    fig.clear()


def train_patch(model, data, selection, model_info, data_config, config, output, *, plots=True):
    for value in [model_info['checkpoint']] + ([data.root] if hasattr(data,'root') else []):
        check_separate(Path(output)/'patches',value)
    if canonical_hash(selection['objects']) != selection['subset_sha256'] or len({o['sample_id'] for o in selection['objects']}) != len(selection['objects']):
        raise ValueError('Invalid train selection identity/hash')
    if len(selection['objects']) != selection['size'] or len(data) != selection['size'] or not len(data):
        raise ValueError('Train subset/dataset size mismatch')
    if any(not o['image'].startswith('VisDrone2019-DET-train/images/') for o in selection['objects']):
        raise ValueError('Validation cannot be used to optimize a patch')
    device = next(model.parameters()).device
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FP32 classifier required')
    run = new_run(output, 'patches', {'optimization': config.to_dict(), 'source': model_info,
            'data_config': data_config, 'train_selection_sha256': selection['subset_sha256']})
    write_json(run / 'train_subset.json', selection)
    side = patch_side(config.area_fraction)
    if config.initialization == 'uniform':
        initial = torch.rand((3, side, side), generator=torch.Generator(device='cpu').manual_seed(config.seed), dtype=torch.float32)
    else:
        initial = torch.full((3, side, side), .5, dtype=torch.float32)
    patch = nn.Parameter(initial.to(device))
    optimizer_class = torch.optim.Adam if config.optimizer == 'adam' else torch.optim.SGD
    optimizer = optimizer_class([patch], lr=config.lr)
    history = []
    started = time.monotonic()
    try:
        with frozen_classifier(model), torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            for step in range(config.steps):
                indices = random.Random(int.from_bytes(f'{config.seed}:{step}'.encode(), 'big')).sample(
                    range(len(data)), min(config.batch_size, len(data)))
                examples = [data[i] for i in indices]
                images = torch.stack([x for x,y in examples]).to(device, dtype=torch.float32)
                if any(y != selection['objects'][i]['label'] for i,(x,y) in zip(indices,examples)):
                    raise ValueError('Train dataset labels differ from frozen selection')
                labels = torch.tensor([y for x,y in examples], dtype=torch.int64, device=device)
                if images.shape[1:] != (3,224,224):
                    raise ValueError('Patch protocol requires 224x224 RGB object crops')
                coords = placements([selection['objects'][i]['sample_id'] for i in indices],
                                    config.area_fraction, config.seed, step=step)
                row = {'step': step+1, **optimize_step(model, images, labels, patch, optimizer, coords, config.area_fraction)}
                history.append(row)
                # Small patch-only emergency snapshot. No classifier weights stored.
                atomic_save(run / 'last_patch.pt', lambda p: torch.save({'patch': patch.detach().cpu(),
                    'optimizer': optimizer.state_dict(), 'step': step+1, 'config': config.to_dict(),
                    'source_checkpoint_sha256': model_info['checkpoint_sha256'],
                    'train_subset_sha256': selection['subset_sha256'], 'history': history}, p))
                write_csv(run / 'training_loss.csv', history, LOSS_FIELDS)
        updates = sum(row['updated'] for row in history)
        if not updates:
            raise ValueError('No clean-correct optimization samples; no trained patch published')
        nonzero = sum(r['gradient_l1']>0 for r in history)
        if not nonzero:
            raise ValueError('All patch gradients are zero; no optimized patch published')
        if 'optimization_train_index' in model_info and file_digest(Path(model_info['optimization_train_index'])) != selection['index_sha256']:
            raise ValueError('Frozen train index changed during optimization')
        if file_digest(Path(model_info['checkpoint'])) != model_info['checkpoint_sha256']:
            raise ValueError('Source classifier checkpoint changed')
        atomic_save(run / 'patch.pt', lambda p: torch.save(patch.detach().cpu(), p))
        save_patch_image(run / 'patch.png', patch)
        plot_error = None
        if plots:
            try:
                loss_plot(run, history)
            except Exception as error:
                plot_error = f'{type(error).__name__}: {error}'
        write_json(run / 'metadata.json', {'source_model': model_info['model'], 'source_variant': model_info['variant'],
            'source_checkpoint_sha256': model_info['checkpoint_sha256'], 'area_fraction': config.area_fraction,
            'native_resolution': [224,224], 'patch_side': side, 'realized_area_fraction': side*side/(224*224),
            'train_index_sha256': selection['index_sha256'], 'train_subset_sha256': selection['subset_sha256'],
            'seed': config.seed, 'optimizer_updates': updates, 'nonzero_gradient_steps':nonzero, 'runtime_seconds': time.monotonic()-started,
            'patch_sha256': file_digest(run / 'patch.pt'), 'synthetic':model_info.get('synthetic',False),
            'optimization':config.to_dict(), 'git_commit':json.loads((run/'config.json').read_text())['provenance']['git_commit'], 'plot_error': plot_error, 'plots_requested': plots})
        finish(run)
    except BaseException as error:
        write_json(run / 'status.json', {'state': 'failed', 'finished_utc': now(), 'error': str(error)})
        raise
    return run


def load_patch(run):
    run = Path(run)
    config = verify(run, 'patches')
    metadata = json.loads((run / 'metadata.json').read_text())
    if (metadata['source_model'] != config['source']['model']
            or metadata['source_checkpoint_sha256'] != config['source']['checkpoint_sha256']
            or metadata['area_fraction'] != config['optimization']['area_fraction']):
        raise ValueError('Patch metadata/source identity mismatch')
    patch = torch.load(run / 'patch.pt', map_location='cpu', weights_only=True)
    side = patch_side(metadata['area_fraction'], *metadata['native_resolution'])
    if (metadata['patch_side'] != side or metadata['native_resolution'] != [224,224]
            or metadata['source_variant'] != config['source']['variant']):
        raise ValueError('Patch geometry/variant metadata mismatch')
    if (not isinstance(patch, torch.Tensor) or patch.dtype != torch.float32 or patch.shape != (3,side,side)
            or not torch.isfinite(patch).all() or patch.min() < 0 or patch.max() > 1
            or metadata['patch_sha256'] != file_digest(run / 'patch.pt')):
        raise ValueError('Invalid universal patch tensor/identity')
    return patch, metadata
