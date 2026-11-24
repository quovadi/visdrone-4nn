"""Stage 5 universal RGB replacement patches; no rotations or random scaling."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F
from attacks.core import evaluation_mode
from attacks.data import canonical_hash

AREAS = (.05, .10, .15)
MODELS = ('convnext_nano', 'tinyvit', 'convnext_base', 'vit_base')


def patch_side(area_fraction, height=224, width=224):
    if area_fraction not in AREAS or min(height, width) < 1:
        raise ValueError('Patch area must be .05, .10 or .15')
    side = round(math.sqrt(area_fraction * height * width))
    if not 1 <= side <= min(height, width):
        raise ValueError('Requested square area cannot fit image geometry')
    return side


def placements(sample_ids, area_fraction, seed, height=224, width=224, *, step=None):
    side = patch_side(area_fraction, height, width)
    result = []
    for identity in sample_ids:
        # Never includes model identity or batch position. Eval step=None is fixed.
        value = int(canonical_hash(['patch-placement-v1', seed, identity, area_fraction, height, width, step])[:16], 16)
        rng = np.random.default_rng(value)
        result.append((int(rng.integers(height-side+1)), int(rng.integers(width-side+1))))
    return result


def apply_patch(images, patch, coordinates, area_fraction):
    if (images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.float32
            or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1):
        raise ValueError('Expected finite FP32 NCHW RGB [0,1]')
    if (patch.ndim != 3 or patch.shape[0] != 3 or patch.shape[1] != patch.shape[2]
            or patch.dtype != torch.float32 or not torch.isfinite(patch).all()
            or patch.min() < 0 or patch.max() > 1 or patch.device != images.device):
        raise ValueError('Expected finite square FP32 RGB patch [0,1] on input device')
    if len(coordinates) != len(images):
        raise ValueError('One position per image required')
    height, width = images.shape[-2:]
    side = patch_side(area_fraction, height, width)
    rendered = patch if patch.shape[-1] == side else F.interpolate(
        patch[None], size=(side, side), mode='bilinear', align_corners=False)[0]
    output = []
    for image, (top, left) in zip(images, coordinates):
        if type(top) is not int or type(left) is not int or not (0 <= top <= height-side and 0 <= left <= width-side):
            raise ValueError('Patch placement outside image')
        padding = (left, width-left-side, top, height-top-side)
        canvas = F.pad(rendered, padding)
        mask = F.pad(torch.ones((1, side, side), device=images.device, dtype=torch.bool), padding)
        output.append(torch.where(mask, canvas, image))
    return torch.stack(output)


@dataclass(frozen=True)
class PatchConfig:
    area_fraction: float
    steps: int = 500
    batch_size: int = 8
    lr: float = .05
    optimizer: str = 'adam'
    seed: int = 42
    initialization: str = 'uniform'

    def __post_init__(self):
        patch_side(self.area_fraction)
        if type(self.steps) is not int or self.steps < 1 or type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('Positive integer steps/batch_size required')
        if not math.isfinite(self.lr) or self.lr <= 0 or type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('Invalid lr or seed')
        if self.optimizer not in ('adam', 'sgd') or self.initialization not in ('uniform', 'gray'):
            raise ValueError('Supported optimizer: adam/sgd; initialization: uniform/gray')

    def to_dict(self):
        return {**asdict(self), 'objective': 'maximize CE on clean-correct train crops',
                'precision': 'fp32', 'rotation': False, 'random_scaling': False,
                'compositing': 'RGB replacement', 'native_resolution': [224, 224]}


@contextmanager
def frozen_classifier(model):
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in parameters:
            p.requires_grad_(False)
        with evaluation_mode(model):
            yield
    finally:
        for p, flag in parameters:
            p.requires_grad_(flag)


def optimize_step(model, images, labels, patch, optimizer, positions, area_fraction):
    """One ascent step. Caller owns frozen_classifier; never backprop to images."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Classifier must be frozen during patch optimization')
    images = images.detach()
    with torch.no_grad():
        clean_logits = model(images)
        if clean_logits.shape != (len(labels), 10) or not torch.isfinite(clean_logits).all():
            raise ValueError('Classifier must return finite [N,10] logits')
        correct = clean_logits.argmax(1) == labels
    optimizer.zero_grad(set_to_none=True)
    if not correct.any():
        return {'loss': None, 'clean_correct': 0, 'updated': False, 'gradient_l1': 0.}
    chosen = correct.nonzero().flatten().tolist()
    attacked = apply_patch(images[correct], patch, [positions[i] for i in chosen], area_fraction)
    logits = model(attacked)
    loss = F.cross_entropy(logits, labels[correct])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite patch objective')
    # Minimize negative CE; autograd targets the one shared patch only.
    patch.grad = torch.autograd.grad(-loss, patch)[0]
    if not torch.isfinite(patch.grad).all():
        raise FloatingPointError('Nonfinite patch gradient')
    gradient = float(patch.grad.abs().sum())
    optimizer.step()
    with torch.no_grad():
        patch.clamp_(0, 1)
    return {'loss': float(loss.detach()), 'clean_correct': len(chosen), 'updated': True, 'gradient_l1': gradient}


if __name__ == '__main__':
    from attacks.patch_cli import main
    main()
