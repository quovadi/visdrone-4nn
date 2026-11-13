"""One interface for FGSM and single-start Linf PGD; no parameter updates."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import warnings

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttackConfig:
    name: str
    epsilon: float
    steps: int = 20
    seed: int = 42

    def __post_init__(self):
        if self.name not in ('fgsm', 'pgd'):
            raise ValueError('Only fgsm and pgd are implemented')
        if not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError('epsilon must be finite in [0,1] pixel units')
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')

    def parameters(self):
        return {**asdict(self), 'steps': 1 if self.name == 'fgsm' else self.steps,
                'control': self.epsilon == 0, 'norm': 'linf', 'targeted': False,
                'alpha': self.epsilon if self.name == 'fgsm' else self.epsilon / 4,
                'iterations': 1 if self.name == 'fgsm' else self.steps,
                'random_start': self.name == 'pgd', 'restarts': 1, 'precision': 'fp32'}


@contextmanager
def evaluation_mode(model):
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def check_bounds(original, adversarial, epsilon):
    if (original.shape != adversarial.shape or adversarial.dtype != torch.float32
            or not torch.isfinite(adversarial).all()):
        raise ValueError('Invalid adversarial tensor shape/dtype/finite values')
    distances = (adversarial - original).abs().flatten(1).amax(1)
    if (adversarial.min() < 0 or adversarial.max() > 1
            or torch.any(distances > epsilon + 1e-6)):
        raise ValueError('Attack violated RGB range or Linf budget')
    return distances


def project(candidate, original, epsilon):
    return torch.maximum(torch.minimum(candidate, original + epsilon), original - epsilon).clamp(0, 1)


def generate(model, images, labels, config, *, sample_seeds=None, observer=None, diagnostic=None):
    """Return detached RGB. observer(step, x) includes PGD's initial point (0).

    autograd.grad targets inputs only; existing parameter gradients/flags and
    global RNG are preserved. No optimizer, AMP or parameter dtype conversion.
    """
    if (images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0
            or images.dtype != torch.float32 or not torch.isfinite(images).all()
            or images.min() < 0 or images.max() > 1):
        raise ValueError('Expected finite FP32 NCHW RGB in [0,1]')
    if labels.shape != (len(images),) or labels.dtype != torch.int64:
        raise ValueError('Expected int64 labels with shape [N]')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Attacks require FP32 model parameters')
    original = images.detach().clone()
    if config.epsilon == 0:
        return original

    def report(kind, **details):
        event = {'kind': kind, **details}
        if diagnostic:
            diagnostic(event)
        else:
            warnings.warn(str(event), RuntimeWarning, stacklevel=2)

    with evaluation_mode(model), torch.enable_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        current = original.clone()
        if config.name == 'pgd':
            seeds = sample_seeds if sample_seeds is not None else [config.seed + i for i in range(len(images))]
            if len(seeds) != len(images):
                raise ValueError('One random-start seed per image is required')
            noise = torch.stack([torch.rand(original.shape[1:], generator=torch.Generator().manual_seed(int(seed)))
                                 for seed in seeds]).to(images.device)
            current = project(original + (noise * 2 - 1) * config.epsilon, original, config.epsilon)
            check_bounds(original, current, config.epsilon)
            if observer:
                observer(0, current.detach().clone())
        for step in range(1 if config.name == 'fgsm' else config.steps):
            current = current.detach().requires_grad_(True)
            logits = model(current)
            if not torch.isfinite(logits).all():
                report('nonfinite_logits', step=step)
                raise FloatingPointError('Non-finite attack logits')
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                report('nonfinite_loss', step=step)
                raise FloatingPointError('Non-finite attack loss')
            if not loss.requires_grad:
                report('disconnected_input_gradient', step=step)
                raise RuntimeError('Loss is disconnected from input RGB')
            gradient = torch.autograd.grad(loss, current, only_inputs=True, allow_unused=True)[0]
            if gradient is None or not torch.isfinite(gradient).all():
                report('invalid_input_gradient', step=step)
                raise FloatingPointError('Missing or non-finite RGB gradient')
            zero = gradient.flatten(1).abs().amax(1) == 0
            if zero.any():
                report('zero_input_gradient', step=step, batch_positions=zero.nonzero().flatten().tolist())
            alpha = config.epsilon if config.name == 'fgsm' else config.epsilon / 4
            current = project(current.detach() + alpha * gradient.sign(), original, config.epsilon).detach()
            check_bounds(original, current, config.epsilon)
            if observer:
                observer(step + 1, current.clone())
        return current.detach()
