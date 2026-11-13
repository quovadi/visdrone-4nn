"""Preserve timm eval geometry; move its only normalization into the graph."""
import math
import torch
from torch import nn
from torchvision.transforms import Compose, Normalize
from training.models import make_transforms


def rgb_transform(data_config):
    if tuple(data_config['input_size']) != (3, 224, 224):
        raise ValueError('Evaluation protocol requires 3x224x224')
    _, evaluation = make_transforms(data_config)
    operations = list(evaluation.transforms)
    normalizations = [op for op in operations if isinstance(op, Normalize)]
    if len(normalizations) != 1 or operations[-1] is not normalizations[0]:
        raise ValueError('Unsupported timm transform: expected exactly one final Normalize')
    normalization = normalizations[0]
    if tuple(normalization.mean) != tuple(data_config['mean']) or tuple(normalization.std) != tuple(data_config['std']):
        raise ValueError('timm normalization disagrees with recorded data config')
    return Compose(operations[:-1])


class PixelModel(nn.Module):
    """Classifier on RGB [0,1], with differentiable model-specific normalization."""
    def __init__(self, model, data_config):
        super().__init__()
        self.classifier = model
        mean, std = data_config['mean'], data_config['std']
        if (len(mean) != 3 or len(std) != 3 or not all(math.isfinite(v) for v in (*mean, *std))
                or min(std) <= 0):
            raise ValueError('Invalid RGB mean/std')
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, rgb):
        return self.classifier((rgb - self.mean) / self.std)
