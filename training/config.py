"""Explicit, persisted training protocol. No implicit architecture fallback."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math

PRESETS = Path(__file__).resolve().parents[1] / 'configs/baseline.json'


@dataclass(frozen=True)
class TrainConfig:
    model: str
    model_id: str
    root: str
    indices: str
    output: str
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.05
    workers: int = 2
    seed: int = 42
    train_size: int | None = None
    smoke: bool = False
    amp: bool = True
    device: str = 'auto'

    def __post_init__(self):
        for name in ('batch_size', 'epochs'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.workers < 0 or self.seed < 0:
            raise ValueError('workers and seed must be non-negative')
        if self.train_size is not None and self.train_size < 1:
            raise ValueError('train_size must be positive')
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('lr must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and non-negative')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')

    def to_dict(self):
        return asdict(self)


def new_config(model, **overrides):
    presets = json.loads(PRESETS.read_text())
    if model not in presets['models']:
        raise ValueError(f'Unknown model {model}; choose {list(presets["models"])}')
    values = {**presets['defaults'], **overrides}
    return TrainConfig(model=model, model_id=presets['models'][model], **values)
