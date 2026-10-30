"""Image-level splits and validated object crops, without materializing crops."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor')
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


@dataclass(frozen=True)
class CropRecord:
    image: str  # relative to dataset root, including official split
    box: tuple[int, int, int, int]  # clipped integer xyxy
    label: int  # 0..9; original DET category minus one
    line: int


def discover_images(root, split):
    folder = Path(root) / split / 'images'
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    images = sorted(p.relative_to(root).as_posix() for p in folder.iterdir()
                    if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f'No images in {folder}')
    stems = [Path(p).stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError('Ambiguous image stems within split')
    return images


def make_splits(root, train_split, val_split, holdout_fraction=0.2, seed=42):
    """Hold out images from official val; never use unlabelled test-dev as test.

    Random image-level split is deterministic but not class-stratified. Inspect
    class support before freezing it. It cannot prevent cross-image scene leakage.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must be between 0 and 1')
    train = discover_images(root, train_split)
    val = discover_images(root, val_split)
    # Also reject matching source IDs across official partitions.
    if {Path(p).stem for p in train} & {Path(p).stem for p in val}:
        raise ValueError('Source image IDs overlap between official splits')
    if len(val) < 2:
        raise ValueError('Need at least two labelled validation images')
    for name in train + val:
        annotation_path(root, name).read_text()  # fail on unlabelled splits
    random.Random(seed).shuffle(val)
    n = max(1, min(len(val) - 1, round(len(val) * holdout_fraction)))
    return {'train': train, 'validation': sorted(val[n:]), 'holdout': sorted(val[:n])}


def annotation_path(root, image):
    p = Path(image)
    if p.is_absolute() or '..' in p.parts or len(p.parts) != 3 or p.parts[1] != 'images':
        raise ValueError(f'Expected split/images/file, got {image}')
    return Path(root) / p.parts[0] / 'annotations' / (p.stem + '.txt')


def build_index(root, images):
    """Strict 8-column DET parser. Truncation/occlusion do not remove targets.

    Reject malformed rows; skip ignored (0), others (11), score=0, invalid
    scores/categories/boxes. Partial out-of-frame boxes are clipped. Fractional
    bounds use floor(left/top), ceil(right/bottom). Missing annotations or corrupt
    images fail loudly rather than silently losing sources.
    """
    records, counts = [], Counter()
    if len(images) != len(set(images)):
        raise ValueError('Duplicate source images')
    for name in sorted(images):
        ann = annotation_path(root, name)
        with Image.open(Path(root) / name) as im:
            im.load()
            width, height = im.size
        counts['images'] += 1
        for line, raw in enumerate(ann.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            counts['rows'] += 1
            try:
                fields = [v.strip() for v in raw.split(',')]
                if len(fields) != 8:
                    raise ValueError()
                values = [float(v) for v in fields]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError()
                x, y, w, h, score, category, truncation, occlusion = values
                if any(v != int(v) for v in (score, category, truncation, occlusion)):
                    raise ValueError()
                if truncation not in (0, 1, 2) or occlusion not in (0, 1, 2):
                    raise ValueError()
            except ValueError:
                counts['malformed'] += 1
                continue
            if score == 0:
                counts['score_zero'] += 1
                continue
            if category in (0, 11):
                counts['ignored_or_others'] += 1
                continue
            if score != 1 or category not in range(1, 11):
                counts['invalid_label_or_score'] += 1
                continue
            if w <= 0 or h <= 0:
                counts['invalid_box'] += 1
                continue
            box = (max(0, math.floor(x)), max(0, math.floor(y)),
                   min(width, math.ceil(x + w)), min(height, math.ceil(y + h)))
            if box[2] <= box[0] or box[3] <= box[1]:
                counts['invalid_box'] += 1
                continue
            records.append(CropRecord(name, box, int(category) - 1, line))
            counts['accepted'] += 1
    return records, dict(counts)


class ClassificationTransform:
    """Deterministic baseline resize; no model normalization at data stage.

    Produces RGB float32 CHW in [0,1]. Later stages must use each pretrained
    model's timm data config; this resize is for data validation only.
    """
    def __init__(self, size=224):
        if size <= 0:
            raise ValueError('size must be positive')
        self.size = size

    def __call__(self, image):
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255


class VisDroneCrops(Dataset):
    def __init__(self, root, records, transform=None):
        self.root = Path(root)
        self.records = list(records)
        self.transform = transform if transform is not None else ClassificationTransform()

    def __len__(self):
        return len(self.records)

    def raw_crop(self, index):
        record = self.records[index]
        with Image.open(self.root / record.image) as im:
            # Do not EXIF-transpose: DET coordinates refer to stored pixel axes.
            return im.convert('RGB').crop(record.box)

    def __getitem__(self, index):
        return self.transform(self.raw_crop(index)), self.records[index].label


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size=32, shuffle=False, workers=0, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available())


def class_statistics(records):
    counts = Counter(r.label for r in records)
    sources = {r.image for r in records}
    return {'crops': len(records), 'images_with_crops': len(sources),
            'classes': {name: counts[i] for i, name in enumerate(CLASSES)}}


def visualize(dataset, output, count=12, seed=42):
    import matplotlib.pyplot as plt
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    if not indices:
        raise ValueError('No crops to visualize')
    fig, axes = plt.subplots(math.ceil(len(indices) / 4), 4, squeeze=False,
                             figsize=(12, 3 * math.ceil(len(indices) / 4)))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, indices):
        ax.imshow(dataset.raw_crop(index))
        ax.set_title(CLASSES[dataset.records[index].label])
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def prepare(root, output, train_split='VisDrone2019-DET-train',
            val_split='VisDrone2019-DET-val', holdout_fraction=0.2, seed=42):
    output = Path(output)
    # Freeze partitions: refuse accidental regeneration/overwriting results.
    output.mkdir(parents=True, exist_ok=False)
    splits = make_splits(root, train_split, val_split, holdout_fraction, seed)
    report = {'seed': seed, 'holdout_fraction': holdout_fraction, 'splits': {}}
    manifest = {'schema_version': 1, 'splits': splits, 'annotation_sha256': {}}
    for name, images in splits.items():
        records, rejected = build_index(root, images)
        if not records:
            raise ValueError(f'No valid objects in {name}')
        report['splits'][name] = {**class_statistics(records), 'parser': rejected,
                                  'source_images': len(images)}
        (output / f'{name}.json').write_text(json.dumps([asdict(r) for r in records], indent=2))
        visualize(VisDroneCrops(root, records), output / f'{name}.png', seed=seed)
        for image in images:
            manifest['annotation_sha256'][image] = hashlib.sha256(annotation_path(root, image).read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'statistics.json').write_text(json.dumps(report, indent=2))
    return report
