"""Read frozen indices, never prepare/split data or open holdout."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from data.visdrone import CropRecord, VisDroneCrops, make_loader, class_statistics


def load_records(path):
    records = []
    for raw in json.loads(Path(path).read_text()):
        record = CropRecord(**{**raw, 'box': tuple(raw['box'])})
        p = Path(record.image)
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 3 or p.parts[1] != 'images':
            raise ValueError(f'Unsafe source path: {record.image}')
        if (type(record.label) is not int or record.label not in range(10)
                or len(record.box) != 4 or any(type(v) is not int for v in record.box)
                or min(record.box) < 0 or record.box[2] <= record.box[0]
                or record.box[3] <= record.box[1]):
            raise ValueError(f'Invalid frozen crop: {record}')
        records.append(record)
    if not records:
        raise ValueError(f'Empty index: {path}')
    return records


def subset(records, size, seed):
    if size is None or size >= len(records):
        return records
    indices = sorted(random.Random(seed).sample(range(len(records)), size))
    return [records[i] for i in indices]


def make_datasets(config, train_transform, val_transform):
    base = Path(config.indices)
    # Only these two filenames are ever read. Hash inputs before subsetting.
    hashes = {name: hashlib.sha256((base / f'{name}.json').read_bytes()).hexdigest()
              for name in ('train', 'validation')}
    train = load_records(base / 'train.json')
    validation = load_records(base / 'validation.json')
    if {Path(r.image).stem for r in train} & {Path(r.image).stem for r in validation}:
        raise ValueError('Train/validation source image IDs overlap')
    size = config.train_size
    if config.smoke:
        size = min(size or 2 * config.batch_size, 2 * config.batch_size)
        validation = subset(validation, 2 * config.batch_size, config.seed)
    train = subset(train, size, config.seed)
    # Fingerprints retain image identity, box, label and annotation line.
    metadata = {'index_sha256': hashes, 'splits': {}}
    for name, records in (('train', train), ('validation', validation)):
        metadata['splits'][name] = {
            **class_statistics(records),
            'selected_records_sha256': hashlib.sha256(json.dumps(
                [asdict(r) for r in records], sort_keys=True).encode()).hexdigest()}
    return (VisDroneCrops(config.root, train, train_transform),
            VisDroneCrops(config.root, validation, val_transform), metadata)


def epoch_loaders(config, train, validation, epoch):
    # Fresh worker processes each epoch; order AND augmentation seeds depend only
    # on epoch. An epoch-boundary restart reproduces uninterrupted data loading.
    return (make_loader(train, config.batch_size, True, config.workers, config.seed + epoch),
            make_loader(validation, config.batch_size, False, config.workers, config.seed))
