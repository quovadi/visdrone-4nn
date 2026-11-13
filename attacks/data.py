"""Validation only. Identities include the frozen index hash and original row."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from data.visdrone import VisDroneCrops, CLASSES
from training.data import load_records


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def selected_validation(index, seed, size=None):
    index = Path(index)
    if index.name != 'validation.json':
        raise ValueError('Only the frozen validation.json is permitted')
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    records = load_records(index)  # no train/holdout access
    if hashlib.sha256(index.read_bytes()).hexdigest() != before:
        raise ValueError('Validation index changed during reading')
    if size is not None and (type(size) is not int or not 1 <= size <= len(records)):
        raise ValueError('subset-size must be within the validation population')
    positions = sorted(random.Random(seed).sample(range(len(records)), size)) if size else list(range(len(records)))
    selected, identities = [], []
    for position in positions:
        record = records[position]
        raw = asdict(record)
        sample_id = f'{before}:{position}:{canonical_hash(raw)}'
        identities.append({'sample_id': sample_id, 'index_row': position, **raw,
                           'attack_seed': int(canonical_hash([seed, sample_id])[:15], 16)})
        selected.append(record)
    return selected, {'index_sha256': before, 'population': len(records), 'size': len(selected),
                      'selection': 'sorted random.Random(seed).sample of original row positions; full if size=None',
                      'seed': seed, 'classes': list(CLASSES), 'objects': identities,
                      'subset_sha256': canonical_hash(identities)}


def dataset(root, records, transform):
    return VisDroneCrops(root, records, transform=transform)
