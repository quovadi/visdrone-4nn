"""Separate official-train optimization and frozen Stage 5 validation manifest."""
from dataclasses import asdict
import json
from pathlib import Path
import random
from attacks.data import canonical_hash, selected_validation
from data.visdrone import CLASSES
from training.data import load_records
from training.storage import check_separate
from training.consistency import file_digest
from training.checkpoint import write_json
from attacks.patch_io import new_run, finish, verify

IDENTITY_FIELDS = ('sample_id', 'index_row', 'image', 'box', 'label', 'line')


def read_train(index, size=None, seed=42):
    index = Path(index)
    if index.name != 'train.json':
        raise ValueError('Patch optimization requires train.json, never validation')
    sha = file_digest(index)
    records = load_records(index)
    if any(Path(r.image).parts[0] != 'VisDrone2019-DET-train' for r in records):
        raise ValueError('Optimization must use official DET train images only')
    if size is not None and (type(size) is not int or not 1 <= size <= len(records)):
        raise ValueError('Invalid patch-training subset size')
    rows = sorted(random.Random(seed).sample(range(len(records)), size)) if size else list(range(len(records)))
    objects = []
    for i in rows:
        raw = asdict(records[i])
        objects.append({'sample_id': f'{sha}:{i}:{canonical_hash(raw)}', 'index_row': i, **raw})
    if file_digest(index) != sha:
        raise ValueError('Train index changed while reading')
    return [records[i] for i in rows], {'index_sha256': sha, 'population': len(records),
        'size': len(rows), 'seed': seed, 'objects': objects, 'subset_sha256': canonical_hash(objects),
        'classes': list(CLASSES), 'selection': 'sorted uniform train-only row sample, no size/class filtering'}


def validation_population(index, seed):
    records, manifest = selected_validation(index, seed)
    if any(Path(r.image).parts[0] != 'VisDrone2019-DET-val' for r in records):
        raise ValueError('Validation must use official DET val images only')
    return records, manifest


def prepare_subset(index, output, size=1024, seed=42, square_subset=None):
    check_separate(Path(output)/'subsets',Path(index).parent)
    records, population = validation_population(index, seed)
    if type(size) is not int or not 1 <= size <= len(records):
        raise ValueError('Requested Stage 5 subset exceeds validation population')
    objects = population['objects']
    by_id = {o['sample_id']: o for o in objects}
    inherited, inheritance = [], None
    if square_subset is not None:
        previous = json.loads(Path(square_subset).read_text())
        if (previous['index_sha256'] != population['index_sha256']
                or canonical_hash(previous['objects']) != previous['subset_sha256']):
            raise ValueError('Square subset index/hash mismatch')
        inherited = [o['sample_id'] for o in previous['objects']]
        if len(inherited) != len(set(inherited)) or len(inherited) > size:
            raise ValueError('Invalid inherited Square subset size/duplicates')
        for old in previous['objects']:
            if old['sample_id'] not in by_id or canonical_hash({k:old[k] for k in IDENTITY_FIELDS}) != canonical_hash({k:by_id[old['sample_id']][k] for k in IDENTITY_FIELDS}):
                raise ValueError('Square subset object differs from frozen validation')
        inheritance = {'path': str(Path(square_subset).resolve()), 'sha256': file_digest(Path(square_subset)),
                       'retained_sample_ids': inherited, 'retained_count': len(inherited)}
    chosen = set(inherited)
    groups = {label: [o for o in objects if o['label'] == label and o['sample_id'] not in chosen] for label in range(10)}
    needed = size-len(chosen)
    total = sum(map(len, groups.values()))
    quotas = {label: needed*len(groups[label])//total if total else 0 for label in range(10)}
    remainder = needed-sum(quotas.values())
    order = sorted(range(10), key=lambda label: (-(needed*len(groups[label]) % total) if total else 0, label))
    for label in order[:remainder]:
        quotas[label] += 1
    for label, candidates in groups.items():
        rng = random.Random(int(canonical_hash([seed, label])[:16], 16))
        chosen.update(o['sample_id'] for o in rng.sample(candidates, quotas[label]))
    selected = [o for o in objects if o['sample_id'] in chosen]
    manifest = {**{k:v for k,v in population.items() if k not in ('objects', 'subset_sha256', 'selection', 'size')},
        'objects': selected, 'size': len(selected), 'subset_sha256': canonical_hash(selected),
        'selection': 'retain supplied Square IDs; Hamilton proportional strata over remaining ground-truth classes',
        'class_distribution': {name: sum(o['label'] == i for o in selected) for i,name in enumerate(CLASSES)},
        'additional_class_quotas': quotas, 'inheritance': inheritance}
    run = new_run(output, 'subsets', {'validation_index': str(Path(index).resolve()), 'size': size, 'seed': seed})
    write_json(run / 'subset.json', manifest)
    finish(run)
    return run / 'subset.json'


def read_subset(index, subset):
    subset = Path(subset).resolve()
    if subset.name != 'subset.json':
        raise ValueError('Expected Stage 5 subset.json artifact location')
    bundle = verify(subset.parent, 'subsets')
    if subset.parent.name != bundle['run_id'] or subset.parent.parent.name != 'subsets':
        raise ValueError('Frozen Stage 5 subset must remain in its published subsets/<run_id> bundle')
    saved = json.loads(subset.read_text())
    records, current = validation_population(index, saved['seed'])
    if saved['index_sha256'] != current['index_sha256'] or canonical_hash(saved['objects']) != saved['subset_sha256']:
        raise ValueError('Frozen Stage 5 subset/index checksum mismatch')
    selected, seen = [], set()
    for obj in saved['objects']:
        i = obj['index_row']
        if type(i) is not int or not 0 <= i < len(records):
            raise ValueError('Invalid frozen subset row')
        expected = current['objects'][i]
        if obj['sample_id'] in seen or canonical_hash(obj) != canonical_hash(expected):
            raise ValueError('Frozen Stage 5 object identity mismatch')
        seen.add(obj['sample_id'])
        selected.append(records[i])
    distribution = {name: sum(o['label']==i for o in saved['objects']) for i,name in enumerate(CLASSES)}
    if saved['population'] != len(records) or saved['classes'] != list(CLASSES) or saved['class_distribution'] != distribution:
        raise ValueError('Frozen Stage 5 class distribution/population mismatch')
    if len(selected) != saved['size'] or not selected:
        raise ValueError('Frozen Stage 5 subset size mismatch')
    return selected, saved
