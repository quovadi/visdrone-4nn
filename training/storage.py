"""Small explicit allowlist for durable export; never mirror full checkpoints."""
import json
from pathlib import Path
import shutil
import uuid

from training.checkpoint import atomic_save, write_json

EXPORT_FILES = ('config.json', 'provenance.json', 'data.json', 'preprocessing.json',
                'metrics.json', 'summary.json', 'storage.json', 'source.zip',
                'environment.txt', 'best_weights.pt')


def check_separate(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('Storage paths must be separate directory trees')


def durable_directory(config, run, resume=False):
    if not config.economical:
        return None
    if resume:
        record = json.loads((run / 'storage.json').read_text())
        destination = Path(record['durable_directory'])
        verify_owner(run, destination)
        return destination
    destination = Path(config.output).resolve() / config.model / run.name
    destination.mkdir(parents=True, exist_ok=False)
    record = {'run_id': uuid.uuid4().hex, 'durable_directory': str(destination)}
    # Claim the new directory exclusively; never adopt an existing run.
    write_json(destination / 'storage.json', record)
    write_json(run / 'storage.json', record)
    return destination


def verify_owner(run, destination):
    local = json.loads((run / 'storage.json').read_text())
    remote = json.loads((destination / 'storage.json').read_text())
    if local != remote:
        raise ValueError('Durable directory belongs to a different run')


def sync_results(run, destination, weights=True):
    if destination is None:
        return
    verify_owner(run, destination)
    for name in EXPORT_FILES:
        if name == 'best_weights.pt' and not weights:
            continue
        source = run / name
        if source.is_file():
            atomic_save(destination / name, lambda p: shutil.copyfile(source, p))
