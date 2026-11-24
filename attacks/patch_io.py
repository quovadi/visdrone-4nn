"""Patch-specific artifact schema using shared atomic writes and SHA checks."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import re
from training.storage import check_separate
import uuid
from attacks.evaluation import code_provenance
from training.checkpoint import atomic_save, write_json
from training.consistency import file_digest


def now():
    return datetime.now(timezone.utc).isoformat()


def new_run(output, kind, config):
    run = Path(output) / kind / uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'config.json', {**config, 'kind': kind, 'run_id': run.name,
                                   'started_utc': now(), 'provenance': code_provenance()})
    write_json(run / 'status.json', {'state': 'running'})
    return run


def write_csv(path, rows, fields):
    def writer(p):
        with p.open('w', newline='') as stream:
            out = csv.DictWriter(stream, fieldnames=fields)
            out.writeheader()
            out.writerows(rows)
    atomic_save(path, writer)


def finish(run):
    write_json(run / 'status.json', {'state': 'completed', 'finished_utc': now()})
    config = json.loads((run / 'config.json').read_text())
    files = {p.relative_to(run).as_posix(): file_digest(p) for p in sorted(run.rglob('*'))
             if p.is_file() and p.name != 'run_manifest.json' and not p.name.endswith('.tmp')}
    write_json(run / 'run_manifest.json', {'run_id': run.name, 'kind': config['kind'], 'sha256': files})


def verify(run, kind=None):
    run = Path(run)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if (config['kind'] not in ('patches','evaluations','subsets','reports')
            or not re.fullmatch(r'[0-9a-f]{32}',config['run_id'])
            or manifest['run_id'] != config['run_id'] or manifest['kind'] != config['kind']
            or kind is not None and manifest['kind'] != kind
            or json.loads((run / 'status.json').read_text())['state'] != 'completed'):
        raise ValueError('Incomplete or foreign Stage 5 artifact')
    required = {'config.json', 'status.json'}
    if config['kind'] == 'patches':
        required |= {'patch.pt', 'patch.png', 'metadata.json', 'training_loss.csv', 'last_patch.pt', 'train_subset.json'}
    if config['kind'] == 'evaluations':
        required |= {'predictions.csv', 'metrics.json', 'per_class_metrics.csv', 'metadata.json'}
    if config['kind'] == 'subsets':
        required |= {'subset.json'}
    if config['kind'] == 'reports':
        required |= {'comparison.csv','transfer_matrices.json','plot_status.json'}
    if not required.issubset(manifest['sha256']):
        raise ValueError('Artifact manifest lacks required files')
    for name, digest in manifest['sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or file_digest(run / path) != digest:
            raise ValueError('Stage 5 artifact checksum mismatch')
    return config


def backup_bundle(run, backup_root):
    """Completed patch bundle backup, not classifier-state backup masquerading."""
    run = Path(run)
    config = verify(run)
    check_separate(run,backup_root)
    destination = Path(backup_root) / config['kind'] / config['run_id']
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        def copy(p):
            shutil.copyfile(run / name, p)
            if file_digest(p) != digest:
                raise OSError('Patch backup copy checksum mismatch')
        atomic_save(target, copy)
    atomic_save(destination / 'run_manifest.json', lambda p: shutil.copyfile(run / 'run_manifest.json', p))
    verify(destination)
    return destination
