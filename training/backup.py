"""Verified single-generation full backups. Full checkpoints must be trusted."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from training.checkpoint import atomic_save, read_checkpoint, write_json
from training.config import TrainConfig
from training.consistency import file_digest, reconcile
from training.storage import EXPORT_FILES, verify_owner, check_separate


class BackupError(RuntimeError):
    pass


def read_json(path):
    return json.loads(path.read_text())


def validate_state(state, directory, run_id):
    required = {'model', 'optimizer', 'scheduler', 'scaler', 'rng', 'provenance',
                'config', 'history', 'data_metadata', 'epoch', 'best_f1'}
    if not required.issubset(state):
        raise BackupError('Incomplete full checkpoint: missing resume state')
    if state.get('run_id', run_id) != run_id:
        raise BackupError('Checkpoint belongs to a different run')
    expected = TrainConfig(**read_json(directory / 'config.json')).to_dict()
    actual = TrainConfig(**state['config']).to_dict()
    for key in ('root', 'device'):
        expected.pop(key)
        actual.pop(key)
    if actual != expected or state['data_metadata'] != read_json(directory / 'data.json'):
        raise BackupError('Checkpoint configuration/dataset metadata do not match run')
    if (state['epoch'] + 1 != len(state['history']) or state['epoch'] < 0
            or [row['epoch'] for row in state['history']] != list(range(1, state['epoch'] + 2))):
        raise BackupError('Invalid completed epoch/history in checkpoint')


def sync_checkpoint(run, destination):
    if destination is None:
        return
    try:
        verify_owner(run, destination)
        run_id = read_json(run / 'storage.json')['run_id']
        source = run / 'last.pt'
        digest = file_digest(source)
        state = read_checkpoint(source)
        validate_state(state, run, run_id)
        record = {'run_id': run_id, 'completed_epoch': state['epoch'] + 1,
                  'sha256': digest, 'file_size_bytes': source.stat().st_size,
                  'checkpoint_format_version': state['format_version'],
                  'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                  'metadata_sha256': {name: file_digest(run / name) for name in
                                      ('config.json', 'data.json', 'preprocessing.json')}}
        if (destination / 'last.pt').is_file() and file_digest(destination / 'last.pt') == digest:
            # Reuse/repair a verified publication instead of uploading it again.
            try:
                verify_backup(destination)
                return
            except BackupError:
                pass
        def stage(temporary):
            shutil.copyfile(source, temporary)
            if file_digest(temporary) != digest:
                raise BackupError('Staging checkpoint SHA-256 mismatch')
            staged = read_checkpoint(temporary)
            validate_state(staged, run, run_id)
            if staged['epoch'] != state['epoch']:
                raise BackupError('Staging checkpoint epoch mismatch')
            # Durable intent proves integrity if publication precedes manifest.
            write_json(destination / 'backup_pending.json', record)
        atomic_save(destination / 'last.pt', stage)
        if file_digest(destination / 'last.pt') != digest:
            raise BackupError('Published checkpoint SHA-256 mismatch')
        write_json(destination / 'backup.json', record)
        if file_digest(destination / 'last.pt') != digest:
            raise BackupError('Published checkpoint changed after verification')
    except Exception as exc:
        raise BackupError(f'Full checkpoint backup failed; local last.pt retained. Restore Drive access/space and retry --resume: {exc}') from exc


def verify_backup(destination):
    """Recover stale/missing manifest only from a matching verified intent."""
    destination = Path(destination)
    try:
        owner = read_json(destination / 'storage.json')
        digest = file_digest(destination / 'last.pt')
        record = None
        for name in ('backup.json', 'backup_pending.json'):
            if (destination / name).exists():
                candidate = read_json(destination / name)
                if candidate['sha256'] == digest:
                    record = candidate
                    break
        if record is None:
            raise BackupError('No manifest or verified intent matches checkpoint SHA-256')
        if record['run_id'] != owner['run_id']:
            raise BackupError('Backup belongs to a different run')
        if record['file_size_bytes'] != (destination / 'last.pt').stat().st_size:
            raise BackupError('Backup size mismatch')
        for name in ('config.json', 'data.json', 'preprocessing.json'):
            if file_digest(destination / name) != record['metadata_sha256'][name]:
                raise BackupError(f'Run metadata checksum mismatch: {name}')
        state = read_checkpoint(destination / 'last.pt')
        validate_state(state, destination, owner['run_id'])
        if (state['epoch'] + 1 != record['completed_epoch'] or
                state['format_version'] != record['checkpoint_format_version']):
            raise BackupError('Backup epoch/format mismatch')
        if not (destination / 'backup.json').exists() or read_json(destination / 'backup.json') != record:
            write_json(destination / 'backup.json', record)
        return state
    except Exception as exc:
        raise BackupError(f'Cannot restore trusted full backup: {exc}') from exc


def restore_run(durable_dir, work_dir=None):
    destination = Path(durable_dir).resolve()
    state = verify_backup(destination)
    config = TrainConfig(**state['config'])
    if not config.economical:
        raise BackupError('restore-from requires an economical run')
    base = Path(work_dir or config.work_dir).resolve()
    check_separate(base, destination)
    check_separate(base, config.indices)
    run = base / config.model / destination.name
    run.mkdir(parents=True, exist_ok=False)
    # Exclusive directory: failures leave a partial directory for inspection;
    # never delete/adopt another run. Retry with a fresh work-dir.
    for name in (*EXPORT_FILES, 'last.pt'):
        source = destination / name
        if source.is_file():
            atomic_save(run / name, lambda p: shutil.copyfile(source, p))
    if file_digest(run / 'last.pt') != file_digest(destination / 'last.pt'):
        raise BackupError('Restored checkpoint checksum mismatch')
    for name, digest in read_json(destination / 'backup.json')['metadata_sha256'].items():
        if name not in ('config.json', 'data.json', 'preprocessing.json') or file_digest(run / name) != digest:
            raise BackupError('Restored metadata checksum mismatch')
    verify_owner(run, destination)
    validate_state(read_checkpoint(run / 'last.pt'), run,
                   read_json(run / 'storage.json')['run_id'])
    # Checkpoint provenance is authoritative if metadata sync was interrupted.
    write_json(run / 'provenance.json', state['provenance'])
    reconcile(run, state, True, destination)
    return run
