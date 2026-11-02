"""last.pt commits an epoch; staged weights are published/repaired from that commit."""
import hashlib
import json
import shutil
import torch
from training.checkpoint import atomic_save, export_state_dict, read_checkpoint, write_json


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        if value.is_floating_point():
            value = value.float()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matches(path, selection):
    try:
        return tensor_digest(torch.load(path, map_location='cpu', weights_only=True)) == selection['sha256']
    except Exception:
        return False


def reconcile(run, state, economical, durable=None):
    history = state['history']
    best = max(history, key=lambda row: row['validation']['macro_f1'])
    epoch, score = best['epoch'], best['validation']['macro_f1']
    selection = state.get('best_selection')
    full = None
    if (run / 'best.pt').is_file():
        try:
            candidate = read_checkpoint(run / 'best.pt')
            if candidate['epoch'] + 1 == epoch:
                full = candidate
        except Exception:
            pass
    if selection is None:  # Legacy: only a full checkpoint proves weight provenance.
        source = state if epoch == state['epoch'] + 1 else full
        if source is None:
            raise RuntimeError('Cannot verify legacy best weights: restore trusted best.pt or committed selection metadata')
        selection = {'epoch': epoch, 'score': score, 'sha256': tensor_digest(source['model'])}
    if selection['epoch'] != epoch or selection['score'] != score or state['best_f1'] != score:
        raise RuntimeError('Checkpoint best selection disagrees with committed history')
    if not matches(run / 'best_weights.pt', selection):
        sources = [run / 'best_pending.pt']
        if durable is not None:
            sources.append(durable / 'best_weights.pt')
        for source in sources:
            if matches(source, selection):
                atomic_save(run / 'best_weights.pt', lambda p: shutil.copyfile(source, p))
                break
        else:
            source = state if epoch == state['epoch'] + 1 else full
            if source is None or tensor_digest(source['model']) != selection['sha256']:
                raise RuntimeError('Cannot recover committed best weights: missing or corrupt best/pending/backup data')
            export_state_dict(run / 'best_weights.pt', source['model'])
    if not economical and (full is None or tensor_digest(full['model']) != selection['sha256']):
        if epoch != state['epoch'] + 1:
            raise RuntimeError('Cannot recover full best.pt for an earlier epoch')
        atomic_save(run / 'best.pt', lambda p: shutil.copyfile(run / 'last.pt', p))
    write_json(run / 'metrics.json', history)
    write_json(run / 'summary.json', {'completed_epochs': len(history),
               'best_validation_macro_f1': score, 'best_epoch': epoch,
               'smoke': state['config']['smoke'],
               'selection_metric': 'validation macro-F1, all 10 labels; first maximum'})
    write_json(run / 'commit.json', {'epoch': state['epoch'] + 1, 'best_selection': selection,
               'sha256': {name: file_digest(run / name) for name in
                          ('best_weights.pt', 'metrics.json', 'summary.json')}})
    return selection


def verify_commit(run):
    record = json.loads((run / 'commit.json').read_text())
    for name, digest in record['sha256'].items():
        if name not in ('best_weights.pt', 'metrics.json', 'summary.json') or file_digest(run / name) != digest:
            raise RuntimeError('Incomplete or corrupt result publication; resume from local last.pt')
    return record
