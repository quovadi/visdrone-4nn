"""Square query records supplement the unchanged paired-prediction schema."""
import json
import math
from pathlib import Path
from attacks.results import read_predictions


def query_summary(records):
    counts = [r['queries'] for r in records]
    correct = [r['queries'] for r in records if r['clean_correct']]
    successful = [r['queries'] for r in records if r['success']]
    return {'clean_correct_samples': len(correct),
            'mean_queries_clean_correct': sum(correct)/len(correct) if correct else None,
            'mean_queries_successful': sum(successful)/len(successful) if successful else None,
            'samples': len(counts), 'total_queries': sum(counts),
            'mean_queries': sum(counts)/len(counts) if counts else None,
            'max_queries_used': max(counts) if counts else 0,
            'successful_samples': sum(r['success'] for r in records)}


def read_square_predictions(run, *, allow_stale_partial=False):
    run = Path(run)
    rows = read_predictions(run, allow_stale_partial=allow_stale_partial)
    config = json.loads((run / 'config.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    files = sorted((run / 'queries').glob('batch-*.json'))
    csv_files = sorted((run / 'predictions').glob('batch-*.csv'))
    if [p.stem for p in files[:len(csv_files)]] != [p.stem for p in csv_files]:
        raise ValueError('Missing Square query chunk')
    if status['state'] == 'completed' and len(files) != len(csv_files):
        raise ValueError('Completed Square query coverage mismatch')
    records = [r for p in files[:len(csv_files)] for r in json.loads(p.read_text())]
    if len(records) != len(rows):
        raise ValueError('Square query count/row coverage mismatch')
    for row, record in zip(rows, records):
        if (record['sample_id'] != row['sample_id'] or record['run_id'] != config['run_id']
                or record['checkpoint_sha256'] != config['checkpoint_sha256']):
            raise ValueError('Square query identity mismatch')
        q = record['queries']
        if type(q) is not int or not 1 <= q <= config['attack']['max_queries']:
            raise ValueError('Invalid Square query budget')
        if (record['final_prediction'] != row['adversarial_prediction'] or record['linf'] != row['linf']
                or record['clean_correct'] != row['clean_correct']
                or not math.isfinite(record['final_margin'])
                or type(record['accepted_updates']) is not int
                or not 0 <= record['accepted_updates'] <= max(0, q-2)):
            raise ValueError('Square diagnostics disagree with predictions/queries')
        if record['success'] != (row['attack_success'] is True):
            raise ValueError('Square success disagrees with paired predictions')
        reason = record['stop_reason']
        if (reason not in ('clean_incorrect', 'epsilon_zero', 'success', 'budget', 'no_effective_proposal')
                or (reason == 'budget' and q != config['attack']['max_queries'])
                or (reason == 'success') != record['success']
                or (not row['clean_correct'] and (reason != 'clean_incorrect' or q != 1))
                or (row['epsilon'] == 0 and row['clean_correct'] and (reason != 'epsilon_zero' or q != 1))):
            raise ValueError('Square stop reason/query inconsistency')
        row.update(record)
    path = run / 'query_metrics.json'
    if status['state'] == 'completed' and (not path.exists() or json.loads(path.read_text()) != query_summary(records)):
        raise ValueError('Square aggregate query metrics mismatch')
    return rows
