"""Exact ASR trajectories from ONE completed Square run; no model/data access."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid

from attacks.evaluation import code_provenance
from attacks.square_results import read_square_predictions
from training.checkpoint import atomic_save, write_json
from training.consistency import file_digest
from training.storage import check_separate

DISPLAY_CHECKPOINTS = (100, 250, 500, 750, 1000, 1500)


def compute_trajectory(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    if status['state'] != 'completed' or config['attack']['name'] != 'square':
        raise ValueError('Exact trajectory requires one completed Square run')
    cap = config['attack']['max_queries']
    if type(cap) is not int or cap < 1:
        raise ValueError('Invalid max_queries')
    rows = read_square_predictions(run)  # Legacy/missing indices fail closed.
    if not all(row['first_success_query_recorded'] for row in rows):
        raise ValueError('Exact first-success queries unavailable')
    denominator = sum(row['clean_correct'] for row in rows)
    histogram = [0]*(cap+1)
    for row in rows:
        if row['first_success_query'] is not None:
            histogram[row['first_success_query']] += 1
    cumulative, curve = 0, []
    for q in range(1, cap+1):
        cumulative += histogram[q]
        curve.append({'query_budget': q, 'successful_samples': cumulative,
                      'clean_correct_samples': denominator,
                      'asr': cumulative/denominator if denominator else None})
    metrics = json.loads((run / 'metrics.json').read_text())
    if denominator != metrics['clean_correct_count'] or curve[-1]['asr'] != metrics['asr']:
        raise ValueError('Trajectory endpoint/denominator disagrees with recorded ASR')
    return config, curve


def plot_trajectory(path, config, curve, checkpoints):
    # Imported only after CSV export: missing/broken plotting cannot affect attacks.
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                                'axes.labelsize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        figure = Figure(figsize=(6.4, 4.2), layout='constrained')
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        if curve[0]['asr'] is None:
            axes.text(.5, .5, 'ASR undefined: no clean-correct samples',
                      ha='center', va='center', transform=axes.transAxes)
        else:
            axes.step([r['query_budget'] for r in curve], [100*r['asr'] for r in curve],
                      where='post', color='#2459a6', linewidth=1.8)
            for index, q in enumerate(checkpoints):
                if q <= len(curve):
                    value = 100*curve[q-1]['asr']
                    axes.plot(q, value, 'o', color='#2459a6', markersize=4)
                    offset = (6 + 12*(index % 3)) * (-1 if value >= 85 else 1)
                    axes.annotate(f'Q{q}', (q, value), xytext=(0, offset),
                                  textcoords='offset points', fontsize=8, ha='center')
        axes.set(xlabel='Queries per sample (includes clean inference)', ylabel='ASR (%)',
                 xlim=(1, max(2, len(curve))), ylim=(0, 100),
                 title=f"{config['model']} · {config['variant']}\n"
                       f"epsilon={config['attack']['epsilon']:.5g}; M={len(curve)}; N_clean={curve[0]['clean_correct_samples']}")
        axes.grid(alpha=.25, linestyle='--')
        for extension in ('png', 'pdf'):
            atomic_save(path / f'asr_vs_queries.{extension}',
                        lambda p: figure.savefig(p, format=extension, dpi=300))
        figure.clear()


def export_trajectory(run, output, *, checkpoints=()):
    run, output = Path(run).resolve(), Path(output).resolve()
    check_separate(run, output)  # Never add postprocessing to immutable attack runs.
    if any(type(q) is not int or q not in DISPLAY_CHECKPOINTS for q in checkpoints):
        raise ValueError(f'Display checkpoints must be selected from {DISPLAY_CHECKPOINTS}')
    checkpoints = sorted(set(checkpoints))
    inputs = [run / name for name in ('config.json', 'subset.json', 'status.json', 'metrics.json', 'query_metrics.json')]
    inputs += sorted((run / 'predictions').glob('batch-*.csv')) + sorted((run / 'queries').glob('batch-*.json'))
    hashes = {p.relative_to(run).as_posix(): file_digest(p) for p in inputs}
    config, curve = compute_trajectory(run)
    if any(file_digest(run / name) != digest for name, digest in hashes.items()):
        raise ValueError('Source run changed during analysis')
    destination = output / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json(destination / 'postprocessing_status.json', {'state': 'running', 'attack_state': 'completed', 'started_utc': started})
    def csv_writer(path):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
    atomic_save(destination / 'trajectory.csv', csv_writer)
    try:
        plotting_version = importlib.metadata.version('matplotlib')
    except importlib.metadata.PackageNotFoundError:
        plotting_version = None
    write_json(destination / 'analysis.json', {'source_run': str(run), 'source_run_id': config['run_id'],
        'model': config['model'], 'variant': config['variant'], 'checkpoint_sha256': config['checkpoint_sha256'],
        'attack': config['attack'], 'selection': config['selection'], 'source_sha256': hashes,
        'provenance': code_provenance(), 'matplotlib_version': plotting_version,
        'denominator': curve[0]['clean_correct_samples'], 'endpoint_asr': curve[-1]['asr'],
        'interpretation': 'prefix of this single run only; not independently rescheduled budgets'})
    write_json(destination / 'checkpoints.json', [
        {'query_budget': q, 'available': q <= len(curve),
         'asr': curve[q-1]['asr'] if q <= len(curve) else None} for q in checkpoints])
    try:
        plot_trajectory(destination, config, curve, checkpoints)
    except Exception as error:
        write_json(destination / 'postprocessing_status.json', {'state': 'plot_failed', 'attack_state': 'completed',
            'csv_complete': True, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'error': f'{type(error).__name__}: {error}'})
    else:
        write_json(destination / 'postprocessing_status.json', {'state': 'completed', 'attack_state': 'completed',
            'csv_complete': True, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='One completed Square run; no combining runs')
    parser.add_argument('--output', type=Path, required=True, help='Separate analysis output root')
    parser.add_argument('--display-checkpoints', type=int, nargs='+', choices=DISPLAY_CHECKPOINTS, default=[])
    args = parser.parse_args()
    destination = export_trajectory(args.run, args.output, checkpoints=args.display_checkpoints)
    print(f'ANALYSIS_DIR={destination}', flush=True)
    if json.loads((destination / 'postprocessing_status.json').read_text())['state'] != 'completed':
        print('Plotting failed; trajectory CSV retained and attack results unchanged', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
