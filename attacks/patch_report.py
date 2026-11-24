"""Aggregate one coherent 4x3x4 experiment, or an explicitly partial pilot."""
import json
from pathlib import Path
import numpy as np
from attacks.adversarial_patch import AREAS, MODELS
from attacks.patch_evaluation import read_evaluation
from attacks.patch_training import load_patch
from attacks.patch_io import new_run, finish, write_csv
from training.checkpoint import atomic_save, write_json
from training.consistency import file_digest
from training.storage import check_separate


def transfer_matrices(evaluations, allow_partial=False):
    matrices = {area: np.full((4,4), np.nan) for area in AREAS}
    seen, target_sha, source_sha, clean_predictions = set(), {}, {}, {}
    cohort = None
    records = []
    for run in evaluations:
        config, rows, metrics = read_evaluation(run)
        if not allow_partial and len(rows)!=1024:
            raise ValueError('Full Stage 5 report requires the fixed 1024-object cohort')
        source, target = config['source']['source_model'], config['target']['model']
        if not config['target'].get('synthetic',False) and (config['target'].get('variant') != 'baseline-v1'
                or config['source'].get('source_variant') != 'baseline-v1'):
            raise ValueError('Stage 5 matrix requires baseline-v1, not mixed variants')
        area = config['source']['area_fraction']
        key = (source, target, area)
        if source not in MODELS or target not in MODELS or area not in AREAS or key in seen:
            raise ValueError('Unknown/duplicate source-target-area configuration')
        seen.add(key)
        identity = (config['subset_sha256'], config['placement_seed'], tuple(config['data_config']['input_size']))
        if cohort is None:
            cohort = identity
        if cohort != identity:
            raise ValueError('Transfer runs use different subsets/placements/geometries')
        for registry, name, value in ((target_sha, target, config['target']['checkpoint_sha256']),
                                      (source_sha, source, config['source']['source_checkpoint_sha256'])):
            if name in registry and registry[name] != value:
                raise ValueError('Different checkpoints mixed for one architecture')
            registry[name] = value
        predictions = [(r['sample_id'], r['clean_prediction']) for r in rows]
        if target in clean_predictions and clean_predictions[target] != predictions:
            raise ValueError('Target clean predictions changed between patch evaluations')
        clean_predictions[target] = predictions
        value = metrics['asr']
        matrices[area][MODELS.index(source), MODELS.index(target)] = np.nan if value is None else value
        records.append({'synthetic':config['target'].get('synthetic',False),'source':source,'target':target,'area_fraction':area,
                        **{k:metrics[k] for k in ('asr','clean_accuracy','adversarial_accuracy','adversarial_macro_f1')},
                        'run':str(Path(run).resolve()),'patch_sha256':config['patch_sha256']})
    if not seen or (not allow_partial and len(seen) != 48):
        raise ValueError('Full report requires 48 unique combinations; explicitly allow a partial pilot')
    for name in source_sha.keys() & target_sha.keys():
        if source_sha[name] != target_sha[name]:
            raise ValueError('Source/target baseline checkpoint mismatch on diagonal')
    return matrices, records


def render_report(run, matrices, records, patches):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    def save(fig, stem):
        FigureCanvasAgg(fig)
        for ext in ('png','pdf'):
            atomic_save(run / f'{stem}.{ext}', lambda p:fig.savefig(p, format=ext, dpi=300))
        fig.clear()
    prefix = 'SYNTHETIC · ' if any(r['synthetic'] for r in records) else ''
    with matplotlib.rc_context({'font.size':9,'pdf.fonttype':42}):
        for area, matrix in matrices.items():
            fig = Figure(figsize=(6,5),layout='constrained')
            ax = fig.subplots()
            image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0,vmax=1,cmap='viridis')
            ax.set_xticks(range(4), MODELS, rotation=30, ha='right')
            ax.set_yticks(range(4), MODELS)
            ax.set(xlabel='Target classifier', ylabel='Patch source', title=f'{prefix}Transfer ASR, area={area:.0%}')
            for i in range(4):
                for j in range(4):
                    v = matrix[i,j]
                    ax.text(j,i,'—' if np.isnan(v) else f'{v:.1%}',ha='center',va='center',color='red' if np.isnan(v) else 'white')
            fig.colorbar(image,ax=ax,label='ASR (fraction)')
            save(fig,f'transfer_{area:.2f}')
        for metric in ('asr','adversarial_accuracy','adversarial_macro_f1'):
            fig = Figure(figsize=(7,4.5),layout='constrained')
            ax = fig.subplots()
            for i, area in enumerate(AREAS):
                values = [next((r[metric] for r in records if r['source']==name and r['target']==name and r['area_fraction']==area),None) for name in MODELS]
                ax.bar(np.arange(4)+(i-1)*.25,[np.nan if v is None else v for v in values],width=.25,label=f'{area:.0%}')
            ax.set_xticks(range(4),MODELS,rotation=20,ha='right')
            ax.set(ylabel=metric,ylim=(0,1),title=prefix+'Diagonal patch effectiveness')
            ax.legend(title='Patch area')
            save(fig,metric)
        fig = Figure(figsize=(9,7),layout='constrained')
        axes = fig.subplots(3,4)
        fig.suptitle(prefix+'Universal patch patterns')
        for i, area in enumerate(AREAS):
            for j, model in enumerate(MODELS):
                ax=axes[i,j]
                patch = patches.get((model,area))
                if patch is not None:
                    ax.imshow(patch.permute(1,2,0).numpy())
                else:
                    ax.text(.5,.5,'missing',ha='center',va='center')
                ax.set_title(f'{model}\n{area:.0%}')
                ax.axis('off')
        save(fig,'patch_grid')


def report(evaluations, output, *, patch_runs=None, allow_partial=False, plots=True):
    for value in list(evaluations) + list(patch_runs or []):
        check_separate(Path(output)/'reports',value)
    matrices, records = transfer_matrices(evaluations, allow_partial)
    paths = patch_runs or sorted({json.loads((Path(r) / 'config.json').read_text())['source_patch_run'] for r in evaluations})
    patches, patch_hashes = {}, {}
    for path in paths:
        check_separate(Path(output)/'reports',path)
        patch, meta = load_patch(path)
        key = (meta['source_model'],meta['area_fraction'])
        if key in patches:
            raise ValueError('Duplicate source patch in report')
        patches[key] = patch
        patch_hashes[key] = meta['patch_sha256']
    for r in records:
        if patch_hashes.get((r['source'],r['area_fraction'])) != r['patch_sha256']:
            raise ValueError('Report patch tensor does not match evaluation')
    if not allow_partial and len(patches) != 12:
        raise ValueError('Full report requires 12 unique patches')
    run = new_run(output,'reports',{'evaluation_runs':[str(Path(p).resolve()) for p in evaluations],
        'input_manifest_sha256':{str(Path(p).resolve()):file_digest(Path(p)/'run_manifest.json') for p in evaluations},
        'partial':allow_partial, 'synthetic':any(r['synthetic'] for r in records),
        'patch_manifest_sha256':{str(Path(p).resolve()):file_digest(Path(p)/'run_manifest.json') for p in paths},
        'patch_runs':[str(Path(p).resolve()) for p in paths]})
    write_csv(run / 'comparison.csv',records,list(records[0]))
    write_json(run / 'transfer_matrices.json',{'models':list(MODELS),'partial':allow_partial,
        'matrices':{str(a):[[None if np.isnan(v) else float(v) for v in row] for row in matrix] for a,matrix in matrices.items()},
        'note':'null cells are missing or undefined, never zero; inspect comparison.csv for coverage'})
    plot_error = None
    if plots:
        try:
            render_report(run,matrices,records,patches)
        except Exception as error:
            plot_error = f'{type(error).__name__}: {error}'
    write_json(run / 'plot_status.json',{'requested':plots,'error':plot_error})
    finish(run)
    return run
