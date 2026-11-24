"""Evaluate a fixed source patch on a target model; never optimize on validation."""
import csv
import json
from pathlib import Path
import time
import torch
from attacks.adversarial_patch import apply_patch, placements, patch_side, frozen_classifier
from attacks.metrics import PairedMetrics
from attacks.data import canonical_hash
from attacks.patch_io import new_run, finish, verify, write_csv, now
from attacks.patch_training import load_patch
from data.visdrone import CLASSES
from training.checkpoint import write_json
from training.consistency import file_digest
from training.storage import check_separate

FIELDS = ('run_id','sample_id','index_row','image','box','line','true_class','clean_prediction',
          'adversarial_prediction','clean_correct','attack_success','source_model','target_model',
          'source_checkpoint_sha256','target_checkpoint_sha256','patch_sha256','area_fraction',
          'top','left','patch_side','placement_seed')
CLASS_FIELDS = ('label','class_name','support','clean_correct','attack_success_count','asr',
                'adversarial_accuracy','clean_f1','adversarial_f1')


def metrics_for(rows):
    paired = PairedMetrics()
    per_class = []
    for row in rows:
        paired.update([int(row['true_class'])], [int(row['clean_prediction'])], [int(row['adversarial_prediction'])])
    metrics = paired.compute()
    for label, name in enumerate(CLASSES):
        group = [r for r in rows if int(r['true_class']) == label]
        clean = sum(int(r['clean_prediction']) == label for r in group)
        succeeded = sum(int(r['clean_prediction']) == label and int(r['adversarial_prediction']) != label for r in group)
        correct = sum(int(r['adversarial_prediction']) == label for r in group)
        per_class.append({'label': label, 'class_name': name, 'support': len(group), 'clean_correct': clean,
            'attack_success_count': succeeded, 'asr': succeeded/clean if clean else None,
            'adversarial_accuracy': correct/len(group) if group else None,
            'clean_f1': metrics['clean_per_class_f1'][label], 'adversarial_f1': metrics['adversarial_per_class_f1'][label]})
    return metrics, per_class


def evaluate_patch(model, data, selection, target_info, data_config, patch_run, output, *,
                   batch_size=8, placement_seed=42, subset_file_sha256=None):
    for value in [patch_run,target_info['checkpoint']] + ([data.root] if hasattr(data,'root') else []):
        check_separate(Path(output)/'evaluations',value)
    if canonical_hash(selection['objects']) != selection['subset_sha256'] or len({o['sample_id'] for o in selection['objects']}) != len(selection['objects']):
        raise ValueError('Invalid frozen selection identity/hash')
    if len(selection['objects']) != selection['size'] or len(data) != selection['size'] or not len(data) or batch_size < 1:
        raise ValueError('Invalid frozen evaluation dataset/batch size')
    if any(not o['image'].startswith('VisDrone2019-DET-val/images/') for o in selection['objects']):
        raise ValueError('Patch evaluation uses official validation only')
    patch, source = load_patch(patch_run)
    device = next(model.parameters()).device
    patch = patch.to(device)
    area = source['area_fraction']
    run = new_run(output, 'evaluations', {'source_patch_run': str(Path(patch_run).resolve()),
        'patch_sha256': source['patch_sha256'], 'source': source, 'target': target_info,
        'data_config': data_config, 'subset_sha256': selection['subset_sha256'],
        'subset_file_sha256': subset_file_sha256, 'placement_seed': placement_seed, 'batch_size': batch_size})
    write_json(run / 'subset.json', selection)
    rows = []
    started = time.monotonic()
    try:
        with frozen_classifier(model), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for offset in range(0, len(data), batch_size):
                identities = selection['objects'][offset:offset+batch_size]
                batch = [data[i] for i in range(offset, min(offset+batch_size,len(data)))]
                images = torch.stack([x for x,y in batch]).to(device, dtype=torch.float32)
                if images.shape[1:] != (3,224,224):
                    raise ValueError('Evaluation requires 224x224 object crops')
                coords = placements([o['sample_id'] for o in identities], area, placement_seed)
                clean_logits = model(images)
                adv_logits = model(apply_patch(images, patch, coords, area))
                if (clean_logits.shape != (len(batch),10) or adv_logits.shape != clean_logits.shape
                        or not torch.isfinite(clean_logits).all() or not torch.isfinite(adv_logits).all()):
                    raise ValueError('Nonfinite/invalid target logits')
                clean, adv = clean_logits.argmax(1).tolist(), adv_logits.argmax(1).tolist()
                for i, obj in enumerate(identities):
                    truth = batch[i][1]
                    if truth != obj['label']:
                        raise ValueError('Dataset label differs from frozen subset')
                    correct = clean[i] == truth
                    rows.append({'run_id': run.name, 'sample_id': obj['sample_id'], 'index_row': obj['index_row'],
                        'image': obj['image'], 'box': json.dumps(obj['box']), 'line': obj['line'], 'true_class': truth,
                        'clean_prediction': clean[i], 'adversarial_prediction': adv[i], 'clean_correct': correct,
                        'attack_success': adv[i] != truth if correct else None, 'source_model': source['source_model'],
                        'target_model': target_info['model'], 'source_checkpoint_sha256': source['source_checkpoint_sha256'],
                        'target_checkpoint_sha256': target_info['checkpoint_sha256'], 'patch_sha256': source['patch_sha256'],
                        'area_fraction': area, 'top': coords[i][0], 'left': coords[i][1], 'patch_side': patch_side(area),
                        'placement_seed': placement_seed})
                write_csv(run / 'predictions.csv', rows, FIELDS)  # Atomic committed prefix.
        if 'validation_index' in target_info and file_digest(Path(target_info['validation_index'])) != selection['index_sha256']:
            raise ValueError('Frozen validation index changed during evaluation')
        if 'stage5_subset' in target_info and file_digest(Path(target_info['stage5_subset'])) != subset_file_sha256:
            raise ValueError('Frozen Stage 5 subset changed during evaluation')
        if file_digest(Path(patch_run)/'patch.pt') != source['patch_sha256']:
            raise ValueError('Source patch changed during evaluation')
        if file_digest(Path(target_info['checkpoint'])) != target_info['checkpoint_sha256']:
            raise ValueError('Target checkpoint changed')
        metrics, per_class = metrics_for(rows)
        metrics['runtime_seconds'] = time.monotonic()-started
        write_json(run / 'metrics.json', metrics)
        write_csv(run / 'per_class_metrics.csv', per_class, CLASS_FIELDS)
        write_json(run / 'metadata.json', {'source_model': source['source_model'], 'target_model': target_info['model'],
            'source_checkpoint_sha256': source['source_checkpoint_sha256'], 'target_checkpoint_sha256': target_info['checkpoint_sha256'],
            'patch_sha256': source['patch_sha256'], 'area_fraction': area, 'subset_sha256': selection['subset_sha256'],
            'validation_index_sha256': selection['index_sha256'], 'placement_seed': placement_seed})
        finish(run)
    except BaseException as error:
        write_json(run / 'status.json', {'state':'failed', 'finished_utc':now(), 'error':str(error)})
        raise
    return run


def read_evaluation(run):
    run = Path(run)
    config = verify(run, 'evaluations')
    subset = json.loads((run / 'subset.json').read_text())
    from attacks.data import canonical_hash
    if subset['subset_sha256'] != config['subset_sha256'] or canonical_hash(subset['objects']) != subset['subset_sha256']:
        raise ValueError('Evaluation subset mismatch')
    with (run / 'predictions.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('Invalid Stage 5 CSV schema')
        rows = list(reader)
    if len(rows) != subset['size'] or len(rows) != len(subset['objects']):
        raise ValueError('Incomplete Stage 5 evaluation')
    for row, obj in zip(rows, subset['objects']):
        for key in ('index_row','line','true_class','clean_prediction','adversarial_prediction','top','left','patch_side','placement_seed'):
            row[key] = int(row[key])
        area = float(row['area_fraction'])
        expected = placements([obj['sample_id']], area, config['placement_seed'])[0]
        if (row['sample_id'] != obj['sample_id'] or row['image'] != obj['image'] or json.loads(row['box']) != obj['box']
                or row['index_row'] != obj['index_row'] or row['line'] != obj['line'] or row['true_class'] != obj['label']
                or (row['top'],row['left']) != expected or row['patch_side'] != patch_side(area)
                or row['patch_sha256'] != config['patch_sha256'] or row['run_id'] != config['run_id']
                or row['target_model'] != config['target']['model'] or row['source_model'] != config['source']['source_model']
                or row['target_checkpoint_sha256'] != config['target']['checkpoint_sha256']
                or row['source_checkpoint_sha256'] != config['source']['source_checkpoint_sha256']
                or area != config['source']['area_fraction'] or row['placement_seed'] != config['placement_seed']):
            raise ValueError('Stage 5 prediction identity/placement mismatch')
        correct = row['clean_prediction'] == row['true_class']
        success = row['adversarial_prediction'] != row['true_class'] if correct else None
        if row['clean_correct'] != str(correct) or row['attack_success'] != (str(success) if success is not None else ''):
            raise ValueError('Stage 5 correctness flags mismatch')
        row['clean_correct']=correct
        row['attack_success']=success
        row['area_fraction']=area
    metrics, per_class = metrics_for(rows)
    saved = json.loads((run / 'metrics.json').read_text())
    if any(saved[k] != v for k,v in metrics.items()):
        raise ValueError('Stage 5 metrics disagree with predictions')
    with (run/'per_class_metrics.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames != list(CLASS_FIELDS):
            raise ValueError('Invalid per-class metrics schema')
        saved_classes=list(reader)
    expected_classes=[{k: '' if v is None else str(v) for k,v in r.items()} for r in per_class]
    if saved_classes != expected_classes:
        raise ValueError('Per-class metrics disagree with predictions')
    return config, rows, saved
