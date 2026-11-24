"""Explicit Stage 5 jobs: prepare-subset, optimize, evaluate, transfer, report."""
import argparse
import json
from pathlib import Path
import torch
from attacks.adversarial_patch import PatchConfig, AREAS, MODELS
from attacks.data import dataset
from attacks.models import load_classifier
from attacks.preprocessing import PixelModel, rgb_transform
from attacks.patch_data import read_train, prepare_subset, read_subset
from attacks.patch_training import train_patch
from attacks.patch_evaluation import evaluate_patch
from attacks.patch_report import report
from attacks.patch_io import backup_bundle
from training.run import seed_everything
from training.consistency import file_digest
from training.storage import check_separate


def classifier(model_run, checkpoint, device, checkpoint_format='compact', trusted_full=False):
    model, cfg, info = load_classifier(model_run,checkpoint,'baseline-v1',checkpoint_format,trusted_full)
    return PixelModel(model,cfg).to(device), cfg, info


def protect(output, kind, inputs):
    for value in inputs:
        check_separate(Path(output)/kind, value)


def frozen_index_matches(info, split, index):
    expected = info['training_metadata']['data']['index_sha256'][split]
    if file_digest(Path(index)) != expected:
        raise ValueError(f'{split} index differs from classifier frozen training metadata')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    jobs = parser.add_subparsers(dest='job',required=True)
    subset = jobs.add_parser('prepare-subset')
    subset.add_argument('--validation-index',type=Path,required=True)
    subset.add_argument('--square-subset',type=Path)
    subset.add_argument('--size',type=int,default=1024)
    subset.add_argument('--seed',type=int,default=42)
    subset.add_argument('--output',type=Path,required=True)
    for name in ('optimize','evaluate','transfer'):
        job = jobs.add_parser(name)
        job.add_argument('--root',type=Path,required=True)
        job.add_argument('--output',type=Path,required=True)
        job.add_argument('--backup-root',type=Path)
        job.add_argument('--batch-size',type=int,default=8)
        job.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        if name != 'transfer':
            job.add_argument('--model-run',type=Path,required=True)
            job.add_argument('--checkpoint',type=Path,required=True)
            job.add_argument('--checkpoint-format',choices=('compact','full'),default='compact')
            job.add_argument('--trusted-full-checkpoint',action='store_true')
        if name == 'optimize':
            job.add_argument('--train-index',type=Path,required=True)
            job.add_argument('--train-size',type=int,default=4096)
            job.add_argument('--area',type=float,choices=AREAS,required=True)
            job.add_argument('--steps',type=int,default=500)
            job.add_argument('--lr',type=float,default=.05)
            job.add_argument('--optimizer',choices=('adam','sgd'),default='adam')
            job.add_argument('--seed',type=int,default=42)
            job.add_argument('--initialization',choices=('uniform','gray'),default='uniform')
        else:
            job.add_argument('--validation-index',type=Path,required=True)
            job.add_argument('--subset',type=Path,required=True)
            job.add_argument('--placement-seed',type=int,default=42)
            if name == 'evaluate':
                job.add_argument('--patch',type=Path,required=True)
            else:
                job.add_argument('--patches',type=Path,nargs='+',required=True)
                job.add_argument('--targets',type=Path,required=True,help='JSON model alias -> model_run/checkpoint options')
    summary = jobs.add_parser('report')
    summary.add_argument('--evaluations',type=Path,nargs='+',required=True)
    summary.add_argument('--patches',type=Path,nargs='+')
    summary.add_argument('--output',type=Path,required=True)
    summary.add_argument('--allow-partial',action='store_true')
    args = parser.parse_args()
    if args.job == 'prepare-subset':
        protect(args.output,'subsets',[args.validation_index.parent]+([args.square_subset] if args.square_subset else []))
        print(prepare_subset(args.validation_index,args.output,args.size,args.seed,args.square_subset))
        return
    if args.job == 'report':
        protect(args.output,'reports',args.evaluations+(args.patches or []))
        print(report(args.evaluations,args.output,patch_runs=args.patches,allow_partial=args.allow_partial))
        return
    kind = 'patches' if args.job == 'optimize' else 'evaluations'
    inputs = [args.root]
    if args.job == 'optimize':
        inputs += [args.train_index.parent,args.model_run,args.checkpoint]
    else:
        inputs += [args.validation_index.parent,args.subset]
        inputs += [args.patch,args.model_run,args.checkpoint] if args.job == 'evaluate' else args.patches+[args.targets]
    protect(args.output,kind,inputs)
    if args.backup_root:
        check_separate(args.output,args.backup_root)
        protect(args.backup_root,kind,inputs)
    seed_everything(args.seed if args.job == 'optimize' else args.placement_seed)
    if args.job == 'optimize':
        model,cfg,info = classifier(args.model_run,args.checkpoint,args.device,args.checkpoint_format,args.trusted_full_checkpoint)
        frozen_index_matches(info,'train',args.train_index)
        records,selection = read_train(args.train_index,args.train_size,args.seed)
        info['optimization_train_index'] = str(args.train_index.resolve())
        config = PatchConfig(args.area,args.steps,args.batch_size,args.lr,args.optimizer,args.seed,args.initialization)
        run = train_patch(model,dataset(args.root,records,rgb_transform(cfg)),selection,info,cfg,config,args.output)
        if args.backup_root:
            print('BACKUP=',backup_bundle(run,args.backup_root))
        print('PATCH_RUN=',run)
        return
    records,selection = read_subset(args.validation_index,args.subset)
    subset_sha = file_digest(args.subset)
    if args.job == 'evaluate':
        targets = {'selected':{'model_run':str(args.model_run),'checkpoint':str(args.checkpoint),
            'checkpoint_format':args.checkpoint_format,'trusted_full':args.trusted_full_checkpoint}}
        patches = [args.patch]
    else:
        targets = json.loads(args.targets.read_text())
        if not targets or any(name not in MODELS for name in targets):
            raise ValueError('Targets must name the four baseline aliases')
        patches = args.patches
        for target in targets.values():
            protect(args.output,kind,[target['model_run'],target['checkpoint']])
            if args.backup_root:
                protect(args.backup_root,kind,[target['model_run'],target['checkpoint']])
    # Sequential target models: patch tensors only are reused across targets.
    for name,target in targets.items():
        model,cfg,info = classifier(target['model_run'],target['checkpoint'],args.device,
            target.get('checkpoint_format','compact'),target.get('trusted_full',False))
        if args.job == 'transfer' and info['model'] != name:
            raise ValueError('Target registry alias differs from checkpoint metadata')
        frozen_index_matches(info,'validation',args.validation_index)
        info.update(validation_index=str(args.validation_index.resolve()), stage5_subset=str(args.subset.resolve()))
        data = dataset(args.root,records,rgb_transform(cfg))
        for patch in patches:
            run = evaluate_patch(model,data,selection,info,cfg,patch,args.output,batch_size=args.batch_size,
                placement_seed=args.placement_seed,subset_file_sha256=subset_sha)
            if args.backup_root:
                print('BACKUP=',backup_bundle(run,args.backup_root))
            print('EVALUATION_RUN=',run)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
