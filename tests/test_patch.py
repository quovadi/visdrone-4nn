"""Stage 5 tests use synthetic crops/weights only, without real training."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch as mock
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from attacks.adversarial_patch import (AREAS, MODELS, PatchConfig, apply_patch, patch_side,
                                      placements, frozen_classifier, optimize_step)
from attacks.data import selected_validation, dataset
from attacks.preprocessing import PixelModel, rgb_transform
from attacks.patch_data import prepare_subset, read_subset, read_train
from attacks.patch_training import train_patch, load_patch
from attacks.patch_evaluation import evaluate_patch, read_evaluation, metrics_for
from attacks.patch_io import backup_bundle, verify, finish
from attacks.patch_report import transfer_matrices, report
from training.consistency import file_digest


torch.set_num_threads(1)


@pytest.mark.parametrize('area,side',[(.05,50),(.1,71),(.15,87)])
def test_area_and_replacement_mask_gradients(area,side):
    assert patch_side(area) == side
    x = torch.stack([torch.zeros(3,224,224),torch.ones(3,224,224)])
    original = x.clone()
    pattern = nn.Parameter(torch.full((3,side,side),.25))
    coords = [(0,0),(224-side,224-side)]
    out = apply_patch(x,pattern,coords,area)
    for i,(top,left) in enumerate(coords):
        assert torch.equal(out[i,:,top:top+side,left:left+side],pattern)
        mask = torch.zeros_like(x[i],dtype=torch.bool)
        mask[:,top:top+side,left:left+side] = True
        assert torch.equal(out[i][~mask],x[i][~mask])
        assert mask.sum() == 3*side*side
    out.sum().backward()
    assert torch.equal(pattern.grad,torch.full_like(pattern,2.))
    assert torch.equal(x,original) and out.min() >= 0 and out.max() <= 1
    with pytest.raises(ValueError,match='outside'):
        apply_patch(x,pattern,[(-1,0),(0,0)],area)


def test_positions_repeat_across_order_models_and_change_during_optimization():
    ids = ['a','b','c']
    first = placements(ids,.1,42)
    assert first == placements(ids,.1,42)
    assert placements(ids[::-1],.1,42) == first[::-1]
    assert placements(ids,.1,42,step=0) != placements(ids,.1,42,step=1)
    assert all(0<=top<=153 and 0<=left<=153 for top,left in first)


def classifier():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,10))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].weight[0].fill_(1)
        model[-1].weight[1].fill_(-1)
        model[-1].bias.fill_(-10)
        model[-1].bias[:2].zero_()
    return model


def test_optimization_changes_only_patch_and_ascends_ce():
    model = classifier().train()
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    images = torch.full((2,3,224,224),.6)
    original = images.clone()
    labels = torch.tensor([0,0])
    area = .1
    pattern = nn.Parameter(torch.full((3,71,71),.5))
    before = pattern.detach().clone()
    optimizer = torch.optim.Adam([pattern],lr=.05)
    coords = [(0,0),(10,10)]
    with frozen_classifier(model):
        loss0 = torch.nn.functional.cross_entropy(model(apply_patch(images,pattern,coords,area)),labels)
        row = optimize_step(model,images,labels,pattern,optimizer,coords,area)
        loss1 = torch.nn.functional.cross_entropy(model(apply_patch(images,pattern,coords,area)),labels)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert row['updated'] and row['gradient_l1'] > 0 and loss1 > loss0
    assert not torch.equal(before,pattern) and model.training and torch.equal(images,original)
    assert all(p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    assert pattern.min() >= 0 and pattern.max() <= 1


def fixture(tmp_path, count=12):
    root = tmp_path / 'images'
    indices = tmp_path / 'indices'
    indices.mkdir()
    for split,official in [('train','VisDrone2019-DET-train'),('validation','VisDrone2019-DET-val')]:
        records=[]
        for i in range(count):
            name=f'{official}/images/{split}-{i}.png'
            path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(18,16),(160,160,160)).save(path)
            records.append({'image':name,'box':[1,1,15,15],'label':i%3,'line':i+1})
        (indices/f'{split}.json').write_text(json.dumps(records))
    cfg={'input_size':[3,224,224],'mean':[.5]*3,'std':[.5]*3,'crop_pct':1.,'crop_mode':'center','interpolation':'bicubic'}
    model=classifier()
    checkpoint=tmp_path/'synthetic.pt'
    torch.save(model.state_dict(),checkpoint)
    info={'model':'convnext_nano','model_id':'synthetic-mean','variant':'synthetic-smoke',
          'checkpoint':str(checkpoint),'checkpoint_sha256':file_digest(checkpoint),'synthetic':True}
    return root,indices,PixelModel(model,cfg),cfg,info


def train_fixture(tmp_path,plots=False):
    root,indices,model,cfg,info=fixture(tmp_path)
    records,selection=read_train(indices/'train.json',size=6)
    run=train_patch(model,dataset(root,records,rgb_transform(cfg)),selection,info,cfg,
                    PatchConfig(.1,steps=2,batch_size=6,initialization='gray'),tmp_path/'stage5',plots=plots)
    return root,indices,model,cfg,info,run


def test_subset_retains_square_ids_stratifies_and_never_changes_indices(tmp_path):
    root,indices,model,cfg,info=fixture(tmp_path)
    before={p.name:p.read_bytes() for p in indices.iterdir()}
    _,previous=selected_validation(indices/'validation.json',3,3)
    square=tmp_path/'square_subset.json'
    square.write_text(json.dumps(previous))
    file=prepare_subset(indices/'validation.json',tmp_path/'stage5',size=9,seed=9,square_subset=square)
    records,selection=read_subset(indices/'validation.json',file)
    assert len(records)==9 and len({o['sample_id'] for o in selection['objects']})==9
    assert set(o['sample_id'] for o in previous['objects']).issubset({o['sample_id'] for o in selection['objects']})
    assert sum(selection['additional_class_quotas'].values())==6
    assert sum(selection['class_distribution'].values())==9
    again=prepare_subset(indices/'validation.json',tmp_path/'stage5',size=9,seed=9,square_subset=square)
    assert again!=file and json.loads(again.read_text())==json.loads(file.read_text())
    assert before=={p.name:p.read_bytes() for p in indices.iterdir()}
    with pytest.raises(ValueError,match='train.json'):
        read_train(indices/'validation.json')
    # Renaming val as train does not bypass split isolation.
    (indices/'train.json').write_bytes((indices/'validation.json').read_bytes())
    with pytest.raises(ValueError,match='official DET train'):
        read_train(indices/'train.json')


def test_patch_roundtrip_evaluation_backup_and_report(tmp_path):
    root,indices,model,cfg,info,trained=train_fixture(tmp_path,plots=True)
    pattern,meta=load_patch(trained)
    assert pattern.shape==(3,71,71) and meta['optimizer_updates']==2
    assert (trained/'training_loss.png').read_bytes().startswith(b'\x89PNG')
    subset=prepare_subset(indices/'validation.json',tmp_path/'stage5',size=6)
    records,selection=read_subset(indices/'validation.json',subset)
    ds=dataset(root,records,rgb_transform(cfg))
    before={p:p.read_bytes() for p in trained.rglob('*') if p.is_file()}
    run=evaluate_patch(model,ds,selection,info,cfg,trained,tmp_path/'stage5',batch_size=2)
    config,rows,metrics=read_evaluation(run)
    assert len(rows)==6 and metrics['clean_correct_count']==2
    assert metrics['support']==[2,2,2]+[0]*7
    second=evaluate_patch(model,ds,selection,info,cfg,trained,tmp_path/'stage5',batch_size=3)
    _,other,_=read_evaluation(second)
    assert [(r['top'],r['left'],r['adversarial_prediction']) for r in rows]==[(r['top'],r['left'],r['adversarial_prediction']) for r in other]
    assert before=={p:p.read_bytes() for p in trained.rglob('*') if p.is_file()}
    backup=backup_bundle(trained,tmp_path/'drive')
    assert torch.equal(load_patch(backup)[0],pattern)
    with pytest.raises(FileExistsError):
        backup_bundle(trained,tmp_path/'drive')
    output=report([run],tmp_path/'stage5',allow_partial=True)
    assert (output/'transfer_0.10.pdf').read_bytes().startswith(b'%PDF')
    assert (output/'patch_grid.png').read_bytes().startswith(b'\x89PNG')
    with pytest.raises(ValueError,match='1024'):
        report([run],tmp_path/'stage5')
    (trained/'patch.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):
        load_patch(trained)


def test_metrics_use_target_clean_correct_and_zero_denominators():
    rows=[{'true_class':0,'clean_prediction':0,'adversarial_prediction':1},
          {'true_class':1,'clean_prediction':1,'adversarial_prediction':1},
          {'true_class':1,'clean_prediction':2,'adversarial_prediction':1}]
    metrics,per=metrics_for(rows)
    assert metrics['asr']==.5 and metrics['adversarial_accuracy']==2/3
    assert per[0]['asr']==1 and per[1]['asr']==0 and per[2]['asr'] is None
    assert per[1]['adversarial_accuracy']==1
    other,_=metrics_for([{'true_class':0,'clean_prediction':1,'adversarial_prediction':0}])
    assert other['asr'] is None


def test_transfer_matrix_orientation_full_coverage_and_conflicts():
    examples={}
    for i,source in enumerate(MODELS):
        for j,target in enumerate(MODELS):
            for area in AREAS:
                config={'source':{'source_model':source,'area_fraction':area,'source_checkpoint_sha256':source,'source_variant':'baseline-v1'},
                        'target':{'model':target,'checkpoint_sha256':target,'variant':'baseline-v1'},'subset_sha256':'same',
                        'placement_seed':42,'data_config':{'input_size':[3,224,224]},'patch_sha256':source+str(area)}
                examples[f'{source}-{target}-{area}']=(config,[{'sample_id':str(k),'clean_prediction':0} for k in range(1024)],
                    {'asr':(i+j)/10,'clean_accuracy':1.,'adversarial_accuracy':.5,'adversarial_macro_f1':.05})
    with mock('attacks.patch_report.read_evaluation',side_effect=lambda key:examples[key]):
        matrices,records=transfer_matrices(list(examples))
        assert len(records)==48 and matrices[.1][0,3]==.3 and matrices[.15][2,1]==.3
        with pytest.raises(ValueError,match='duplicate'):
            transfer_matrices([next(iter(examples))]*2,True)
        with pytest.raises(ValueError,match='48'):
            transfer_matrices(list(examples)[:-1])
        key=next(iter(examples))
        examples[key][0]['subset_sha256']='foreign'
        with pytest.raises(ValueError,match='different subsets'):
            transfer_matrices(list(examples))


def test_transfer_evaluation_does_not_train_target_or_patch(tmp_path):
    root,indices,source,cfg,source_info,trained=train_fixture(tmp_path)
    frozen=prepare_subset(indices/'validation.json',tmp_path/'stage5',size=6)
    records,selection=read_subset(indices/'validation.json',frozen)
    ds=dataset(root,records,rgb_transform(cfg))
    target=classifier()
    with torch.no_grad():
        target[-1].weight.zero_()
        target[-1].bias.zero_()
        target[-1].bias[1]=10
    target_file=tmp_path/'synthetic-target.pt'
    torch.save(target.state_dict(),target_file)
    info={**source_info,'model':'tinyvit','checkpoint':str(target_file),'checkpoint_sha256':file_digest(target_file)}
    original={k:v.clone() for k,v in target.state_dict().items()}
    patch_bytes=(trained/'patch.pt').read_bytes()
    with mock('torch.optim.Adam',side_effect=AssertionError('No optimization in transfer')), \
         mock('torch.optim.SGD',side_effect=AssertionError('No optimization in transfer')):
        transfer=evaluate_patch(PixelModel(target,cfg),ds,selection,info,cfg,trained,tmp_path/'stage5')
        diagonal=evaluate_patch(source,ds,selection,source_info,cfg,trained,tmp_path/'stage5')
    _,target_rows,metrics=read_evaluation(transfer)
    _,source_rows,_=read_evaluation(diagonal)
    assert metrics['clean_correct_count']==2
    assert [r['clean_correct'] for r in target_rows]!=[r['clean_correct'] for r in source_rows]
    assert [(r['top'],r['left']) for r in target_rows]==[(r['top'],r['left']) for r in source_rows]
    assert (trained/'patch.pt').read_bytes()==patch_bytes
    assert all(torch.equal(original[k],v) for k,v in target.state_dict().items())
    matrices,_=transfer_matrices([diagonal,transfer],True)
    assert np.isfinite(matrices[.1][0,1])


def test_zero_clean_correct_skips_patch_update():
    model=classifier()
    x=torch.ones(1,3,224,224)*.7
    p=nn.Parameter(torch.ones(3,50,50)*.5)
    before=p.detach().clone()
    optimizer=torch.optim.Adam([p],lr=1.)
    with frozen_classifier(model):
        row=optimize_step(model,x,torch.tensor([9]),p,optimizer,[(0,0)],.05)
    assert row['loss'] is None and not row['updated'] and torch.equal(before,p) and not optimizer.state


@pytest.mark.parametrize('model_id', ['convnext_nano.d1h_in1k','tiny_vit_5m_224.dist_in22k_ft_in1k',
                                     'convnext_base.fb_in1k','vit_base_patch16_224.augreg_in1k'])
def test_four_real_architecture_graphs_without_pretrained_downloads(model_id):
    from training.models import build_model
    model,cfg=build_model(model_id,pretrained=False)
    pixel=PixelModel(model,cfg)
    images=torch.full((1,3,224,224),.5)
    pattern=nn.Parameter(torch.full((3,50,50),.4))
    first=next(model.parameters()).detach().clone()
    with frozen_classifier(pixel):
        logits=pixel(apply_patch(images,pattern,[(12,13)],.05))
        gradient=torch.autograd.grad(logits.sum(),pattern)[0]
        assert logits.shape==(1,10) and torch.isfinite(logits).all() and torch.isfinite(gradient).all()
        assert gradient.shape==pattern.shape and all(p.grad is None for p in model.parameters())
    assert torch.equal(first,next(model.parameters()))



def test_resize_pattern_and_extreme_update_clamping():
    images=torch.zeros(1,3,16,16)
    pattern=nn.Parameter(torch.full((3,71,71),.25))
    out=apply_patch(images,pattern,[(11,11)],.1)
    assert torch.allclose(out[0,:,11:16,11:16],torch.full((3,5,5),.25))
    assert int((out!=0).sum())==3*25
    model=classifier()
    p=nn.Parameter(torch.full((3,71,71),.5))
    optimizer=torch.optim.Adam([p],lr=10.)
    with frozen_classifier(model):
        optimize_step(model,torch.full((1,3,224,224),.6),torch.tensor([0]),p,optimizer,[(0,0)],.1)
    assert p.min()>=0 and p.max()<=1 and (p==0).any()


def test_validation_cannot_enter_optimizer_and_backup_failure_is_detected(tmp_path):
    root,indices,model,cfg,info=fixture(tmp_path)
    subset=prepare_subset(indices/'validation.json',tmp_path/'stage5',size=6)
    records,selection=read_subset(indices/'validation.json',subset)
    with pytest.raises(ValueError,match='Validation cannot'):
        train_patch(model,dataset(root,records,rgb_transform(cfg)),selection,info,cfg,
                    PatchConfig(.05,steps=1),tmp_path/'stage5',plots=False)
    train_records,selection=read_train(indices/'train.json',6)
    run=train_patch(model,dataset(root,train_records,rgb_transform(cfg)),selection,info,cfg,
                    PatchConfig(.05,steps=1,batch_size=6),tmp_path/'stage5',plots=False)
    original={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    def corrupt_copy(source,target):
        Path(target).write_bytes(b'broken copy')
    with mock('attacks.patch_io.shutil.copyfile',side_effect=corrupt_copy):
        with pytest.raises(OSError,match='checksum'):
            backup_bundle(run,tmp_path/'bad-drive')
    assert original=={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='separate'):
        backup_bundle(run,run/'backup')


@pytest.mark.parametrize('damage', ['edited', 'missing', 'incomplete', 'foreign', 'outside', 'moved_bundle', 'renamed_bundle'])
def test_frozen_subset_requires_published_manifest(tmp_path, damage):
    from attacks.data import canonical_hash
    root, indices, model, cfg, info = fixture(tmp_path)
    subset = prepare_subset(indices/'validation.json', tmp_path/'stage5', size=6)
    assert len(read_subset(indices/'validation.json', subset)[0]) == 6
    if damage == 'edited':
        saved = json.loads(subset.read_text())
        saved['objects'].reverse()
        saved['subset_sha256'] = canonical_hash(saved['objects'])
        subset.write_text(json.dumps(saved))
    elif damage == 'missing':
        (subset.parent/'run_manifest.json').unlink()
    elif damage == 'incomplete':
        (subset.parent/'status.json').write_text('{"state":"failed"}')
    elif damage == 'foreign':
        config = json.loads((subset.parent/'config.json').read_text())
        config['kind'] = 'patches'
        (subset.parent/'config.json').write_text(json.dumps(config))
        finish(subset.parent)
    elif damage == 'outside':
        outside = tmp_path/'subset.json'
        outside.write_bytes(subset.read_bytes())
        subset = outside
    elif damage == 'moved_bundle':
        import shutil
        moved = tmp_path/'moved_bundle'
        shutil.move(str(subset.parent), moved)
        subset = moved/'subset.json'
    elif damage == 'renamed_bundle':
        renamed = subset.parent.parent/'renamed-bundle'
        subset.parent.rename(renamed)
        subset = renamed/'subset.json'
    with pytest.raises((ValueError, FileNotFoundError)):
        read_subset(indices/'validation.json', subset)


def matrix_patch_fixtures(tmp_path):
    """Published synthetic bundles, not twelve real optimization experiments."""
    import shutil
    import uuid
    from attacks.adversarial_patch import patch_side
    _, _, _, _, _, template = train_fixture(tmp_path)
    runs = []
    for model in MODELS:
        for area in AREAS:
            run = template.parent/uuid.uuid4().hex
            shutil.copytree(template, run)
            config = json.loads((run/'config.json').read_text())
            meta = json.loads((run/'metadata.json').read_text())
            config['run_id'] = run.name
            config['source']['model'] = model
            config['source']['checkpoint_sha256'] = model+'-synthetic-sha'
            config['optimization']['area_fraction'] = area
            meta.update(source_model=model, source_checkpoint_sha256=model+'-synthetic-sha',
                        area_fraction=area, patch_side=patch_side(area),
                        optimization=config['optimization'])
            torch.save(torch.full((3,patch_side(area),patch_side(area)), .5), run/'patch.pt')
            meta['patch_sha256'] = file_digest(run/'patch.pt')
            (run/'config.json').write_text(json.dumps(config))
            (run/'metadata.json').write_text(json.dumps(meta))
            finish(run)
            runs.append(run)
    return runs


def test_complete_patch_protocol_and_partial_are_accepted(tmp_path):
    from attacks.patch_report import validate_patch_protocol
    runs = matrix_patch_fixtures(tmp_path)
    patches, metadata = validate_patch_protocol(runs)
    assert set(patches) == {(m,a) for m in MODELS for a in AREAS}
    assert len({m['source_checkpoint_sha256'] for m in metadata.values()}) == 4
    assert len(validate_patch_protocol(runs[:2], True)[0]) == 2
    with pytest.raises(ValueError, match='12'):
        validate_patch_protocol(runs[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        validate_patch_protocol(runs+[runs[0]])


@pytest.mark.parametrize('field', ['index', 'subset', 'objects', 'selection_seed', 'size',
    'lr', 'steps', 'batch_size', 'optimizer', 'initialization', 'seed',
    'objective', 'precision', 'compositing', 'native_resolution', 'rotation', 'random_scaling',
    'config_binding', 'metadata_binding', 'manifest'])
def test_patch_protocol_rejects_mixed_or_internally_invalid_bundles(tmp_path, field):
    from attacks.patch_report import validate_patch_protocol
    from attacks.data import canonical_hash
    runs = matrix_patch_fixtures(tmp_path)
    run = runs[-1]
    config = json.loads((run/'config.json').read_text())
    meta = json.loads((run/'metadata.json').read_text())
    selection = json.loads((run/'train_subset.json').read_text())
    if field == 'index':
        selection['index_sha256'] = meta['train_index_sha256'] = 'foreign-index'
    elif field in ('subset', 'objects'):
        if field == 'subset':
            selection['objects'].reverse()
        else:
            selection['objects'][0]['sample_id'] = 'foreign-object'
        sha = canonical_hash(selection['objects'])
        selection['subset_sha256'] = config['train_selection_sha256'] = meta['train_subset_sha256'] = sha
    elif field == 'selection_seed':
        selection['seed'] += 1
    elif field == 'size':
        selection['size'] += 1
    elif field == 'config_binding':
        config['train_selection_sha256'] = 'foreign'
    elif field == 'metadata_binding':
        meta['train_subset_sha256'] = 'foreign'
    elif field != 'manifest':
        alternatives = {'lr': .123, 'steps': 99, 'batch_size': 99, 'optimizer': 'sgd',
            'initialization':'uniform', 'seed': 99, 'objective':'other', 'precision':'fp16',
            'compositing':'blend', 'native_resolution':[32,32], 'rotation':True, 'random_scaling':True}
        config['optimization'][field] = alternatives[field]
        meta['optimization'] = config['optimization']
        if field == 'seed':
            meta['seed'] = alternatives[field]
    (run/'config.json').write_text(json.dumps(config))
    (run/'metadata.json').write_text(json.dumps(meta))
    (run/'train_subset.json').write_text(json.dumps(selection))
    if field != 'manifest':
        finish(run)  # Exercise semantic consistency even with refreshed local digests.
    else:
        (run/'train_subset.json').write_text('{}')
    for partial in (False, True):
        with pytest.raises(ValueError):
            validate_patch_protocol(runs, partial)
