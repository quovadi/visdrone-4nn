"""One synthetic patch, two targets, partial report. Never the 12/48 real grid."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import torch
from torch import nn
from attacks.data import dataset, selected_validation
from attacks.preprocessing import PixelModel, rgb_transform
from attacks.adversarial_patch import PatchConfig
from attacks.patch_data import read_train, prepare_subset, read_subset
from attacks.patch_training import train_patch
from attacks.patch_evaluation import evaluate_patch, read_evaluation
from attacks.patch_report import report
from training.consistency import file_digest


def toy():
    model=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,10))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].weight[0].fill_(1)
        model[-1].weight[1].fill_(-1)
        model[-1].bias.fill_(-10)
        model[-1].bias[:2].zero_()
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    with TemporaryDirectory(prefix='patch-smoke-') as tmp:
        root=Path(tmp)
        for split,official in [('train','VisDrone2019-DET-train'),('validation','VisDrone2019-DET-val')]:
            rows=[]
            for i in range(12):
                name=f'{official}/images/{split}-{i}.png'
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(16,16),(160,)*3).save(path)
                rows.append({'image':name,'box':[1,1,15,15],'label':i%3,'line':i+1})
            (root/f'{split}.json').write_text(json.dumps(rows))
        cfg={'input_size':[3,224,224],'mean':[.5]*3,'std':[.5]*3,'crop_pct':1.,'crop_mode':'center','interpolation':'bicubic'}
        source=toy()
        checkpoint=root/'synthetic.pt'
        torch.save(source.state_dict(),checkpoint)
        info={'model':'convnext_nano','model_id':'fixed-toy','variant':'synthetic-smoke','synthetic':True,
              'checkpoint':str(checkpoint),'checkpoint_sha256':file_digest(checkpoint)}
        records,selection=read_train(root/'train.json',size=6)
        patch_run=train_patch(PixelModel(source,cfg),dataset(root,records,rgb_transform(cfg)),selection,info,cfg,
                              PatchConfig(.05,steps=3,batch_size=6,initialization='gray'),args.output)
        _,pilot=selected_validation(root/'validation.json',42,3)
        (root/'square_subset.json').write_text(json.dumps(pilot))
        frozen=prepare_subset(root/'validation.json',args.output,size=6,square_subset=root/'square_subset.json')
        records,selection=read_subset(root/'validation.json',frozen)
        ds=dataset(root,records,rgb_transform(cfg))
        evaluations=[]
        for name in ('convnext_nano','tinyvit'):
            target=toy()
            run=evaluate_patch(PixelModel(target,cfg),ds,selection,{**info,'model':name},cfg,patch_run,args.output,batch_size=2)
            read_evaluation(run)
            evaluations.append(run)
        summary=report(evaluations,args.output,allow_partial=True)
        print(json.dumps({'synthetic_only':True,'patch_run':str(patch_run),'subset_file':str(frozen),
             'evaluations':[str(r) for r in evaluations],'report':str(summary),
             'optimizations':1,'transfer_evaluations':2,'real_models_trained':0},indent=2))


if __name__=='__main__':
    main()
