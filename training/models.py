"""Registry checks and pretrained preprocessing, with a single normalization."""
import argparse
import json
import timm
from timm.data import create_transform, resolve_model_data_config
from timm.models import get_pretrained_cfg
from torchvision.transforms import Compose, RandomHorizontalFlip

from training.config import PRESETS


def check_model(model_id):
    # Full weight tag is mandatory; is_model(base) alone does not verify weights.
    if model_id not in timm.list_models(pretrained=True, include_tags=True):
        raise ValueError(f'Unavailable pretrained timm model: {model_id}. No substitution performed.')
    cfg = get_pretrained_cfg(model_id)
    if cfg is None or not (cfg.hf_hub_id or cfg.url):
        raise ValueError(f'No pretrained weight source for {model_id}')
    return cfg.to_dict()


def build_model(model_id, pretrained=True):
    check_model(model_id)
    # On resume weights are loaded from our checkpoint, avoiding a new download.
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=10)
    data_config = resolve_model_data_config(model)
    if tuple(data_config['input_size']) != (3, 224, 224):
        raise ValueError(f'Unexpected input size for fixed 224 protocol: {data_config}')
    return model, data_config


def make_transforms(data_config):
    # Deliberately mild identical augmentation. No random resized crops that
    # could remove small target objects. Keep pretrained evaluation geometry.
    evaluation = create_transform(**data_config, is_training=False)
    return Compose([RandomHorizontalFlip(p=0.5), evaluation]), evaluation


def main():
    parser = argparse.ArgumentParser(description='Check registry; optionally download/instantiate ONE model')
    parser.add_argument('--model', choices=list(json.loads(PRESETS.read_text())['models']))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    presets = json.loads(PRESETS.read_text())['models']
    if args.download and args.model is None:
        parser.error('--download requires --model (one model at a time)')
    for name, model_id in presets.items():
        if args.model and name != args.model:
            continue
        info = check_model(model_id)
        if args.download:
            model, data_config = build_model(model_id)
            info['resolved_data_config'] = data_config
            info['parameters'] = sum(p.numel() for p in model.parameters())
            del model
        print(json.dumps({'model': name, 'model_id': model_id, 'pretrained': info}, indent=2))


if __name__ == '__main__':
    main()
