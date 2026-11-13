"""Strict loading of already-trained weights; no pretrained downloads."""
import json
from pathlib import Path
from training.checkpoint import load_weights, read_checkpoint
from training.models import build_model
from training.config import PRESETS
from training.consistency import file_digest


def json_value(value):
    return json.loads(json.dumps(value))


def load_classifier(model_run, checkpoint, variant, checkpoint_format='compact', trusted_full=False):
    model_run, checkpoint = Path(model_run), Path(checkpoint)
    metadata = {name: json.loads((model_run / f'{name}.json').read_text())
                for name in ('config', 'preprocessing', 'data', 'provenance')}
    config = metadata['config']
    special = (config['model'] == 'convnext_base' and config.get('train_size') == 50000
               and config.get('lr') == 3e-5)
    if variant not in ('baseline-v1', 'convnext-base-v2') or special != (variant == 'convnext-base-v2'):
        raise ValueError('ConvNeXt-Base 50k/lr=3e-5 must be labelled convnext-base-v2, separate from baseline-v1')
    presets = json.loads(PRESETS.read_text())['models']
    if config.get('model') not in presets or config.get('model_id') != presets[config['model']]:
        raise ValueError('Model-run architecture/weight tag is outside the four-model protocol')
    if config.get('smoke'):
        raise ValueError('A smoke-trained checkpoint is not a baseline research model')
    expected = json_value(metadata['preprocessing']['data_config'])
    before = file_digest(checkpoint)
    model, resolved = build_model(config['model_id'], pretrained=False)
    if json_value(resolved) != expected or json_value(metadata['data']['resolved_data_config']) != expected:
        raise ValueError('Recorded preprocessing differs from resolved model config; no silent substitution')
    if checkpoint_format == 'compact':
        load_weights(checkpoint, model)
    elif checkpoint_format == 'full' and trusted_full:
        state = read_checkpoint(checkpoint)
        if (state['config']['model_id'] != config['model_id']
                or state['config']['model'] != config['model']
                or json_value(state['data_metadata']) != metadata['data']):
            raise ValueError('Full checkpoint does not match the model-run metadata')
        model.load_state_dict(state['model'], strict=True)
    else:
        raise ValueError('Full checkpoint requires --trusted-full-checkpoint; compact uses weights_only=True')
    if file_digest(checkpoint) != before:
        raise ValueError('Checkpoint changed while loading')
    model.float().eval()
    return model, expected, {'model': config['model'], 'model_id': config['model_id'],
                            'variant': variant, 'checkpoint': str(checkpoint.resolve()),
                            'checkpoint_sha256': before, 'checkpoint_format': checkpoint_format,
                            'model_run': str(model_run.resolve()), 'training_metadata': metadata}
