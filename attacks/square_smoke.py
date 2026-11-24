"""CPU end-to-end Square smoke with generated crops and fixed toy weights."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import torch
from torch import nn
from attacks.data import selected_validation, dataset
from attacks.preprocessing import PixelModel, rgb_transform
from attacks.square import SquareConfig
from attacks.square_evaluation import evaluate_square
from attacks.square_results import read_square_predictions, query_summary
from training.consistency import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with TemporaryDirectory(prefix='square-smoke-') as temporary:
        root = Path(temporary)
        folder = root / 'VisDrone2019-DET-val' / 'images'
        folder.mkdir(parents=True)
        records = []
        for i, label in enumerate((0, 1, 2, 1)):
            image = folder / f'{i}.png'
            Image.new('RGB', (16, 16), (140 if i % 2 == 0 else 110,)*3).save(image)
            records.append({'image': image.relative_to(root).as_posix(),
                            'box': [1, 1, 15, 15], 'label': label, 'line': i+1})
        index = root / 'validation.json'
        index.write_text(json.dumps(records))
        cfg = {'input_size': [3, 224, 224], 'mean': [.5]*3, 'std': [.5]*3,
               'crop_pct': 1., 'crop_mode': 'center', 'interpolation': 'bicubic'}
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 10))
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].weight[0].fill_(1)
            model[-1].weight[1].fill_(-1)
            model[-1].bias.fill_(-20)
            model[-1].bias[:2].zero_()
        checkpoint = root / 'synthetic_weights.pt'
        torch.save(model.state_dict(), checkpoint)
        selected, identities = selected_validation(index, 42)
        info = {'model': 'synthetic', 'model_id': 'fixed-mean-toy', 'variant': 'synthetic-smoke',
                'checkpoint': str(checkpoint), 'checkpoint_sha256': file_digest(checkpoint),
                'validation_index': str(index), 'synthetic': True}
        run = evaluate_square(PixelModel(model, cfg), dataset(root, selected, rgb_transform(cfg)),
                              identities, info, cfg, SquareConfig(4/255, 8), args.output, batch_size=2)
        rows = read_square_predictions(run)
        print(json.dumps({'run': str(run), 'synthetic_only': True, **query_summary(rows)}, indent=2))


if __name__ == '__main__':
    main()
