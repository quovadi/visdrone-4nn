"""Run with python -m data.prepare --root ... --output ..."""
import argparse
import json
from .visdrone import prepare


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-split', default='VisDrone2019-DET-train')
    parser.add_argument('--val-split', default='VisDrone2019-DET-val')
    parser.add_argument('--holdout-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
