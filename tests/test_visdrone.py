import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from data.visdrone import (CLASSES, ClassificationTransform, VisDroneCrops,
                           build_index, make_loader, make_splits, prepare)


def source(root, split, name, rows):
    folder = root / split
    (folder / 'images').mkdir(parents=True, exist_ok=True)
    (folder / 'annotations').mkdir(exist_ok=True)
    im = Image.new('RGB', (20, 10), 'red')
    for x in range(10, 20):
        for y in range(10):
            im.putpixel((x, y), (0, 255, 0))
    im.save(folder / 'images' / f'{name}.png')
    (folder / 'annotations' / f'{name}.txt').write_text(rows)
    return f'{split}/images/{name}.png'


ROWS = '0,0,10,10,1,1,0,0\n10,0,10,10,1,4,1,2\n'


def test_multiple_objects_pixels_and_loader(tmp_path):
    image = source(tmp_path, 'train', 'a', ROWS)
    records, stats = build_index(tmp_path, [image])
    assert [r.label for r in records] == [0, 3]
    assert stats['accepted'] == 2
    ds = VisDroneCrops(tmp_path, records, ClassificationTransform(8))
    assert ds.raw_crop(0).getpixel((0, 0)) == (255, 0, 0)
    assert ds.raw_crop(1).getpixel((0, 0)) == (0, 255, 0)
    x, y = next(iter(make_loader(ds, batch_size=2)))
    assert x.shape == (2, 3, 8, 8)
    assert x.dtype == torch.float32 and x.min() == 0 and x.max() == 1
    assert y.tolist() == [0, 3]


def test_filtering_and_clipping(tmp_path):
    rows = ROWS + '\n'.join([
        '0,0,5,5,0,4,0,0', '0,0,5,5,1,0,0,0',
        '0,0,5,5,1,11,0,0', '0,0,-1,5,1,4,0,0',
        '30,0,2,2,1,4,0,0', '0,0,0,5,1,4,0,0',
        '-2,-2,5,5,1,10,0,0', 'bad,row', 'nan,0,1,1,1,1,0,0',
        '0,0,2,2,1,4.5,0,0', '0,0,2,2,2,4,0,0'])
    name = source(tmp_path, 'train', 'a', rows)
    records, stats = build_index(tmp_path, [name])
    assert len(records) == 3
    assert records[-1].box == (0, 0, 3, 3) and records[-1].label == 9
    assert stats == dict(images=1, rows=13, accepted=3, score_zero=1,
                        ignored_or_others=2, invalid_box=3, malformed=3,
                        invalid_label_or_score=1)


def test_all_ten_classes(tmp_path):
    name = source(tmp_path, 'train', 'a', '\n'.join(
        f'0,0,2,2,1,{c},0,0' for c in range(1, 11)))
    records, _ = build_index(tmp_path, [name])
    assert [r.label for r in records] == list(range(len(CLASSES)))


def fixture_splits(tmp_path):
    source(tmp_path, 'train', 't', ROWS)
    for i in range(5):
        source(tmp_path, 'val', f'v{i}', ROWS)


def test_split_reproducibility_and_no_source_leakage(tmp_path):
    fixture_splits(tmp_path)
    a = make_splits(tmp_path, 'train', 'val', seed=7)
    assert a == make_splits(tmp_path, 'train', 'val', seed=7)
    groups = list(a.values())
    for i, group in enumerate(groups):
        for other in groups[i + 1:]:
            assert set(group).isdisjoint(other)
    assert sorted(a['validation'] + a['holdout']) == [f'val/images/v{i}.png' for i in range(5)]
    for group in groups:
        records, _ = build_index(tmp_path, group)
        assert len(records) == 2 * len(group)


def test_preparation_artifacts_and_overwrite_guard(tmp_path):
    fixture_splits(tmp_path)
    output = tmp_path / 'results'
    report = prepare(tmp_path, output, 'train', 'val', seed=7)
    assert report['splits']['train']['classes']['car'] == 1
    assert report['splits']['train']['classes']['bus'] == 0
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['annotation_sha256']) == 6
    for split in ('train', 'validation', 'holdout'):
        assert (output / f'{split}.png').stat().st_size > 0
        assert json.loads((output / f'{split}.json').read_text())
    with pytest.raises(FileExistsError):
        prepare(tmp_path, output, 'train', 'val')


def test_missing_annotations_duplicates_and_overlap(tmp_path):
    name = source(tmp_path, 'train', 'a', ROWS)
    with pytest.raises(ValueError, match='Duplicate'):
        build_index(tmp_path, [name, name])
    source(tmp_path, 'val', 'a', ROWS)
    source(tmp_path, 'val', 'b', ROWS)
    with pytest.raises(ValueError, match='overlap'):
        make_splits(tmp_path, 'train', 'val')
    (tmp_path / 'train/annotations/a.txt').unlink()
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path, [name])


def test_loader_shuffle_reproducible(tmp_path):
    name = source(tmp_path, 'train', 'a', ROWS)
    records, _ = build_index(tmp_path, [name])
    ds = VisDroneCrops(tmp_path, records)
    def order():
        return [y.item() for _, y in make_loader(ds, 1, shuffle=True, seed=5)]
    assert order() == order()


def test_cli(tmp_path):
    import subprocess
    import sys
    fixture_splits(tmp_path)
    result = subprocess.run([sys.executable, '-m', 'data.prepare',
                             '--root', str(tmp_path), '--output', str(tmp_path / 'cli'),
                             '--train-split', 'train', '--val-split', 'val'],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['splits']['holdout']['crops'] == 2
