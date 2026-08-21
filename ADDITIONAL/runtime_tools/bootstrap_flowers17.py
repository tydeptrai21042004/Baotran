#!/usr/bin/env python3
'''Prepare the official Oxford 17 Category Flower Dataset for ADDITIONAL CNN runs.

Only official Oxford VGG endpoints are used. We use official split 1
(trn1/val1/tst1) from datasplits.mat and preserve it identically for all methods
and model seeds. The resulting tree is exposed to torchvision ImageFolder and
the cloned whc-dt1d dataset router is patched locally in the Kaggle working copy.
'''
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path
from scipy.io import loadmat

OFFICIAL_PAGE = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/'
IMAGE_URLS = [
    'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
    'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
]
SPLIT_URLS = [
    'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat',
    'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat',
]
EXPECTED_SHA384 = 'ad5c0d6272d52d2d2c27726fde18bcede3f8a2c6cd834891bf9819fac4e3f6047cfb8bd921851cb3048323df3ad06e3d'
PAPER = 'M.-E. Nilsback and A. Zisserman, A Visual Vocabulary for Flower Classification, CVPR 2006, vol. 2, pp. 1447-1454.'


def digest(path: Path, alg='sha256') -> str:
    h = hashlib.new(alg)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(urls, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return 'cached:' + str(dest)
    tmp = dest.with_suffix(dest.suffix + '.part')
    tmp.unlink(missing_ok=True)
    last = None
    for url in urls:
        cmd = ['curl', '-L', '--fail', '--retry', '3', '--retry-delay', '3', '--connect-timeout', '20', '-o', str(tmp), url]
        print('DOWNLOAD', url, flush=True)
        p = subprocess.run(cmd)
        if p.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return url
        last = f'curl rc={p.returncode} for {url}'
        tmp.unlink(missing_ok=True)
    raise SystemExit('Official Oxford download failed: ' + str(last))


def prepare(data_root: Path, provenance_root: Path):
    base = data_root / 'oxford_flowers17'
    raw = base / 'raw'
    prepared = base / 'official_split1'
    marker = prepared / 'PREPARED.json'
    image_archive = raw / '17flowers.tgz'
    split_file = raw / 'datasplits.mat'
    src_img_dir = raw / 'jpg'
    image_source = 'already-prepared'
    split_source = 'already-prepared'
    if not marker.is_file():
        raw.mkdir(parents=True, exist_ok=True)
        image_source = download(IMAGE_URLS, image_archive)
        got = digest(image_archive, 'sha384')
        if got.lower() != EXPECTED_SHA384.lower():
            raise SystemExit(f'17flowers.tgz checksum mismatch: {got}')
        split_source = download(SPLIT_URLS, split_file)
        if not src_img_dir.is_dir():
            with tarfile.open(image_archive, 'r:gz') as tf:
                try:
                    tf.extractall(raw, filter='data')
                except TypeError:
                    tf.extractall(raw)
        imgs = sorted(src_img_dir.glob('image_*.jpg'))
        if len(imgs) != 1360:
            raise SystemExit(f'Expected 1360 Oxford Flowers17 images, found {len(imgs)}')
        mat = loadmat(split_file)
        required = ('trn1', 'val1', 'tst1')
        missing = [k for k in required if k not in mat]
        if missing:
            raise SystemExit(f'Official datasplits.mat missing keys: {missing}; keys={sorted(mat)}')
        split_ids = {k: [int(x) for x in mat[k].reshape(-1).tolist()] for k in required}
        all_ids = split_ids['trn1'] + split_ids['val1'] + split_ids['tst1']
        if len(all_ids) != 1360 or len(set(all_ids)) != 1360 or min(all_ids) != 1 or max(all_ids) != 1360:
            raise SystemExit('Official split 1 is not a disjoint full partition of image ids 1..1360')
        shutil.rmtree(prepared, ignore_errors=True)
        name_map = {'trn1': 'train', 'val1': 'val', 'tst1': 'test'}
        manifest = {'split_source': 'official datasplits.mat split 1', 'splits': {}}
        for key, out_split in name_map.items():
            ids = split_ids[key]
            manifest['splits'][out_split] = ids
            for image_id in ids:
                cls = (image_id - 1) // 80
                if not 0 <= cls < 17:
                    raise SystemExit(f'Bad class derived from image id {image_id}')
                src = src_img_dir / f'image_{image_id:04d}.jpg'
                dst_dir = prepared / out_split / f'class_{cls+1:02d}'
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
        meta = {
            'dataset': 'Oxford 17 Category Flower Dataset',
            'classes': 17,
            'images': 1360,
            'images_per_class': 80,
            'official_page': OFFICIAL_PAGE,
            'image_source_url': image_source,
            'split_source_url': split_source,
            'image_archive_sha384': digest(image_archive, 'sha384'),
            'datasplits_sha256': digest(split_file, 'sha256'),
            'split_used': 'official split 1: trn1 / val1 / tst1',
            'paper': PAPER,
            'manifest': manifest,
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    meta = json.loads(marker.read_text(encoding='utf-8'))
    provenance_root.mkdir(parents=True, exist_ok=True)
    (provenance_root / 'DATASET_SOURCE.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return prepared, meta


def patch_router(repo_root: Path):
    p = repo_root / 'datasets' / 'build.py'
    text = p.read_text(encoding='utf-8')
    if 'def _build_flowers17(' not in text:
        marker = 'def _build_flowers102(args, split: str):\n'
        if marker not in text:
            raise SystemExit('Could not find _build_flowers102 insertion point in datasets/build.py')
        fn = '''def _build_flowers17(args, split: str):
    root = Path(args.data_path) / "oxford_flowers17" / "official_split1" / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"Prepared Oxford Flowers17 split not found at {root}. "
            "Run ADDITIONAL/runtime_tools/bootstrap_flowers17.py first."
        )
    ds = datasets.ImageFolder(root=str(root), transform=_img_transforms(args, split == "train"))
    if len(ds.classes) != 17:
        raise RuntimeError(f"Expected 17 Flowers17 classes in {root}, got {len(ds.classes)}")
    return ds, 17


'''
        text = text.replace(marker, fn + marker, 1)
    alias = '    ("flowers17", "oxfordflowers17", "oxford_flowers17"): _build_flowers17,\n'
    if alias not in text:
        marker = '    ("flowers102", "oxfordflowers102", "oxford_flowers102"): _build_flowers102,\n'
        if marker not in text:
            raise SystemExit('Could not find Flowers102 builder mapping insertion point')
        text = text.replace(marker, alias + marker, 1)
    p.write_text(text, encoding='utf-8')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', required=True, type=Path)
    ap.add_argument('--data-root', required=True, type=Path)
    ap.add_argument('--provenance-root', required=True, type=Path)
    ns = ap.parse_args()
    prepared, meta = prepare(ns.data_root.resolve(), ns.provenance_root.resolve())
    patched = patch_router(ns.repo_root.resolve())
    print(json.dumps({'prepared': str(prepared), 'router': str(patched), 'dataset': meta['dataset'], 'split': meta['split_used']}, indent=2))


if __name__ == '__main__':
    main()
