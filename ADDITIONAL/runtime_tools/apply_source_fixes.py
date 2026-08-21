#!/usr/bin/env python3
"""Small audited runtime fixes required by v6 dense sessions."""
from pathlib import Path
import hashlib, json

root=Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2]/'dense_prediction').exists() else Path.cwd()
p=root/'dense_prediction'/'datasets.py'
text=p.read_text()
old='''        images = {p.stem: p for p in _files(image_dir)}\n        masks = {p.stem: p for p in _files(mask_dir)}\n        keys = sorted(set(images).intersection(masks))\n'''
new='''        images = {p.stem: p for p in _files(image_dir)}\n        # Real PennFudan masks use names such as FudanPed00001_mask.png,\n        # whereas images use FudanPed00001.png. Normalize only the documented\n        # trailing _mask suffix before pairing.\n        masks = {(p.stem[:-5] if p.stem.endswith("_mask") else p.stem): p for p in _files(mask_dir)}\n        keys = sorted(set(images).intersection(masks))\n'''
if old in text:
    p.write_text(text.replace(old,new))
elif 'p.stem.endswith("_mask")' not in text:
    raise SystemExit('PennFudan pairing source shape changed; refusing blind patch')
print(json.dumps({'patched':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
