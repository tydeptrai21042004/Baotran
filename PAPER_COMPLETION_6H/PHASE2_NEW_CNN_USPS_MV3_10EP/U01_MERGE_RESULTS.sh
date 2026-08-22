#!/usr/bin/env bash
set -Eeuo pipefail
WORKDIR=/kaggle/working
OUT="$WORKDIR/U01_USPS_MV3_10EP_MERGED"
rm -rf "$OUT"; mkdir -p "$OUT"
python - <<'PYMERGE'
from pathlib import Path
import zipfile
inp=Path('/kaggle/input'); out=Path('/kaggle/working/U01_USPS_MV3_10EP_MERGED')
zs=sorted(inp.rglob('U01_USPS_MV3_10EP_*_results.zip'))
if not zs: raise SystemExit('No U01 result ZIPs found under /kaggle/input')
for z in zs:
    print('MERGE',z)
    with zipfile.ZipFile(z) as f: f.extractall(out)
print('MERGED_ZIPS',len(zs),'->',out)
PYMERGE
find "$OUT" -type f | sort | sed -n '1,260p'
