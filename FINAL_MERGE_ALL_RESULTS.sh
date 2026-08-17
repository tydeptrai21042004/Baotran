#!/usr/bin/env bash
set -Eeuo pipefail
W=/kaggle/working; M="$W/FINAL_MERGED"; rm -rf "$M" "$W/merge-dt1d" "$W/merge-vit"; mkdir -p "$M/extracted" "$M/cnn_raw" "$M/dense_raw" "$M/vit_csv" "$M/final_tables" "$M/final_figures"
python - <<'PYEX'
import zipfile
from pathlib import Path
inp=Path('/kaggle/input'); out=Path('/kaggle/working/FINAL_MERGED/extracted'); zs=list(inp.rglob('*_results.zip')) if inp.exists() else []
print('ZIP count',len(zs))
for z in zs:
 d=out/z.stem; d.mkdir(parents=True,exist_ok=True)
 try: zipfile.ZipFile(z).extractall(d); print('EXTRACT',z)
 except Exception as e: print('WARN',z,e)
PYEX
git clone --depth 1 "${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}" "$W/merge-dt1d"
git clone --depth 1 "${DT1D_VIT_REPO_URL:-https://github.com/tydeptrai21042004/whc-vit.git}" "$W/merge-vit"
[[ -z "${DT1D_CNN_COMMIT:-}" ]] || git -C "$W/merge-dt1d" fetch -q --depth 1 origin "$DT1D_CNN_COMMIT" || true
[[ -z "${DT1D_CNN_COMMIT:-}" ]] || git -C "$W/merge-dt1d" checkout -q "$DT1D_CNN_COMMIT"
[[ -z "${DT1D_VIT_COMMIT:-}" ]] || git -C "$W/merge-vit" fetch -q --depth 1 origin "$DT1D_VIT_COMMIT" || true
[[ -z "${DT1D_VIT_COMMIT:-}" ]] || git -C "$W/merge-vit" checkout -q "$DT1D_VIT_COMMIT"
python -m pip install -q --upgrade-strategy only-if-needed -r "$W/merge-dt1d/requirements-kaggle.txt"
python - <<'PYMERGE'
from pathlib import Path
import shutil
b=Path('/kaggle/working/FINAL_MERGED'); ex=b/'extracted'; cnn=b/'cnn_raw'; dense=b/'dense_raw'; vit=b/'vit_csv'
for p in ex.rglob('run_*'):
 if not p.is_dir(): continue
 for t in p.iterdir():
  if not t.is_dir() or t.name=='aggregated': continue
  for m in t.iterdir():
   if not m.is_dir(): continue
   dst=cnn/t.name/m.name; shutil.rmtree(dst,ignore_errors=True); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(m,dst)
for p in ex.rglob('dense_*'):
 if not p.is_dir(): continue
 for t in p.iterdir():
  if not t.is_dir() or t.name=='aggregated': continue
  for m in t.iterdir():
   if not m.is_dir(): continue
   dst=dense/t.name/m.name; shutil.rmtree(dst,ignore_errors=True); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(m,dst)
for p in ex.rglob('*fair_three_seed.csv'):
 dst=vit/p.name
 if dst.exists(): dst=vit/(p.parent.parent.name+'__'+p.name)
 shutil.copy2(p,dst)
print('CNN',sorted(x.name for x in cnn.iterdir()) if cnn.exists() else [])
print('DENSE',sorted(x.name for x in dense.iterdir()) if dense.exists() else [])
print('VIT',sorted(x.name for x in vit.iterdir()) if vit.exists() else [])
PYMERGE
cd "$W/merge-dt1d"
for T in table_03 table_04 table_05 table_06 table_07 table_08 table_09 table_10 table_11 table_12 table_13 table_14_15 table_18_19 ablation_dtd_resnet18 ablation_flowers102_resnet50; do python tools/aggregate_cnn_paper.py --root "$M/cnn_raw" --target "$T" --output-dir "$M/final_tables/$T" --require-seeds 0,1,2 --allow-incomplete || true; done
python tools/aggregate_dense_results.py --input-root "$M/dense_raw" --output-dir "$M/final_tables/dense" --require-seeds 0,1,2 || true
python tools/plot_figure_01_single_seed.py --root "$M/cnn_raw" --seed 0 --output "$M/final_figures/figure_01.png" || true
python tools/plot_figure_04_tradeoff.py --root "$M/cnn_raw" --seed 0 --output "$M/final_figures/figure_04.png" || true
python tools/plot_figure_02_spectral.py --output "$M/final_figures/figure_02.png" || true
python tools/plot_figure_03_architecture.py --output "$M/final_figures/figure_03.png" || true
cd "$W"; zip -qr FINAL_ALL_RESULTS.zip FINAL_MERGED; ls -lh FINAL_ALL_RESULTS.zip
