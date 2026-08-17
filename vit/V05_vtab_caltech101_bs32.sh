#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Manuscript ViT result: VTAB-Caltech101 / ViT-B16 / BS32
SESSION_ID="V05"; DATASET="vtab-caltech101"; BATCH_SIZE="32"; VPT_TOKENS="10"; METHODS="dt1d,vpt,pfeiffer,full,linear"; RESULT_MODE="table"; FINAL_SEEDS="0,1,2"
REPO_URL="${DT1D_VIT_REPO_URL:-https://github.com/tydeptrai21042004/whc-vit.git}"; REPO_COMMIT="${DT1D_VIT_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-vit-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; MODEL_ROOT="$WORKDIR/models_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/vit_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID DATASET BATCH_SIZE VPT_TOKENS METHODS RESULT_MODE FINAL_SEEDS REPO_DIR DATA_ROOT MODEL_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
pack_results() { [[ -d "$OUTPUT_ROOT" ]] || return 0; python - <<'PYZIP'
import os,zipfile
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);d=Path(os.environ['RESULT_ZIP']);d.unlink(missing_ok=True)
with zipfile.ZipFile(d,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in r.rglob('*'):
  if p.is_file() and p.suffix.lower() not in {'.pth','.pt','.ckpt'}: z.write(p,p.relative_to(r.parent))
print('RESULT_ZIP=',d)
PYZIP
}
trap 'rc=$?; trap - EXIT; [[ -f "$RESULT_ZIP" ]] || pack_results || true; exit $rc' EXIT
rm -rf "$REPO_DIR"; for n in 1 2 3; do git clone --depth 1 "$REPO_URL" "$REPO_DIR" && break || sleep $((n*5)); done; cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then git fetch --depth 1 origin "$REPO_COMMIT"; git checkout --detach "$REPO_COMMIT"; fi
SOURCE_COMMIT="$(git rev-parse HEAD)"; export SOURCE_COMMIT
python -m pip install -q --upgrade-strategy only-if-needed scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML tensorflow-datasets six
if ! python -c 'import tensorflow; print(tensorflow.__version__)'; then python -m pip install -q 'tensorflow>=2.16,<2.20'; fi
python validate_dt1d_vit.py; python verify_vpt_original.py; python verify_fair_protocol.py
python -m pytest -q tests/test_dt1d_token_adapter.py tests/test_fair_protocol.py tests/test_repository_contracts.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);zs=list(Path('/kaggle/input').rglob(f'V05_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_V05';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'vit_V05'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT" "$MODEL_ROOT"
WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz"; if [[ ! -s "$WEIGHT_FILE" ]]; then tmp="$WEIGHT_FILE.part"; rm -f "$tmp"; curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz' -o "$tmp"; mv "$tmp" "$WEIGHT_FILE"; fi
python - "$WEIGHT_FILE" <<'PYW'
import sys,numpy as np; z=np.load(sys.argv[1]); assert len(z.files)==200,len(z.files); print('ViT tensors=',len(z.files))
PYW
if [[ "$DATASET" == "flowers102" ]]; then
  FLOWERS_ROOT="$DATA_ROOT/flowers_download"; mkdir -p "$FLOWERS_ROOT"; DATA_PATH="$(python - "$FLOWERS_ROOT" <<'PYF'
import json,sys
from pathlib import Path
from torchvision.datasets import Flowers102
r=Path(sys.argv[1]); ds={s:Flowers102(str(r),split=s,download=True) for s in ('train','val','test')}; img=r/'flowers-102'/'jpg'; assert img.is_dir()
for s,d in ds.items(): (img/f'{s}.json').write_text(json.dumps({Path(str(p)).name:int(y) for p,y in zip(d._image_files,d._labels)}))
print(img)
PYF
)"
else
  DATA_PATH="$DATA_ROOT/tfds"; mkdir -p "$DATA_PATH"; export DATA_PATH DATASET
  python - <<'PYD'
import os,tensorflow_datasets as tfds
spec={'vtab-caltech101':'caltech101:3.*.*','vtab-dtd':'dtd:3.*.*','vtab-eurosat':'eurosat/rgb:2.*.*'}[os.environ['DATASET']]; b=tfds.builder(spec,data_dir=os.environ['DATA_PATH']); b.download_and_prepare(); print('TFDS READY',b.info.full_name)
PYD
fi
export DATA_PATH
COMMON=(--dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT" --batch-sizes "$BATCH_SIZE" --methods "$METHODS" --epochs 10 --resolution 224 --result-mode table --seeds 0,1,2 --tune-seed 42 --weight-decay 1e-4 --warmup-epoch 1 --patience 20 --vpt-tokens "$VPT_TOKENS" --allow-boundary-best)
PREFLIGHT="$WORKDIR/_preflight_$SESSION_ID"; rm -rf "$PREFLIGHT"; python run_fair_vit_comparison.py "${COMMON[@]}" --output-root "$PREFLIGHT" --gpus cpu --dry-run; rm -rf "$PREFLIGHT"
LOG="$OUTPUT_ROOT/session.log"; mkdir -p "$OUTPUT_ROOT"; set +e; setsid python run_fair_vit_comparison.py "${COMMON[@]}" --output-root "$OUTPUT_ROOT" --gpus auto >"$LOG" 2>&1 & p=$!
while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then echo 'TIME CAP'; kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; break; fi; sleep 10; done; wait "$p"; RUN_RC=$?; set -e
python - <<'PYSTATUS'
import json,os,pandas as pd
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']); stem=os.environ['DATASET'].replace('-','_'); cp=r/'aggregated'/f'{stem}_fair_three_seed.csv'; expected=set(os.environ['METHODS'].split(',')); actual=set(); complete=False
if cp.is_file():
 df=pd.read_csv(cp); df=df[df['batch_size'].astype(int)==int(os.environ['BATCH_SIZE'])]; actual=set(df['method_key'].astype(str)); complete=(actual==expected and len(df)==len(expected))
status={'session':os.environ['SESSION_ID'],'family':'vit','dataset':os.environ['DATASET'],'batch_size':int(os.environ['BATCH_SIZE']),'methods':sorted(expected),'final_seeds':[0,1,2],'complete':complete,'actual_methods':sorted(actual),'aggregate_csv':str(cp),'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). If incomplete/time-capped, attach its ZIP and rerun same cell."; exit 0
