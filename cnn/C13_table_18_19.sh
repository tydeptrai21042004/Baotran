#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# EuroSAT/MobileNetV3-Small/25ep
SESSION_ID="C13"
TARGET="table_18_19"
METHODS="linear,dt1d,bam,conv_r4,full"
GROUP0="linear,bam,full"
GROUP1="dt1d,conv_r4"
SEEDS="0,1,2"
KIND="table"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS KIND REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

pack_results() {
  [[ -d "$OUTPUT_ROOT" ]] || return 0
  python - <<'PYZIP'
import os, zipfile
from pathlib import Path
root=Path(os.environ['OUTPUT_ROOT']); dst=Path(os.environ['RESULT_ZIP'])
if dst.exists(): dst.unlink()
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() not in {'.pth','.pt','.ckpt'}:
            z.write(p,p.relative_to(root.parent))
print('RESULT_ZIP=',dst)
PYZIP
  ls -lh "$RESULT_ZIP" || true
}
trap 'rc=$?; trap - EXIT; [[ -f "$RESULT_ZIP" ]] || pack_results || true; exit $rc' EXIT

rm -rf "$REPO_DIR"
for n in 1 2 3; do git clone --depth 1 "$REPO_URL" "$REPO_DIR" && break || sleep $((n*5)); done
[[ -d "$REPO_DIR/.git" ]] || { echo "ERROR: clone failed" >&2; exit 2; }
cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then git fetch --depth 1 origin "$REPO_COMMIT"; git checkout --detach "$REPO_COMMIT"; fi
SOURCE_COMMIT="$(git rev-parse HEAD)"; export SOURCE_COMMIT; echo "SOURCE_COMMIT=$SOURCE_COMMIT"
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py
python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_seed_policy.py tests/test_fair_protocol_contract.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(), 'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
PYGPU

rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os, shutil, zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); inp=Path('/kaggle/input')
zs=list(inp.rglob(f'C13_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C13'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C13'))
    if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Pre-download dataset and pretrained backbone once before the two GPU workers start.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=os.environ['DATA_ROOT']; ds=cfg['dataset']; bb=cfg['backbone']; print('PRELOAD',ds,bb)
if ds=='flowers102': [datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd': [datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn': [datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101': [datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet': [datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101': datasets.Caltech101(root=root,download=True)
elif ds=='eurosat': datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft': [datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else: raise RuntimeError('No standalone preloader for '+ds)
if bb=='resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0': models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small': models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else: raise RuntimeError('No standalone backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

python - <<'PYPOLICY'
import os,yaml
from pathlib import Path
p=Path('configs/experiments')/(os.environ['TARGET']+'.yaml'); d=yaml.safe_load(p.read_text())
seeds=[int(x) for x in os.environ['SEEDS'].split(',') if x]
assert all(s in d['seeds'] for s in seeds)
policy=d.get('seed_policy')
if os.environ['KIND']=='figure': assert policy=='single_seed_figure' and len(seeds)==1
else: assert policy=='multi_seed_table' and len(seeds)>=3
wanted=[x for x in os.environ['METHODS'].split(',') if x]
assert set(wanted)<=set(d['method_order'])
print('POLICY PASS',p.name,policy,seeds,wanted)
PYPOLICY

# Preflight must use a disposable output tree. The real OUTPUT_ROOT may contain
# restored progress from an earlier Kaggle session, and dry-run is allowed to
# recreate method directories. Never let preflight touch resumable real results.
PREFLIGHT_ROOT="$WORKDIR/_preflight_${SESSION_ID}"
rm -rf "$PREFLIGHT_ROOT"; mkdir -p "$PREFLIGHT_ROOT"
for G in "$GROUP0" "$GROUP1"; do
  [[ -z "$G" ]] && continue
  python tools/run_cnn_paper.py --target "$TARGET" --methods "$G" --seeds "$SEEDS" --output-root "$PREFLIGHT_ROOT" --data-path "$DATA_ROOT" --device cpu --plan-only
  python tools/run_cnn_paper.py --target "$TARGET" --methods "$G" --seeds "$SEEDS" --output-root "$PREFLIGHT_ROOT" --data-path "$DATA_ROOT" --device cpu --dry-run
done
rm -rf "$PREFLIGHT_ROOT"

run_group() {
  local group="$1" dev="$2" log="$3"
  [[ -z "$group" ]] && return 0
  python tools/run_cnn_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --output-root "$OUTPUT_ROOT" --data-path "$DATA_ROOT" --device "$dev" --skip-if-complete >"$log" 2>&1
}
export -f run_group
set +e
PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU >= 2 )); then
  [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }
  [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }
else
  [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }
fi
RUN_RC=0
while ((${#PIDS[@]})); do
  alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  (( alive==0 )) && break
  if (( $(date +%s) >= DEADLINE_EPOCH )); then
    echo 'TIME CAP reached; terminating process groups.'
    for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
    sleep 20
    for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
    RUN_RC=124; break
  fi
  sleep 10
done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU < 2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s) < DEADLINE_EPOCH )); then
  setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!
  while kill -0 "$p" 2>/dev/null; do
    if (( $(date +%s) >= DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi
    sleep 10
  done
  wait "$p" || RUN_RC=$?
fi
set -e

AGG="$OUTPUT_ROOT/aggregated/$TARGET"; mkdir -p "$AGG"
if [[ "$KIND" == "figure" ]]; then
  [[ "$TARGET" == "figure_01" ]] && python tools/plot_figure_01_single_seed.py --root "$OUTPUT_ROOT" --seed 0 --output "$AGG/figure_01.png" || true
  [[ "$TARGET" == "figure_04" ]] && python tools/plot_figure_04_tradeoff.py --root "$OUTPUT_ROOT" --seed 0 --output "$AGG/figure_04.png" || true
else
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --output-dir "$AGG" --require-seeds "$SEEDS" --allow-incomplete || true
fi
python - <<'PYSTATUS'
import json,os
from pathlib import Path
root=Path(os.environ['OUTPUT_ROOT']); target=os.environ['TARGET']; methods=[x for x in os.environ['METHODS'].split(',') if x]; seeds=[int(x) for x in os.environ['SEEDS'].split(',') if x]
missing=[[m,s] for m in methods for s in seeds if not (root/target/m/f'seed_{s}'/'test_summary.json').is_file()]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':target,'kind':os.environ['KIND'],'methods':methods,'seeds':seeds,'complete':not missing,'missing':missing,'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT
echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). If incomplete/time-capped, attach ${SESSION_ID}_results.zip and rerun this same cell."
exit 0
