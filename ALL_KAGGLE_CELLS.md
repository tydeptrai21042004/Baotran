# ALL KAGGLE CELLS — DT1D/ViT final manuscript reruns

Each section below is a standalone Kaggle `%%bash`-style shell cell. Run **one session per cell**. Heavy experiments are already split into independent sessions; each session has a hard 710-minute cap and writes a resumable ZIP.

## 1 — C01A — DTD/ResNet50/100ep part 1 of 2

Target: `table_03` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,residual`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# DTD/ResNet50/100ep part 1 of 2
SESSION_ID="C01A"
TARGET="table_03"
METHODS="linear,bitfit,ssf,dt1d,residual"
GROUP0="linear,ssf,residual"
GROUP1="bitfit,dt1d"
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
zs=list(inp.rglob(f'C01A_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C01A'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C01A'))
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

```

## 2 — C01B — DTD/ResNet50/100ep part 2 of 2

Target: `table_03` | Seeds: `0,1,2` | Methods: `conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# DTD/ResNet50/100ep part 2 of 2
SESSION_ID="C01B"
TARGET="table_03"
METHODS="conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="conv_r8,conv_r4,full"
GROUP1="conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C01B_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C01B'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C01B'))
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

```

## 3 — C02 — Flowers102/ResNet50/100ep

Target: `table_04` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,residual,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Flowers102/ResNet50/100ep
SESSION_ID="C02"
TARGET="table_04"
METHODS="linear,bitfit,ssf,dt1d,residual,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,residual,conv_r6,conv_r2"
GROUP1="bitfit,dt1d,conv_r8,conv_r4,full"
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
zs=list(inp.rglob(f'C02_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C02'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C02'))
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

```

## 4 — C03 — Flowers102/ResNet18/10ep

Target: `table_05` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Flowers102/ResNet18/10ep
SESSION_ID="C03"
TARGET="table_05"
METHODS="linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,conv_r8,conv_r4,full"
GROUP1="bitfit,dt1d,conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C03_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C03'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C03'))
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

```

## 5 — C04 — Flowers102/ResNet18/100ep/BS64

Target: `table_06` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Flowers102/ResNet18/100ep/BS64
SESSION_ID="C04"
TARGET="table_06"
METHODS="linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,conv_r8,conv_r4,full"
GROUP1="bitfit,dt1d,conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C04_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C04'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C04'))
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

```

## 6 — C05 — Flowers102/ResNet18/100ep/BS32

Target: `table_07` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Flowers102/ResNet18/100ep/BS32
SESSION_ID="C05"
TARGET="table_07"
METHODS="linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,conv_r8,conv_r4,full"
GROUP1="bitfit,dt1d,conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C05_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C05'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C05'))
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

```

## 7 — C06A — SVHN/ResNet50/10ep part 1 of 2

Target: `table_08` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# SVHN/ResNet50/10ep part 1 of 2
SESSION_ID="C06A"
TARGET="table_08"
METHODS="linear,bitfit,ssf,dt1d,full"
GROUP0="linear,ssf,full"
GROUP1="bitfit,dt1d"
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
zs=list(inp.rglob(f'C06A_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C06A'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C06A'))
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

```

## 8 — C06B — SVHN/ResNet50/10ep part 2 of 2

Target: `table_08` | Seeds: `0,1,2` | Methods: `conv_r8,conv_r6,conv_r4,conv_r2`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# SVHN/ResNet50/10ep part 2 of 2
SESSION_ID="C06B"
TARGET="table_08"
METHODS="conv_r8,conv_r6,conv_r4,conv_r2"
GROUP0="conv_r8,conv_r4"
GROUP1="conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C06B_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C06B'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C06B'))
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

```

## 9 — C07 — Oxford-IIIT Pet/ResNet50/10ep

Target: `table_09` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Oxford-IIIT Pet/ResNet50/10ep
SESSION_ID="C07"
TARGET="table_09"
METHODS="linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,conv_r8,conv_r4,full"
GROUP1="bitfit,dt1d,conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C07_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C07'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C07'))
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

```

## 10 — C08A — Food101/ResNet18/10ep part 1 of 2

Target: `table_10` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Food101/ResNet18/10ep part 1 of 2
SESSION_ID="C08A"
TARGET="table_10"
METHODS="linear,bitfit,ssf,dt1d,full"
GROUP0="linear,ssf,full"
GROUP1="bitfit,dt1d"
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
zs=list(inp.rglob(f'C08A_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C08A'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C08A'))
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

```

## 11 — C08B — Food101/ResNet18/10ep part 2 of 2

Target: `table_10` | Seeds: `0,1,2` | Methods: `conv_r8,conv_r6,conv_r4,conv_r2`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Food101/ResNet18/10ep part 2 of 2
SESSION_ID="C08B"
TARGET="table_10"
METHODS="conv_r8,conv_r6,conv_r4,conv_r2"
GROUP0="conv_r8,conv_r4"
GROUP1="conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C08B_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C08B'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C08B'))
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

```

## 12 — C09A — Food101/EfficientNet-B0/10ep part 1 of 2

Target: `table_11` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Food101/EfficientNet-B0/10ep part 1 of 2
SESSION_ID="C09A"
TARGET="table_11"
METHODS="linear,bitfit,ssf,dt1d,full"
GROUP0="linear,ssf,full"
GROUP1="bitfit,dt1d"
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
zs=list(inp.rglob(f'C09A_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C09A'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C09A'))
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

```

## 13 — C09B — Food101/EfficientNet-B0/10ep part 2 of 2

Target: `table_11` | Seeds: `0,1,2` | Methods: `conv_r8,conv_r6,conv_r4,conv_r2`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Food101/EfficientNet-B0/10ep part 2 of 2
SESSION_ID="C09B"
TARGET="table_11"
METHODS="conv_r8,conv_r6,conv_r4,conv_r2"
GROUP0="conv_r8,conv_r4"
GROUP1="conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C09B_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C09B'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C09B'))
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

```

## 14 — C10 — Oxford-IIIT Pet/EfficientNet-B0/10ep

Target: `table_12` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Oxford-IIIT Pet/EfficientNet-B0/10ep
SESSION_ID="C10"
TARGET="table_12"
METHODS="linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,conv_r8,conv_r4,full"
GROUP1="bitfit,dt1d,conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C10_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C10'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C10'))
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

```

## 15 — C11 — Oxford-IIIT Pet/EfficientNet-B0/100ep

Target: `table_13` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Oxford-IIIT Pet/EfficientNet-B0/100ep
SESSION_ID="C11"
TARGET="table_13"
METHODS="linear,bitfit,ssf,dt1d,conv_r8,conv_r6,conv_r4,conv_r2,full"
GROUP0="linear,ssf,conv_r8,conv_r4,full"
GROUP1="bitfit,dt1d,conv_r6,conv_r2"
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
zs=list(inp.rglob(f'C11_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C11'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C11'))
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

```

## 16 — C12 — Caltech101/ResNet18/10ep accuracy+efficiency

Target: `table_14_15` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,bam,lora_conv,residual,sidetune,conv_r4,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Caltech101/ResNet18/10ep accuracy+efficiency
SESSION_ID="C12"
TARGET="table_14_15"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,sidetune,conv_r4,full"
GROUP0="linear,ssf,bam,residual,conv_r4"
GROUP1="bitfit,dt1d,lora_conv,sidetune,full"
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
zs=list(inp.rglob(f'C12_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_C12'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_C12'))
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

```

## 17 — C13 — EuroSAT/MobileNetV3-Small/25ep

Target: `table_18_19` | Seeds: `0,1,2` | Methods: `linear,dt1d,bam,conv_r4,full`

```bash
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

```

## 18 — F01 — Figure 1 representative convergence, seed 0 only

Target: `figure_01` | Seeds: `0` | Methods: `dt1d`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Figure 1 representative convergence, seed 0 only
SESSION_ID="F01"
TARGET="figure_01"
METHODS="dt1d"
GROUP0="dt1d"
GROUP1=""
SEEDS="0"
KIND="figure"
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
zs=list(inp.rglob(f'F01_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_F01'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_F01'))
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

```

## 19 — F04 — Figure 4 representative trade-off, seed 0 only

Target: `figure_04` | Seeds: `0` | Methods: `linear,bitfit,ssf,dt1d,bam,lora_conv,residual,sidetune,conv_r4,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Figure 4 representative trade-off, seed 0 only
SESSION_ID="F04"
TARGET="figure_04"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,sidetune,conv_r4,full"
GROUP0="linear,ssf,bam,residual,conv_r4"
GROUP1="bitfit,dt1d,lora_conv,sidetune,full"
SEEDS="0"
KIND="figure"
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
zs=list(inp.rglob(f'F04_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_F04'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_F04'))
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

```

## 20 — A01 — Reviewer component ablation DTD/ResNet18

Target: `ablation_dtd_resnet18` | Seeds: `0,1,2` | Methods: `dt1d,previous_plain_axial,dt1d_no_weighted_shift,dt1d_no_l1_projection,routing_reference,direct_symmetric,unshared_coefficients,height_only,fixed_average_routing,single_dilation,gate_off,pointwise_on`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Reviewer component ablation DTD/ResNet18
SESSION_ID="A01"
TARGET="ablation_dtd_resnet18"
METHODS="dt1d,previous_plain_axial,dt1d_no_weighted_shift,dt1d_no_l1_projection,routing_reference,direct_symmetric,unshared_coefficients,height_only,fixed_average_routing,single_dilation,gate_off,pointwise_on"
GROUP0="dt1d,dt1d_no_weighted_shift,routing_reference,unshared_coefficients,fixed_average_routing,gate_off"
GROUP1="previous_plain_axial,dt1d_no_l1_projection,direct_symmetric,height_only,single_dilation,pointwise_on"
SEEDS="0,1,2"
KIND="ablation"
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
zs=list(inp.rglob(f'A01_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_A01'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_A01'))
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

```

## 21 — A02 — Reviewer component ablation Flowers102/ResNet50

Target: `ablation_flowers102_resnet50` | Seeds: `0,1,2` | Methods: `dt1d,previous_plain_axial,dt1d_no_weighted_shift,dt1d_no_l1_projection,routing_reference,direct_symmetric,unshared_coefficients,height_only,fixed_average_routing,single_dilation,gate_off,pointwise_on`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Reviewer component ablation Flowers102/ResNet50
SESSION_ID="A02"
TARGET="ablation_flowers102_resnet50"
METHODS="dt1d,previous_plain_axial,dt1d_no_weighted_shift,dt1d_no_l1_projection,routing_reference,direct_symmetric,unshared_coefficients,height_only,fixed_average_routing,single_dilation,gate_off,pointwise_on"
GROUP0="dt1d,dt1d_no_weighted_shift,routing_reference,unshared_coefficients,fixed_average_routing,gate_off"
GROUP1="previous_plain_axial,dt1d_no_l1_projection,direct_symmetric,height_only,single_dilation,pointwise_on"
SEEDS="0,1,2"
KIND="ablation"
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
zs=list(inp.rglob(f'A02_results.zip')) if inp.exists() else []
if zs:
    z=max(zs,key=lambda p:p.stat().st_mtime); tmp=Path('/kaggle/working')/f'_restore_A02'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'run_A02'))
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

```

## 22 — D01A — DRIVE binary ViT dense part 1 of 2

Target: `binary_vit_drive` | Seeds: `0,1,2` | Methods: `linear,dt1d`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# DRIVE binary ViT dense part 1 of 2
SESSION_ID="D01A"; TARGET="binary_vit_drive"; METHODS="linear,dt1d"; GROUP0="linear"; GROUP1="dt1d"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D01A_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D01A';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D01A'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 23 — D01B — DRIVE binary ViT dense part 2 of 2

Target: `binary_vit_drive` | Seeds: `0,1,2` | Methods: `bitfit,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# DRIVE binary ViT dense part 2 of 2
SESSION_ID="D01B"; TARGET="binary_vit_drive"; METHODS="bitfit,full"; GROUP0="bitfit"; GROUP1="full"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D01B_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D01B';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D01B'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 24 — D02A — DRIVE DeepLab part 1 of 2

Target: `binary_deeplab_drive` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# DRIVE DeepLab part 1 of 2
SESSION_ID="D02A"; TARGET="binary_deeplab_drive"; METHODS="linear,bitfit,ssf,dt1d,full"; GROUP0="linear,ssf,full"; GROUP1="bitfit,dt1d"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D02A_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D02A';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D02A'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 25 — D02B — DRIVE DeepLab part 2 of 2

Target: `binary_deeplab_drive` | Seeds: `0,1,2` | Methods: `bam,lora_conv,residual_adapter,conv_adapter`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# DRIVE DeepLab part 2 of 2
SESSION_ID="D02B"; TARGET="binary_deeplab_drive"; METHODS="bam,lora_conv,residual_adapter,conv_adapter"; GROUP0="bam,residual_adapter"; GROUP1="lora_conv,conv_adapter"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D02B_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D02B';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D02B'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 26 — D03 — Oxford-IIIT Pet semantic segmentation DeepLab

Target: `semantic_pet_deeplab` | Seeds: `0,1,2` | Methods: `linear,dt1d,bam,conv_adapter,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Oxford-IIIT Pet semantic segmentation DeepLab
SESSION_ID="D03"; TARGET="semantic_pet_deeplab"; METHODS="linear,dt1d,bam,conv_adapter,full"; GROUP0="linear,bam,full"; GROUP1="dt1d,conv_adapter"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D03_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D03';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D03'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 27 — D04A — PennFudan binary ViT dense part 1 of 2

Target: `binary_vit_pennfudan` | Seeds: `0,1,2` | Methods: `linear,dt1d`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# PennFudan binary ViT dense part 1 of 2
SESSION_ID="D04A"; TARGET="binary_vit_pennfudan"; METHODS="linear,dt1d"; GROUP0="linear"; GROUP1="dt1d"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D04A_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D04A';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D04A'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 28 — D04B — PennFudan binary ViT dense part 2 of 2

Target: `binary_vit_pennfudan` | Seeds: `0,1,2` | Methods: `bitfit,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# PennFudan binary ViT dense part 2 of 2
SESSION_ID="D04B"; TARGET="binary_vit_pennfudan"; METHODS="bitfit,full"; GROUP0="bitfit"; GROUP1="full"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D04B_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D04B';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D04B'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 29 — D05A — PennFudan DeepLab part 1 of 2

Target: `binary_deeplab_pennfudan` | Seeds: `0,1,2` | Methods: `linear,bitfit,ssf,dt1d,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# PennFudan DeepLab part 1 of 2
SESSION_ID="D05A"; TARGET="binary_deeplab_pennfudan"; METHODS="linear,bitfit,ssf,dt1d,full"; GROUP0="linear,ssf,full"; GROUP1="bitfit,dt1d"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D05A_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D05A';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D05A'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 30 — D05B — PennFudan DeepLab part 2 of 2

Target: `binary_deeplab_pennfudan` | Seeds: `0,1,2` | Methods: `bam,lora_conv,residual_adapter,conv_adapter`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# PennFudan DeepLab part 2 of 2
SESSION_ID="D05B"; TARGET="binary_deeplab_pennfudan"; METHODS="bam,lora_conv,residual_adapter,conv_adapter"; GROUP0="bam,residual_adapter"; GROUP1="lora_conv,conv_adapter"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D05B_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D05B';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D05B'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 31 — D06 — Oxford-IIIT Pet Faster R-CNN feasibility

Target: `detection_pet_fasterrcnn` | Seeds: `0,1,2` | Methods: `head_only,dt1d,bam,conv_adapter,full`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Oxford-IIIT Pet Faster R-CNN feasibility
SESSION_ID="D06"; TARGET="detection_pet_fasterrcnn"; METHODS="head_only,dt1d,bam,conv_adapter,full"; GROUP0="head_only,bam,full"; GROUP1="dt1d,conv_adapter"; SEEDS="0,1,2"
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/whc-dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/dense_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; DEADLINE_EPOCH="$(( $(date +%s) + 710*60 ))"
export SESSION_ID TARGET METHODS GROUP0 GROUP1 SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_dt1d.py; python tools/validate_all_configs.py
python -m pytest -q tests/test_proposal_contract.py tests/test_dense_manifest.py tests/test_dense_direct_configs.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch; assert torch.cuda.is_available(); print('GPU count=',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); zs=list(Path('/kaggle/input').rglob(f'D06_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_D06';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'dense_D06'))
 if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

# Standalone dense-data bootstrap. DRIVE is the only licensed/manual-access exception:
# attach an authorized DRIVE folder as a Kaggle Input containing training/images,
# training/1st_manual, test/images, and test/1st_manual.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
cfg=yaml.safe_load((Path('configs/dense/experiments')/(os.environ['TARGET']+'.yaml')).read_text())
root=Path(os.environ['DATA_ROOT']); root.mkdir(parents=True,exist_ok=True); ds=cfg['dataset']; pipe=cfg['pipeline']; print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE'; req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input'); found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent; rr=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in rr): found=b; break
        if found is None: raise RuntimeError('DRIVE requires an authorized Kaggle Input with the official training/test image+mask folders.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip'; urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc); shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else: raise RuntimeError(ds)
if pipe=='vit_b16_dense': models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
else: raise RuntimeError(pipe)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --target "$TARGET" --methods "$METHODS" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cpu --plan-only --download
run_group() { local group="$1" dev="$2" log="$3"; [[ -z "$group" ]] && return 0; python tools/run_dense_paper.py --target "$TARGET" --methods "$group" --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device "$dev" --download --skip-if-complete >"$log" 2>&1; }
export -f run_group
set +e; PIDS=(); NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( NGPU>=2 )); then [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; [[ -z "$GROUP1" ]] || { setsid bash -c 'run_group "$GROUP1" cuda:1 "$OUTPUT_ROOT/gpu1.log"' & PIDS+=("$!"); }; else [[ -z "$GROUP0" ]] || { setsid bash -c 'run_group "$GROUP0" cuda:0 "$OUTPUT_ROOT/gpu0.log"' & PIDS+=("$!"); }; fi
RUN_RC=0
while ((${#PIDS[@]})); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; ((alive==0)) && break; if (( $(date +%s)>=DEADLINE_EPOCH )); then for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 20; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; RUN_RC=124; break; fi; sleep 10; done
for p in "${PIDS[@]}"; do wait "$p" || RUN_RC=$?; done
if (( NGPU<2 )) && [[ -n "$GROUP1" ]] && (( $(date +%s)<DEADLINE_EPOCH )); then setsid bash -c 'run_group "$GROUP1" cuda:0 "$OUTPUT_ROOT/gpu1.log"' & p=$!; while kill -0 "$p" 2>/dev/null; do if (( $(date +%s)>=DEADLINE_EPOCH )); then kill -TERM -- "-$p" 2>/dev/null || true; sleep 20; kill -KILL -- "-$p" 2>/dev/null || true; RUN_RC=124; break; fi; sleep 10; done; wait "$p" || RUN_RC=$?; fi
set -e
python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" || true
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);t=os.environ['TARGET'];ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x];miss=[]
for m in ms:
 for s in ss:
  p=r/t/m/f'seed_{s}'
  if not any((p/x).is_file() for x in ('metrics.json','summary.json','test_summary.json','run_summary.json')): miss.append([m,s])
status={'session':os.environ['SESSION_ID'],'family':'dense','target':t,'methods':ms,'seeds':ss,'complete':not miss,'missing':miss,'source_commit':os.environ.get('SOURCE_COMMIT')}; (r/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS
pack_results; trap - EXIT; echo "DONE $SESSION_ID (RUN_RC=$RUN_RC). Attach its ZIP and rerun same cell if incomplete."; exit 0

```

## 32 — V01 — NEW result: VTAB-DTD / ViT-B16 / BS32

Target: `vtab-dtd` | Seeds: `tune42 -> 0,1,2` | Methods: `dt1d,vpt,pfeiffer,full,linear`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# NEW result: VTAB-DTD / ViT-B16 / BS32
SESSION_ID="V01"; DATASET="vtab-dtd"; BATCH_SIZE="32"; VPT_TOKENS="10"; METHODS="dt1d,vpt,pfeiffer,full,linear"; RESULT_MODE="table"; FINAL_SEEDS="0,1,2"
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
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);zs=list(Path('/kaggle/input').rglob(f'V01_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_V01';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'vit_V01'))
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

```

## 33 — V02 — NEW result: VTAB-EuroSAT / ViT-B16 / BS32

Target: `vtab-eurosat` | Seeds: `tune42 -> 0,1,2` | Methods: `dt1d,vpt,pfeiffer,full,linear`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# NEW result: VTAB-EuroSAT / ViT-B16 / BS32
SESSION_ID="V02"; DATASET="vtab-eurosat"; BATCH_SIZE="32"; VPT_TOKENS="10"; METHODS="dt1d,vpt,pfeiffer,full,linear"; RESULT_MODE="table"; FINAL_SEEDS="0,1,2"
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
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);zs=list(Path('/kaggle/input').rglob(f'V02_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_V02';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'vit_V02'))
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

```

## 34 — V03 — Manuscript ViT result: Flowers102 / ViT-B16 / BS16

Target: `flowers102` | Seeds: `tune42 -> 0,1,2` | Methods: `dt1d,vpt,pfeiffer,full,linear`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Manuscript ViT result: Flowers102 / ViT-B16 / BS16
SESSION_ID="V03"; DATASET="flowers102"; BATCH_SIZE="16"; VPT_TOKENS="5"; METHODS="dt1d,vpt,pfeiffer,full,linear"; RESULT_MODE="table"; FINAL_SEEDS="0,1,2"
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
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);zs=list(Path('/kaggle/input').rglob(f'V03_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_V03';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'vit_V03'))
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

```

## 35 — V04 — Manuscript ViT result: Flowers102 / ViT-B16 / BS32

Target: `flowers102` | Seeds: `tune42 -> 0,1,2` | Methods: `dt1d,vpt,pfeiffer,full,linear`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE KAGGLE SESSION
# Manuscript ViT result: Flowers102 / ViT-B16 / BS32
SESSION_ID="V04"; DATASET="flowers102"; BATCH_SIZE="32"; VPT_TOKENS="5"; METHODS="dt1d,vpt,pfeiffer,full,linear"; RESULT_MODE="table"; FINAL_SEEDS="0,1,2"
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
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);zs=list(Path('/kaggle/input').rglob(f'V04_results.zip')) if Path('/kaggle/input').exists() else []
if zs:
 z=max(zs,key=lambda p:p.stat().st_mtime);tmp=Path('/kaggle/working')/f'_restore_V04';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);found=list(tmp.rglob(f'vit_V04'))
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

```

## 36 — V05 — Manuscript ViT result: VTAB-Caltech101 / ViT-B16 / BS32

Target: `vtab-caltech101` | Seeds: `tune42 -> 0,1,2` | Methods: `dt1d,vpt,pfeiffer,full,linear`

```bash
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

```

## Static Figures 2–3

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
REPO_URL="${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"; W=/kaggle/working; R="$W/whc-dt1d-static"; O="$W/static_figures"
rm -rf "$R" "$O"; git clone --depth 1 "$REPO_URL" "$R"; cd "$R"; if [[ -n "$REPO_COMMIT" ]]; then git fetch --depth 1 origin "$REPO_COMMIT"; git checkout --detach "$REPO_COMMIT"; fi
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt; mkdir -p "$O"; python tools/plot_figure_02_spectral.py --output "$O/figure_02.png"; python tools/plot_figure_03_architecture.py --output "$O/figure_03.png"; cd "$W"; zip -qr static_figures_results.zip static_figures; ls -lh static_figures_results.zip

```

## Final merge/aggregation

```bash
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

```
