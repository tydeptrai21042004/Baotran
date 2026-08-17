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
