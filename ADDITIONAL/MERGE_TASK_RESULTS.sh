#!/usr/bin/env bash
set -Eeuo pipefail

# Final v10 merger for Kaggle.
# Upload both:
#   (a) already-completed legacy *_results.zip files you want to retain, and
#   (b) every v10 <TASK_ID>_results.zip produced by additional_runs/*_tasks.
# The script merges raw seed results, audits v6 task completeness/provenance,
# aggregates the retained manuscript targets where possible, and never fills
# missing values with legacy guesses.

W=/kaggle/working
INPUT=/kaggle/input
M="$W/V10_FINAL_MERGED"
OUTZIP="$W/V10_FINAL_ALL_RESULTS.zip"
REPO="$W/v10-merge-whc-dt1d"
rm -rf "$M" "$OUTZIP" "$REPO"
mkdir -p "$M/extracted" "$M/cnn_raw" "$M/dense_raw" "$M/vit_csv" "$M/task_metadata" "$M/final_tables"

python - <<'PY'
from pathlib import Path
import csv, json, shutil, zipfile

inp=Path('/kaggle/input'); M=Path('/kaggle/working/V10_FINAL_MERGED'); ex=M/'extracted'
zips=sorted(inp.rglob('*_results.zip')) if inp.exists() else []
print('ZIP count',len(zips))
for i,z in enumerate(zips):
    d=ex/f'{i:03d}_{z.stem}'; d.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(z) as f: f.extractall(d)
        print('EXTRACT',z)
    except Exception as e:
        print('WARN bad zip',z,e)

def copy_file_checked(src: Path, dst: Path, *, prefer_new=False):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if dst.read_bytes()==src.read_bytes(): return
        if prefer_new:
            shutil.copy2(src,dst); return
        raise RuntimeError(f'Conflicting duplicate result file: {dst}')
    shutil.copy2(src,dst)

def merge_tree(src: Path, dst: Path, *, prefer_new=False):
    for p in src.rglob('*'):
        if p.is_file(): copy_file_checked(p,dst/p.relative_to(src),prefer_new=prefer_new)

# 1) Preserve old result layouts first.
for p in ex.rglob('run_*'):
    if not p.is_dir(): continue
    for target in p.iterdir():
        if not target.is_dir() or target.name in {'aggregated','selection'}: continue
        for method in target.iterdir():
            if method.is_dir(): merge_tree(method,M/'cnn_raw'/target.name/method.name)
for p in ex.rglob('dense_*'):
    if not p.is_dir(): continue
    for target in p.iterdir():
        if not target.is_dir() or target.name=='aggregated': continue
        for method in target.iterdir():
            if method.is_dir(): merge_tree(method,M/'dense_raw'/target.name/method.name)
for p in ex.rglob('*fair_three_seed.csv'):
    dst=M/'vit_csv'/p.name
    if dst.exists() and dst.read_bytes()!=p.read_bytes(): dst=M/'vit_csv'/(p.parent.parent.name+'__'+p.name)
    shutil.copy2(p,dst)

# 2) Overlay v10 task results and audit provenance/completeness.
status_rows=[]; seen={}
for r in ex.rglob('progress_*'):
    if not r.is_dir() or not (r/'TASK_SPEC.json').is_file(): continue
    spec=json.loads((r/'TASK_SPEC.json').read_text())
    family=spec.get('family',''); target=spec.get('target',''); method=spec.get('method','')
    key=(family,target,method); fp=(spec.get('git_commit'),spec.get('proposal_fingerprint_sha256'))
    if key in seen and seen[key]!=fp: raise RuntimeError(f'Mixed commit/proposal for {key}: {seen[key]} vs {fp}')
    seen[key]=fp
    if (r/'results').is_dir(): merge_tree(r/'results',M/(family+'_raw'),prefer_new=True)
    task_id=r.name.removeprefix('progress_'); md=M/'task_metadata'/task_id
    if md.exists(): shutil.rmtree(md)
    md.mkdir(parents=True)
    for name in ['TASK_SPEC.json','TASK_STATUS.json','SESSION_PLAN.json','FINAL_SESSION_PLAN.json','runtime_calibration.json','experiment_config.yaml','RUNTIME_REFUSAL.json']:
        q=r/name
        if q.is_file(): shutil.copy2(q,md/name)
    st=json.loads((r/'TASK_STATUS.json').read_text()) if (r/'TASK_STATUS.json').is_file() else {}
    status_rows.append({
        'task_id':task_id,'family':family,'target':target,'method':method,'complete':bool(st.get('complete',False)),
        'git_commit':spec.get('git_commit',''),'proposal_fingerprint_sha256':spec.get('proposal_fingerprint_sha256',''),
        'tuning_missing':json.dumps(st.get('tuning_missing',[])),'missing_seeds':json.dumps(st.get('missing_seeds',[])),
        'final_missing_seeds':json.dumps(st.get('final_missing_seeds',[])),
    })
status_rows.sort(key=lambda x:x['task_id'])
fields=['task_id','family','target','method','complete','git_commit','proposal_fingerprint_sha256','tuning_missing','missing_seeds','final_missing_seeds']
with (M/'TASK_STATUS_SUMMARY.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(status_rows)
incomplete=[x for x in status_rows if not x['complete']]
(M/'INCOMPLETE_V10_TASKS.txt').write_text('\n'.join(x['task_id'] for x in incomplete)+('\n' if incomplete else ''),encoding='utf8')
summary={'uploaded_result_zips':len(zips),'v10_task_count':len(status_rows),'v10_complete':len(status_rows)-len(incomplete),'v10_incomplete':len(incomplete),'cnn_targets':sorted(x.name for x in (M/'cnn_raw').iterdir()),'dense_targets':sorted(x.name for x in (M/'dense_raw').iterdir()),'vit_csv_count':len(list((M/'vit_csv').glob('*.csv')))}
(M/'MERGE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY

# Aggregate with the same repository implementation. Pin the same commit used by
# the tasks when DT1D_CNN_COMMIT is supplied.
git clone --depth 1 "${DT1D_CNN_REPO_URL:-https://github.com/tydeptrai21042004/whc-dt1d.git}" "$REPO"
if [[ -n "${DT1D_CNN_COMMIT:-}" ]]; then
  git -C "$REPO" fetch --depth 1 origin "$DT1D_CNN_COMMIT"
  git -C "$REPO" checkout --detach "$DT1D_CNN_COMMIT"
fi
python -m pip install -q --upgrade-strategy only-if-needed -r "$REPO/requirements-kaggle.txt"

cd "$REPO"
for T in table_03 table_04 table_05 table_12 table_13 table_14_15 table_18_19 v10_n01_stl10_efficientnet_b0_10 v10_n02_flowers17_mobilenet_v3_small_10 v10_n03_stl10_densenet121_100 v10_n04_flowers17_resnet18_100 ablation_dtd_resnet18 ablation_flowers102_resnet50; do
  [[ -d "$M/cnn_raw/$T" ]] || continue
  mkdir -p "$M/final_tables/$T"
  python tools/aggregate_cnn_paper.py --root "$M/cnn_raw" --target "$T" --output-dir "$M/final_tables/$T" --require-seeds 0,1,2 --allow-incomplete || true
done
if [[ -d "$M/dense_raw" ]]; then
  python tools/aggregate_dense_results.py --input-root "$M/dense_raw" --output-dir "$M/final_tables/dense" --require-seeds 0,1,2 || true
fi

cd "$W"
zip -qr "$OUTZIP" V10_FINAL_MERGED
ls -lh "$OUTZIP"
echo "Final merged archive: $OUTZIP"
echo "IMPORTANT: inspect $M/INCOMPLETE_V10_TASKS.txt before updating manuscript numbers."
