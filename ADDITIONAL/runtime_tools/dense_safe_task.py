#!/usr/bin/env python3
"""Resumable dense-prediction Kaggle task with pre-flight session planning.

No watchdog timeout is used. The shell records session start before setup; one
real epoch calibrates the exact target; the runner keeps a conservative 10 h
planning ceiling with a 45 min internal reserve and rechecks before each wave.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, copy, json, math, shutil, subprocess, sys, time
from pathlib import Path
import yaml

def jread(p:Path): return json.loads(p.read_text()) if p.is_file() else {}
def jwrite(p:Path,o): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,default=str)+'\n')
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-root',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--method',required=True); ap.add_argument('--data-path',type=Path,required=True); ap.add_argument('--progress-root',type=Path,required=True)
    ap.add_argument('--safe-minutes',type=float,default=600.0); ap.add_argument('--session-start-epoch',type=float,default=None); ap.add_argument('--safety-factor',type=float,default=1.60); ap.add_argument('--reserve-minutes',type=float,default=45.0); ap.add_argument('--max-gpus',type=int,default=2); ns=ap.parse_args()
    session_start=float(ns.session_start_epoch or time.time()); deadline=session_start+ns.safe_minutes*60; reserve=ns.reserve_minutes*60
    def elapsed(): return max(0,time.time()-session_start)
    def remaining(): return max(0,deadline-time.time()-reserve)
    repo=ns.repo_root.resolve(); sys.path.insert(0,str(repo)); from tools import run_dense_from_config as rd; from proposal_contract import runtime_metadata
    cp=ns.config if ns.config.is_absolute() else repo/ns.config; cfg=rd.load_config(cp)
    if ns.method not in cfg['methods']: raise SystemExit(f'method {ns.method} missing')
    seeds=[int(x) for x in cfg['seeds']]; target=cfg['experiment_id']; epochs=int(cfg['epochs']); root=ns.progress_root.resolve(); root.mkdir(parents=True,exist_ok=True)
    current={'schema_version':2,'family':'dense','target':target,'method':ns.method,'epochs':epochs,'seeds':seeds,'session_limit_minutes':ns.safe_minutes,'reserve_minutes':ns.reserve_minutes,'safety_factor':ns.safety_factor,'session_policy':'preflight_only_no_watchdog_no_timeout',**runtime_metadata(repo)}
    old=jread(root/'TASK_SPEC.json')
    if old:
        for k in ('git_commit','proposal_fingerprint_sha256','target','method'):
            if old.get(k)!=current.get(k): raise SystemExit(f'REFUSING TO MIX RESUMED RESULTS: {k} changed')
    shutil.copy2(cp,root/'experiment_config.yaml'); jwrite(root/'TASK_SPEC.json',current)
    try:
        import torch; ng=torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception: ng=0
    if ng<1: raise SystemExit('Kaggle GPU required')
    ng=max(1,min(ns.max_gpus,ng)); devs=[f'cuda:{i}' for i in range(ng)]
    calp=root/'runtime_calibration.json'; cal=jread(calp)
    if not cal.get('complete'):
        if remaining()<15*60: raise SystemExit('REFUSING: insufficient planning time for calibration')
        tc=copy.deepcopy(cfg); tc['epochs']=1; tc['common_args']=copy.deepcopy(tc['common_args']); tc['common_args']['profile_latency']=False
        tcfg=root/'_calibration_config.yaml'; tcfg.write_text(yaml.safe_dump(tc,sort_keys=False)); out=root/'_calibration'; shutil.rmtree(out,ignore_errors=True)
        cmd=[sys.executable,str(repo/'tools'/'run_dense_from_config.py'),str(tcfg),'--method',ns.method,'--seed',str(seeds[0]),'--data-path',str(ns.data_path),'--output-dir',str(out),'--device',devs[0],'--download','--profile-latency','false']
        t0=time.time(); rc=subprocess.run(cmd,cwd=repo).returncode; wall=time.time()-t0; sm=jread(out/'summary.json'); train=sm.get('total_train_time_sec') or sm.get('mean_epoch_time_sec') or wall
        if rc!=0 or not isinstance(train,(int,float)) or train<=0: raise SystemExit(f'calibration failed rc={rc}')
        cal={'complete':True,'measured_wall_seconds':wall,'measured_train_loop_seconds_per_epoch':float(train),'gpu_count':ng,'gpu_names':[]}
        try: cal['gpu_names']=[torch.cuda.get_device_name(i) for i in range(ng)]
        except Exception: pass
        jwrite(calp,cal); shutil.rmtree(out,ignore_errors=True); tcfg.unlink(missing_ok=True)
    train=float(cal['measured_train_loop_seconds_per_epoch']); epoch_upper=max(train,min(float(cal['measured_wall_seconds']),train*1.35)); unit_upper=epoch_upper*epochs*ns.safety_factor+240
    jwrite(root/'PRE_FLIGHT_ESTIMATE.json',{'session_elapsed_seconds_before_planning':elapsed(),'remaining_usable_seconds':remaining(),'calibrated_epoch_seconds':epoch_upper,'unit_upper_seconds':unit_upper,'calibration':cal})
    if unit_upper>remaining():
        jwrite(root/'RUNTIME_REFUSAL.json',{'reason':'one full dense seed predicted not to fit','unit_upper_seconds':unit_upper,'remaining_usable_seconds':remaining()}); raise SystemExit('REFUSING BEFORE FULL RUN: predicted single seed does not fit safe window')
    rr=root/'results'/target/ns.method
    pending=[s for s in seeds if not (rr/f'seed_{s}'/'summary.json').is_file()]
    obs=[]
    for s in seeds:
        st=jread(rr/f'seed_{s}'/'v9_elapsed.json');
        if isinstance(st.get('elapsed_seconds'),(int,float)): obs.append(float(st['elapsed_seconds']))
    dyn=max([unit_upper]+[x*1.20 for x in obs if x>0]); maxwaves=int(math.floor(remaining()/max(dyn,1))); plan=pending[:maxwaves*ng]
    jwrite(root/'SESSION_PLAN.json',{'pending_before':pending,'planned_seeds':plan,'dynamic_unit_upper_seconds':dyn,'session_elapsed_seconds':elapsed(),'remaining_usable_seconds':remaining(),'rule':'recheck before every wave; no timeout'})
    def run(seed,dev):
        out=rr/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
        if (out/'summary.json').is_file(): return {'seed':seed,'status':'already_complete','elapsed_seconds':0}
        cmd=[sys.executable,str(repo/'tools'/'run_dense_from_config.py'),str(cp),'--method',ns.method,'--seed',str(seed),'--data-path',str(ns.data_path),'--output-dir',str(out),'--device',dev,'--download','--skip-if-complete']
        t0=time.time(); rc=subprocess.run(cmd,cwd=repo).returncode; dur=time.time()-t0
        if rc!=0 or not (out/'summary.json').is_file(): raise RuntimeError(f'dense seed {seed} failed rc={rc}')
        jwrite(out/'v9_elapsed.json',{'elapsed_seconds':dur})
        for p in out.glob('*.pth'): p.unlink(missing_ok=True)
        for p in out.glob('*.pt'): p.unlink(missing_ok=True)
        return {'seed':seed,'status':'complete','elapsed_seconds':dur}
    for off in range(0,len(plan),ng):
        wave=plan[off:off+ng]; ob=[]
        for s in seeds:
            st=jread(rr/f'seed_{s}'/'v9_elapsed.json');
            if isinstance(st.get('elapsed_seconds'),(int,float)): ob.append(float(st['elapsed_seconds']))
        wave_upper=max([unit_upper]+[x*1.20 for x in ob if x>0])
        if remaining()<wave_upper:
            print(f'STOP_BEFORE_WAVE remaining={remaining()/60:.1f}min predicted_wave={wave_upper/60:.1f}min'); break
        with cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
            fs=[ex.submit(run,s,devs[i]) for i,s in enumerate(wave)]
            for f in cf.as_completed(fs): print('DENSE_RESULT',json.dumps(f.result()))
    missing=[s for s in seeds if not (rr/f'seed_{s}'/'summary.json').is_file()]; complete=not missing
    status={'complete':complete,'family':'dense','target':target,'method':ns.method,'missing_seeds':missing,'session_elapsed_minutes':elapsed()/60,'session_limit_minutes':ns.safe_minutes,'instruction':'If incomplete, upload ZIP to fresh session and rerun. No process was intentionally killed.'}; jwrite(root/'TASK_STATUS.json',status); print(json.dumps(status,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
