#!/usr/bin/env python3
"""Resumable, pre-flight planned CNN task for Kaggle.

Scientific protocol is preserved across sessions:
  * the declared common LR grid is evaluated for seeds 0/1/2;
  * one shared LR is selected from mean validation performance;
  * final seeds are retrained with that LR and tested once.

Runtime policy:
  * there is NO 11h50 watchdog and NO training-process timeout;
  * the shell records the Kaggle-session start before clone/install/download;
  * one real epoch calibrates the exact dataset/backbone/method;
  * a conservative upper bound is computed before a full LR/seed unit starts;
  * the default session planning ceiling is 600 min (10 h), with 45 min kept
    unused inside that ceiling, leaving additional margin before Kaggle's 12 h;
  * remaining time is rechecked before every GPU wave. If the next wave is not
    predicted to fit, the runner stops normally and writes a resumable ZIP.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, math, shutil, subprocess, sys, time
from pathlib import Path


def jwrite(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")

def jread(path: Path):
    if not path.is_file(): return {}
    return json.loads(path.read_text(encoding="utf-8"))

def finite(x) -> bool:
    try: return math.isfinite(float(x))
    except Exception: return False

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--method',required=True)
    ap.add_argument('--data-path',type=Path,required=True)
    ap.add_argument('--progress-root',type=Path,required=True)
    ap.add_argument('--safe-minutes',type=float,default=600.0,help='Total planning ceiling from shell session start; default 10h.')
    ap.add_argument('--session-start-epoch',type=float,default=None,help='Unix epoch recorded before clone/install/download.')
    ap.add_argument('--safety-factor',type=float,default=1.60)
    ap.add_argument('--reserve-minutes',type=float,default=45.0)
    ap.add_argument('--max-gpus',type=int,default=2)
    ns=ap.parse_args()
    # Hard safety cap: even direct/manual invocation cannot plan beyond 10h.
    ns.safe_minutes = min(float(ns.safe_minutes), 600.0)
    session_start=float(ns.session_start_epoch or time.time())
    repo=ns.repo_root.resolve(); sys.path.insert(0,str(repo))
    from proposal_contract import runtime_metadata
    from tools import run_experiment as rx
    config_path=ns.config if ns.config.is_absolute() else repo/ns.config
    cfg=rx.load_yaml(config_path); method=ns.method
    if method not in cfg['methods']: raise SystemExit(f'Method {method!r} not in {config_path}')
    seeds=[int(s) for s in cfg['seeds']]; lrs=[float(x) for x in cfg['fairness']['lr_candidates']]
    epochs=int(cfg['epochs']); target=str(cfg['experiment_id'])
    root=ns.progress_root.resolve(); root.mkdir(parents=True,exist_ok=True)
    deadline=session_start+float(ns.safe_minutes)*60.0
    reserve=float(ns.reserve_minutes)*60.0
    def elapsed(): return max(0.0,time.time()-session_start)
    def remaining(): return max(0.0,deadline-time.time()-reserve)
    task_meta={'schema_version':2,'family':'cnn','target':target,'method':method,'seeds':seeds,'lr_candidates':lrs,'epochs':epochs,
               'session_limit_minutes':ns.safe_minutes,'reserve_minutes':ns.reserve_minutes,'safety_factor':ns.safety_factor,
               'session_policy':'preflight_only_no_watchdog_no_timeout',**runtime_metadata(repo)}
    old=jread(root/'TASK_SPEC.json')
    if old:
        for key in ('git_commit','proposal_fingerprint_sha256','target','method'):
            if old.get(key)!=task_meta.get(key): raise SystemExit(f'REFUSING TO MIX RESUMED RESULTS: {key} changed')
    jwrite(root/'TASK_SPEC.json',task_meta); shutil.copy2(config_path,root/'experiment_config.yaml')
    try:
        import torch; ng=torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception: ng=0
    if ng<1: raise SystemExit('Kaggle GPU required')
    ng=max(1,min(int(ns.max_gpus),ng)); devices=[f'cuda:{i}' for i in range(ng)]

    # exact one-epoch pre-flight calibration
    calp=root/'runtime_calibration.json'; cal=jread(calp)
    if not cal.get('complete'):
        if remaining() < 15*60: raise SystemExit('REFUSING: insufficient planning time even for calibration')
        cdir=root/'_calibration'; shutil.rmtree(cdir,ignore_errors=True); cdir.mkdir(parents=True,exist_ok=True)
        args=rx.base_args(cfg,method,seeds[0],ns.data_path,devices[0]); args.update({'epochs':1,'lr':lrs[len(lrs)//2],
              'final_test':False,'eval':False,'save_ckpt':False,'save_history':True,'profile_efficiency':False,'measure_eval_latency':False})
        rx.validate_args(args,cdir); cmd=rx.build_command(args,cdir)
        t0=time.time(); rc=rx.run_process(cmd,repo,cdir/'stdout.log',dry_run=False); wall=time.time()-t0
        conv=jread(cdir/'convergence_summary.json'); train=conv.get('total_train_time_sec') or conv.get('mean_epoch_time_sec') or wall
        if rc!=0 or not finite(train) or float(train)<=0: raise SystemExit(f'Calibration failed rc={rc}')
        cal={'complete':True,'seed':seeds[0],'lr':lrs[len(lrs)//2],'measured_wall_seconds':wall,
             'measured_train_loop_seconds_per_epoch':float(train),'gpu_count':ng,'gpu_names':[]}
        try: cal['gpu_names']=[torch.cuda.get_device_name(i) for i in range(ng)]
        except Exception: pass
        jwrite(calp,cal); shutil.rmtree(cdir,ignore_errors=True)

    train_epoch=float(cal['measured_train_loop_seconds_per_epoch'])
    # Include validation/data-loader overhead conservatively without multiplying all one-time startup cost.
    calibrated_epoch=max(train_epoch, min(float(cal['measured_wall_seconds']), train_epoch*1.35))
    candidate_upper=calibrated_epoch*epochs*float(ns.safety_factor)+180.0
    final_upper=candidate_upper*1.18+180.0
    jwrite(root/'PRE_FLIGHT_ESTIMATE.json',{'session_elapsed_seconds_before_planning':elapsed(),'remaining_usable_seconds':remaining(),
           'calibrated_epoch_seconds':calibrated_epoch,'candidate_upper_seconds':candidate_upper,'final_upper_seconds':final_upper,
           'session_limit_minutes':ns.safe_minutes,'reserve_minutes':ns.reserve_minutes,'safety_factor':ns.safety_factor,'calibration':cal})
    if candidate_upper>remaining():
        jwrite(root/'RUNTIME_REFUSAL.json',{'reason':'one full validation LR/seed unit is predicted not to fit this fresh session',
               'candidate_upper_seconds':candidate_upper,'remaining_usable_seconds':remaining(),
               'action':'Do not start the full unit. Use a faster/smaller target or implement checkpoint-resume for within-run epoch chunking.'})
        raise SystemExit(f'REFUSING BEFORE FULL TRAINING: one unit upper estimate {candidate_upper/3600:.2f}h exceeds remaining safe window {remaining()/3600:.2f}h')

    tune=root/'tuning'/target/method
    def cp(seed,lr): return tune/f'seed_{seed}'/f'lr_{rx.lr_slug(lr)}'/'candidate.json'
    def done(seed,lr):
        d=jread(cp(seed,lr)); return int(d.get('return_code',1))==0 and finite(d.get('best_val_acc1'))
    def run_candidate(unit,dev):
        seed,lr=unit; out=cp(seed,lr).parent
        if done(seed,lr): return {'seed':seed,'lr':lr,'status':'already_complete','elapsed_seconds':0}
        shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True)
        args=rx.base_args(cfg,method,seed,ns.data_path,dev); args.update({'lr':float(lr),'final_test':False,'eval':False,
             'save_ckpt':False,'save_history':True,'profile_efficiency':False,'measure_eval_latency':False})
        rx.validate_args(args,out); cmd=rx.build_command(args,out); t0=time.time(); rc=rx.run_process(cmd,repo,out/'stdout.log',dry_run=False); dur=time.time()-t0
        conv=jread(out/'convergence_summary.json'); row={'seed':seed,'lr':float(lr),'best_val_acc1':conv.get('best_val_acc1'),
             'best_epoch':conv.get('best_epoch'),'return_code':int(rc),'elapsed_seconds':dur,'test_evaluated':False}; jwrite(cp(seed,lr),row)
        if rc!=0 or not finite(row['best_val_acc1']): raise RuntimeError(f'candidate failed seed={seed} lr={lr} rc={rc}')
        for p in out.glob('*.pth'): p.unlink(missing_ok=True)
        return row

    pending=[(s,lr) for s in seeds for lr in lrs if not done(s,lr)]
    observed=[float(jread(cp(s,lr)).get('elapsed_seconds',0)) for s in seeds for lr in lrs if done(s,lr)]
    dynamic_candidate=max([candidate_upper]+[x*1.20 for x in observed if x>0])
    planned=[]; rem=remaining(); maxwaves=int(math.floor(rem/max(dynamic_candidate,1.0))); planned=pending[:maxwaves*ng]
    jwrite(root/'SESSION_PLAN.json',{'phase':'tuning' if pending else 'selection_or_final','pending_before':[{'seed':s,'lr':lr} for s,lr in pending],
           'planned_units':[{'seed':s,'lr':lr} for s,lr in planned],'dynamic_candidate_upper_seconds':dynamic_candidate,
           'predicted_training_wall_seconds_upper':math.ceil(len(planned)/ng)*dynamic_candidate if planned else 0,
           'session_elapsed_seconds':elapsed(),'remaining_usable_seconds':remaining(),'gpu_count':ng,
           'rule':'recheck remaining time before every wave; never start a predicted-over-budget wave'})
    for off in range(0,len(planned),ng):
        wave=planned[off:off+ng]
        # Dynamic no-start gate: account for actual time consumed by previous waves.
        observed_now=[float(jread(cp(s,lr)).get('elapsed_seconds',0)) for s in seeds for lr in lrs if done(s,lr)]
        wave_upper=max([candidate_upper]+[x*1.20 for x in observed_now if x>0])
        if remaining() < wave_upper:
            print(f'STOP_BEFORE_WAVE remaining={remaining()/60:.1f}min predicted_wave={wave_upper/60:.1f}min')
            break
        with cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
            fs=[ex.submit(run_candidate,u,devices[i]) for i,u in enumerate(wave)]
            for f in cf.as_completed(fs): print('TUNE_RESULT',json.dumps(f.result(),default=str))

    pending_after=[(s,lr) for s in seeds for lr in lrs if not done(s,lr)]
    selp=root/'selection'/target/method/'lr_selection_summary.json'
    if not pending_after and not selp.is_file():
        rows=[]
        for s in seeds:
            for lr in lrs:
                d=jread(cp(s,lr)); rows.append({'seed':s,'lr':lr,'best_val_acc1':float(d['best_val_acc1']),'best_epoch':int(d.get('best_epoch',-1)),'return_code':int(d['return_code'])})
        sel=rx.select_shared_lr(rows,seeds=seeds,lr_candidates=lrs); sel.update({'schema_version':1,'experiment_id':target,'method':method,
            'test_used_for_selection':False,'final_policy':'retrain selected LR independently for seeds 0/1/2, then test once at best validation checkpoint'})
        jwrite(selp,sel)
    sel=jread(selp); finalroot=root/'results'/target/method
    if sel:
        selected=float(sel['selected_lr']); pending_final=[s for s in seeds if not (finalroot/f'seed_{s}'/'test_summary.json').is_file()]
        final_observed=[]
        for s in seeds:
            st=jread(finalroot/f'seed_{s}'/'run_status.json')
            if finite(st.get('elapsed_seconds')): final_observed.append(float(st['elapsed_seconds']))
        dyn_final=max([final_upper]+[x*1.20 for x in final_observed if x>0])
        maxwaves=int(math.floor(remaining()/max(dyn_final,1.0))); final_plan=pending_final[:maxwaves*ng]
        jwrite(root/'FINAL_SESSION_PLAN.json',{'selected_lr':selected,'pending_before':pending_final,'planned_seeds':final_plan,
             'dynamic_final_upper_seconds':dyn_final,'session_elapsed_seconds':elapsed(),'remaining_usable_seconds':remaining()})
        def run_final(seed,dev):
            out=finalroot/f'seed_{seed}'
            if (out/'test_summary.json').is_file(): return {'seed':seed,'status':'already_complete','elapsed_seconds':0}
            shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True,exist_ok=True)
            args=rx.base_args(cfg,method,seed,ns.data_path,dev); args.update({'lr':selected,'eval':False,'final_test':True,'save_ckpt':True,'save_history':True,
                 'profile_efficiency':bool(cfg['common_args'].get('profile_efficiency',True)),'measure_eval_latency':bool(cfg['common_args'].get('measure_eval_latency',True))})
            rx.validate_args(args,out); cmd=rx.build_command(args,out); t0=time.time(); rc=rx.run_process(cmd,repo,out/'final_stdout.log',dry_run=False); dur=time.time()-t0
            test=jread(out/'test_summary.json'); conv=jread(out/'convergence_summary.json')
            if rc!=0 or not test: raise RuntimeError(f'final failed seed={seed} rc={rc}')
            test.update({'selected_lr':selected,'selected_lr_mean_val_acc1':sel.get('selected_mean_best_val_acc1'),'lr_selection_scope':'method_across_seeds','test_used_for_hyperparameter_selection':False}); jwrite(out/'test_summary.json',test)
            ms=cfg['methods'][method]; jwrite(out/'run_metadata.json',{'schema_version':4,'target':target,'kind':cfg.get('kind','comparison'),'manuscript_tables':cfg.get('manuscript_tables',[]),'manuscript_figures':cfg.get('manuscript_figures',[]),'method_preset':method,'method_label':ms.get('label',method),'variant':ms.get('variant'),'independent_seed':seed,'proposal':method=='dt1d',**runtime_metadata(repo)})
            jwrite(out/'resolved_config.json',{'schema_version':4,'source_config':str(config_path),'experiment_id':target,'method':method,'seed':seed,'fairness':cfg['fairness'],'selected_lr':selected,'args':args,**runtime_metadata(repo)})
            jwrite(out/'selection_summary.json',sel); jwrite(out/'run_status.json',{'return_code':0,'elapsed_seconds':dur}); jwrite(out/'args.json',args)
            for p in out.glob('checkpoint*.pth'): p.unlink(missing_ok=True)
            return {'seed':seed,'test_acc1':test.get('test_acc1_at_best_val',test.get('acc1')),'best_val':conv.get('best_val_acc1'),'elapsed_seconds':dur}
        for off in range(0,len(final_plan),ng):
            wave=final_plan[off:off+ng]
            obs=[]
            for s in seeds:
                st=jread(finalroot/f'seed_{s}'/'run_status.json')
                if finite(st.get('elapsed_seconds')): obs.append(float(st['elapsed_seconds']))
            wave_upper=max([final_upper]+[x*1.20 for x in obs if x>0])
            if remaining() < wave_upper:
                print(f'STOP_BEFORE_FINAL_WAVE remaining={remaining()/60:.1f}min predicted_wave={wave_upper/60:.1f}min'); break
            with cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
                fs=[ex.submit(run_final,s,devices[i]) for i,s in enumerate(wave)]
                for f in cf.as_completed(fs): print('FINAL_RESULT',json.dumps(f.result(),default=str))

    pending_after=[(s,lr) for s in seeds for lr in lrs if not done(s,lr)]; sel=jread(selp)
    final_missing=[s for s in seeds if not (finalroot/f'seed_{s}'/'test_summary.json').is_file()]
    complete=(not pending_after) and bool(sel) and (not final_missing)
    if complete:
        agg=root/'aggregated'/target; agg.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,str(repo/'tools'/'aggregate_cnn_paper.py'),'--root',str(root/'results'),'--target',target,'--output-dir',str(agg),'--require-seeds',','.join(map(str,seeds))],cwd=repo,check=True)
    status={'complete':complete,'family':'cnn','target':target,'method':method,'tuning_missing':[{'seed':s,'lr':lr} for s,lr in pending_after],
            'selection_complete':bool(sel),'selected_lr':sel.get('selected_lr') if sel else None,'final_missing_seeds':final_missing,
            'session_elapsed_minutes':elapsed()/60.0,'session_limit_minutes':ns.safe_minutes,
            'instruction':'If incomplete, upload this task ZIP to a fresh Kaggle session and rerun the exact same script. No process was intentionally killed.'}
    jwrite(root/'TASK_STATUS.json',status); print(json.dumps(status,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
