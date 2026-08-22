#!/usr/bin/env python3
"""Run exactly ONE CNN method in a Kaggle session, with resumable method-level progress.

Scientific protocol (unchanged):
  * evaluate the common LR grid for seeds 0/1/2;
  * select one shared LR from mean validation accuracy across seeds;
  * retrain final seeds independently with that selected LR;
  * evaluate test only in the final runs.

Runtime policy:
  * no training-process timeout/watchdog is used;
  * session time starts before clone/install/download;
  * one real epoch calibrates the exact dataset/backbone/method;
  * before EVERY GPU wave, estimate whether one complete wave can fit;
  * after a wave finishes, recompute the estimate and continue while safe;
  * stop normally before the safety budget is exhausted and save resumable state.

This fixes the previous scheduler bug where only one precomputed batch was executed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path


def jwrite(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def jread(path: Path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-path", type=Path, required=True)
    ap.add_argument("--progress-root", type=Path, required=True)
    ap.add_argument("--safe-minutes", type=float, default=690.0)
    ap.add_argument("--session-start-epoch", type=float, default=None)
    ap.add_argument("--safety-factor", type=float, default=1.55)
    ap.add_argument("--reserve-minutes", type=float, default=30.0)
    ap.add_argument("--max-gpus", type=int, default=2)
    ns = ap.parse_args()

    ns.safe_minutes = min(float(ns.safe_minutes), 690.0)
    session_start = float(ns.session_start_epoch or time.time())

    repo = ns.repo_root.resolve()
    sys.path.insert(0, str(repo))
    from proposal_contract import runtime_metadata
    from tools import run_experiment as rx

    config_path = ns.config if ns.config.is_absolute() else repo / ns.config
    cfg = rx.load_yaml(config_path)
    method = ns.method
    if method not in cfg["methods"]:
        raise SystemExit(f"Method {method!r} not in {config_path}")

    seeds = [int(s) for s in cfg["seeds"]]
    lrs = [float(x) for x in cfg["fairness"]["lr_candidates"]]
    epochs = int(cfg["epochs"])
    target = str(cfg["experiment_id"])

    root = ns.progress_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    deadline = session_start + float(ns.safe_minutes) * 60.0
    reserve = float(ns.reserve_minutes) * 60.0

    def elapsed() -> float:
        return max(0.0, time.time() - session_start)

    def remaining() -> float:
        return max(0.0, deadline - time.time() - reserve)

    task_meta = {
        "schema_version": 3,
        "family": "cnn",
        "target": target,
        "method": method,
        "seeds": seeds,
        "lr_candidates": lrs,
        "epochs": epochs,
        "session_limit_minutes": ns.safe_minutes,
        "reserve_minutes": ns.reserve_minutes,
        "safety_factor": ns.safety_factor,
        "session_policy": "one_method_dynamic_wave_replanning_no_watchdog_no_timeout",
        **runtime_metadata(repo),
    }

    old = jread(root / "TASK_SPEC.json")
    if old:
        for key in ("git_commit", "proposal_fingerprint_sha256", "target", "method"):
            if old.get(key) != task_meta.get(key):
                raise SystemExit(
                    f"REFUSING TO MIX RESUMED RESULTS: {key} changed "
                    f"({old.get(key)!r} -> {task_meta.get(key)!r})"
                )

    jwrite(root / "TASK_SPEC.json", task_meta)
    shutil.copy2(config_path, root / "experiment_config.yaml")

    try:
        import torch
        ng = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        ng = 0
    if ng < 1:
        raise SystemExit("Kaggle GPU required. Enable a GPU accelerator.")
    ng = max(1, min(int(ns.max_gpus), ng))
    devices = [f"cuda:{i}" for i in range(ng)]

    # 1) One-epoch calibration (once per method; reused after resume).
    calp = root / "runtime_calibration.json"
    cal = jread(calp)
    if not cal.get("complete"):
        if remaining() < 15 * 60:
            raise SystemExit("REFUSING: insufficient safe time even for calibration")
        cdir = root / "_calibration"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True, exist_ok=True)
        args = rx.base_args(cfg, method, seeds[0], ns.data_path, devices[0])
        args.update({
            "epochs": 1,
            "lr": lrs[len(lrs)//2],
            "final_test": False,
            "eval": False,
            "save_ckpt": False,
            "save_history": True,
            "profile_efficiency": False,
            "measure_eval_latency": False,
        })
        rx.validate_args(args, cdir)
        cmd = rx.build_command(args, cdir)
        t0 = time.time()
        rc = rx.run_process(cmd, repo, cdir / "stdout.log", dry_run=False)
        wall = time.time() - t0
        conv = jread(cdir / "convergence_summary.json")
        train = conv.get("total_train_time_sec") or conv.get("mean_epoch_time_sec") or wall
        if rc != 0 or not finite(train) or float(train) <= 0:
            raise SystemExit(f"Calibration failed rc={rc}")
        cal = {
            "complete": True,
            "seed": seeds[0],
            "lr": lrs[len(lrs)//2],
            "measured_wall_seconds": wall,
            "measured_train_loop_seconds_per_epoch": float(train),
            "gpu_count": ng,
            "gpu_names": [],
        }
        try:
            cal["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(ng)]
        except Exception:
            pass
        jwrite(calp, cal)
        shutil.rmtree(cdir, ignore_errors=True)

    train_epoch = float(cal["measured_train_loop_seconds_per_epoch"])
    calibrated_epoch = max(
        train_epoch,
        min(float(cal["measured_wall_seconds"]), train_epoch * 1.35),
    )
    candidate_upper = calibrated_epoch * epochs * float(ns.safety_factor) + 180.0
    final_upper = candidate_upper * 1.18 + 180.0
    jwrite(root / "PRE_FLIGHT_ESTIMATE.json", {
        "session_elapsed_seconds_before_planning": elapsed(),
        "remaining_usable_seconds": remaining(),
        "calibrated_epoch_seconds": calibrated_epoch,
        "candidate_upper_seconds": candidate_upper,
        "final_upper_seconds": final_upper,
        "session_limit_minutes": ns.safe_minutes,
        "reserve_minutes": ns.reserve_minutes,
        "safety_factor": ns.safety_factor,
        "calibration": cal,
    })

    # 2) LR tuning. Re-plan after EVERY wave; this is the key fix.
    tune = root / "tuning" / target / method

    def cp(seed, lr):
        return tune / f"seed_{seed}" / f"lr_{rx.lr_slug(lr)}" / "candidate.json"

    def done(seed, lr):
        d = jread(cp(seed, lr))
        return int(d.get("return_code", 1)) == 0 and finite(d.get("best_val_acc1"))

    def run_candidate(unit, dev):
        seed, lr = unit
        out = cp(seed, lr).parent
        if done(seed, lr):
            return {"seed": seed, "lr": lr, "status": "already_complete", "elapsed_seconds": 0}
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        args = rx.base_args(cfg, method, seed, ns.data_path, dev)
        args.update({
            "lr": float(lr),
            "final_test": False,
            "eval": False,
            "save_ckpt": False,
            "save_history": True,
            "profile_efficiency": False,
            "measure_eval_latency": False,
        })
        rx.validate_args(args, out)
        cmd = rx.build_command(args, out)
        t0 = time.time()
        rc = rx.run_process(cmd, repo, out / "stdout.log", dry_run=False)
        dur = time.time() - t0
        conv = jread(out / "convergence_summary.json")
        row = {
            "seed": seed,
            "lr": float(lr),
            "best_val_acc1": conv.get("best_val_acc1"),
            "best_epoch": conv.get("best_epoch"),
            "return_code": int(rc),
            "elapsed_seconds": dur,
            "test_evaluated": False,
        }
        jwrite(cp(seed, lr), row)
        if rc != 0 or not finite(row["best_val_acc1"]):
            raise RuntimeError(f"candidate failed seed={seed} lr={lr} rc={rc}")
        for p in out.glob("*.pth"):
            p.unlink(missing_ok=True)
        return row

    session_waves = []
    stop_reason = None
    while True:
        pending = [(s, lr) for s in seeds for lr in lrs if not done(s, lr)]
        if not pending:
            break
        observed = [
            float(jread(cp(s, lr)).get("elapsed_seconds", 0))
            for s in seeds for lr in lrs if done(s, lr)
        ]
        wave_upper = max([candidate_upper] + [x * 1.20 for x in observed if x > 0])
        if remaining() < wave_upper:
            stop_reason = "not_enough_safe_time_for_next_tuning_wave"
            print(
                f"STOP_BEFORE_TUNING_WAVE remaining={remaining()/60:.1f}min "
                f"predicted_wave={wave_upper/60:.1f}min",
                flush=True,
            )
            break
        wave = pending[:ng]
        session_waves.append({
            "phase": "tuning",
            "units": [{"seed": s, "lr": lr} for s, lr in wave],
            "started_elapsed_seconds": elapsed(),
            "remaining_before_seconds": remaining(),
            "predicted_wave_upper_seconds": wave_upper,
        })
        jwrite(root / "SESSION_PLAN.json", {
            "method": method,
            "gpu_count": ng,
            "waves": session_waves,
            "remaining_usable_seconds": remaining(),
            "policy": "re-plan after every completed GPU wave; never switch methods inside this session",
        })
        with cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
            fs = [ex.submit(run_candidate, unit, devices[i]) for i, unit in enumerate(wave)]
            for f in cf.as_completed(fs):
                print("TUNE_RESULT", json.dumps(f.result(), default=str), flush=True)

    # 3) Shared LR selection only after all tuning units complete.
    pending_after = [(s, lr) for s in seeds for lr in lrs if not done(s, lr)]
    selp = root / "selection" / target / method / "lr_selection_summary.json"
    if not pending_after and not selp.is_file():
        rows = []
        for s in seeds:
            for lr in lrs:
                d = jread(cp(s, lr))
                rows.append({
                    "seed": s,
                    "lr": lr,
                    "best_val_acc1": float(d["best_val_acc1"]),
                    "best_epoch": int(d.get("best_epoch", -1)),
                    "return_code": int(d["return_code"]),
                })
        sel = rx.select_shared_lr(rows, seeds=seeds, lr_candidates=lrs)
        sel.update({
            "schema_version": 1,
            "experiment_id": target,
            "method": method,
            "test_used_for_selection": False,
            "final_policy": "retrain selected LR independently for seeds 0/1/2, then test once at best validation checkpoint",
        })
        jwrite(selp, sel)

    sel = jread(selp)
    finalroot = root / "results" / target / method

    def run_final(seed, dev, selected, sel):
        out = finalroot / f"seed_{seed}"
        if (out / "test_summary.json").is_file():
            return {"seed": seed, "status": "already_complete", "elapsed_seconds": 0}
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        args = rx.base_args(cfg, method, seed, ns.data_path, dev)
        args.update({
            "lr": selected,
            "eval": False,
            "final_test": True,
            "save_ckpt": True,
            "save_history": True,
            "profile_efficiency": bool(cfg["common_args"].get("profile_efficiency", True)),
            "measure_eval_latency": bool(cfg["common_args"].get("measure_eval_latency", True)),
        })
        rx.validate_args(args, out)
        cmd = rx.build_command(args, out)
        t0 = time.time()
        rc = rx.run_process(cmd, repo, out / "final_stdout.log", dry_run=False)
        dur = time.time() - t0
        test = jread(out / "test_summary.json")
        conv = jread(out / "convergence_summary.json")
        if rc != 0 or not test:
            raise RuntimeError(f"final failed seed={seed} rc={rc}")
        test.update({
            "selected_lr": selected,
            "selected_lr_mean_val_acc1": sel.get("selected_mean_best_val_acc1"),
            "lr_selection_scope": "method_across_seeds",
            "test_used_for_hyperparameter_selection": False,
        })
        jwrite(out / "test_summary.json", test)
        ms = cfg["methods"][method]
        jwrite(out / "run_metadata.json", {
            "schema_version": 4,
            "target": target,
            "kind": cfg.get("kind", "comparison"),
            "manuscript_tables": cfg.get("manuscript_tables", []),
            "manuscript_figures": cfg.get("manuscript_figures", []),
            "method_preset": method,
            "method_label": ms.get("label", method),
            "variant": ms.get("variant"),
            "independent_seed": seed,
            "proposal": method == "dt1d",
            **runtime_metadata(repo),
        })
        jwrite(out / "resolved_config.json", {
            "schema_version": 4,
            "source_config": str(config_path),
            "experiment_id": target,
            "method": method,
            "seed": seed,
            "fairness": cfg["fairness"],
            "selected_lr": selected,
            "args": args,
            **runtime_metadata(repo),
        })
        jwrite(out / "selection_summary.json", sel)
        jwrite(out / "run_status.json", {"return_code": 0, "elapsed_seconds": dur})
        jwrite(out / "args.json", args)
        for p in out.glob("checkpoint*.pth"):
            p.unlink(missing_ok=True)
        return {
            "seed": seed,
            "test_acc1": test.get("test_acc1_at_best_val", test.get("acc1")),
            "best_val": conv.get("best_val_acc1"),
            "elapsed_seconds": dur,
        }

    # 4) Final training/test; also re-plan after each wave.
    if sel:
        selected = float(sel["selected_lr"])
        while True:
            pending_final = [
                s for s in seeds
                if not (finalroot / f"seed_{s}" / "test_summary.json").is_file()
            ]
            if not pending_final:
                break
            final_observed = []
            for s in seeds:
                st = jread(finalroot / f"seed_{s}" / "run_status.json")
                if finite(st.get("elapsed_seconds")):
                    final_observed.append(float(st["elapsed_seconds"]))
            wave_upper = max([final_upper] + [x * 1.20 for x in final_observed if x > 0])
            if remaining() < wave_upper:
                stop_reason = stop_reason or "not_enough_safe_time_for_next_final_wave"
                print(
                    f"STOP_BEFORE_FINAL_WAVE remaining={remaining()/60:.1f}min "
                    f"predicted_wave={wave_upper/60:.1f}min",
                    flush=True,
                )
                break
            wave = pending_final[:ng]
            session_waves.append({
                "phase": "final",
                "units": [{"seed": s} for s in wave],
                "selected_lr": selected,
                "started_elapsed_seconds": elapsed(),
                "remaining_before_seconds": remaining(),
                "predicted_wave_upper_seconds": wave_upper,
            })
            jwrite(root / "SESSION_PLAN.json", {
                "method": method,
                "gpu_count": ng,
                "waves": session_waves,
                "remaining_usable_seconds": remaining(),
                "policy": "re-plan after every completed GPU wave; never switch methods inside this session",
            })
            with cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
                fs = [
                    ex.submit(run_final, seed, devices[i], selected, sel)
                    for i, seed in enumerate(wave)
                ]
                for f in cf.as_completed(fs):
                    print("FINAL_RESULT", json.dumps(f.result(), default=str), flush=True)

    # 5) Status + method-level aggregation.
    pending_after = [(s, lr) for s in seeds for lr in lrs if not done(s, lr)]
    sel = jread(selp)
    final_missing = [
        s for s in seeds
        if not (finalroot / f"seed_{s}" / "test_summary.json").is_file()
    ]
    complete = (not pending_after) and bool(sel) and (not final_missing)
    if complete:
        agg = root / "aggregated" / target
        agg.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable,
            str(repo / "tools" / "aggregate_cnn_paper.py"),
            "--root", str(root / "results"),
            "--target", target,
            "--output-dir", str(agg),
            "--require-seeds", ",".join(map(str, seeds)),
        ], cwd=repo, check=True)

    status = {
        "complete": complete,
        "family": "cnn",
        "target": target,
        "method": method,
        "tuning_missing": [{"seed": s, "lr": lr} for s, lr in pending_after],
        "selection_complete": bool(sel),
        "selected_lr": sel.get("selected_lr") if sel else None,
        "final_missing_seeds": final_missing,
        "session_elapsed_minutes": elapsed() / 60.0,
        "session_limit_minutes": ns.safe_minutes,
        "remaining_usable_minutes": remaining() / 60.0,
        "stop_reason": stop_reason,
        "instruction": "If incomplete, upload THIS METHOD result ZIP to a fresh Kaggle session and rerun the SAME method script.",
    }
    jwrite(root / "TASK_STATUS.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
