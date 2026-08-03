"""Run the Taobao Ad baseline ranking experiments for DCNv2, RankMixer, and OneTrans.

Usage (from repo root):
    PYTHONPATH=. python projects/ranking_taobao_ad/run_baseline.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "benchmark"

MODEL_SPECS = {
    "dcn_v2": {
        "package": "reckit.ranking.dcn_v2",
        "train_config": PROJECT_DIR / "dcn_v2" / "configs" / "train_dcn_v2.json",
        "data_config": PROJECT_DIR / "dcn_v2" / "configs" / "data.json",
        "config_dir": PROJECT_DIR / "dcn_v2" / "configs",
    },
    "rankmixer": {
        "package": "reckit.ranking.rankmixer",
        "train_config": PROJECT_DIR / "rankmixer" / "configs" / "train_rankmixer.json",
        "data_config": PROJECT_DIR / "rankmixer" / "configs" / "data.json",
        "config_dir": PROJECT_DIR / "rankmixer" / "configs",
    },
    "onetrans": {
        "package": "reckit.ranking.onetrans",
        "train_config": PROJECT_DIR / "onetrans" / "configs" / "train_onetrans.json",
        "data_config": PROJECT_DIR / "onetrans" / "configs" / "data.json",
        "config_dir": PROJECT_DIR / "onetrans" / "configs",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Taobao Ad ranking experiments for DCNv2 / RankMixer / OneTrans."
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_SPECS), default=list(MODEL_SPECS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def _log(msg: str) -> None:
    print(f"[run_baseline] {msg}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> str:
    _log(f"run: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:]
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n{tail}"
        )
    return proc.stdout or ""


def _extract_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        if "auc" in line.lower() or "logloss" in line.lower():
            parts = line.replace(",", " ").split()
            for i, token in enumerate(parts):
                key = token.strip().lower().rstrip(":")
                if key in {"auc", "logloss"} and i + 1 < len(parts):
                    try:
                        metrics[key] = float(parts[i + 1])
                    except ValueError:
                        pass
    return metrics


def _run_model(
    model_name: str,
    spec: dict[str, Any],
    seed: int,
    device: str,
    output_dir: Path,
    repo_root: Path,
    python: str,
    eval_splits: list[str],
    force_process: bool,
    skip_process: bool,
    force_train: bool,
    skip_train: bool,
) -> dict[str, Any]:
    model_dir = output_dir / model_name / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)

    config_dir = model_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy and patch data config
    base_data_cfg = _read_json(spec["data_config"])
    data_cfg = _deep_update(dict(base_data_cfg), {})
    data_cfg_path = config_dir / "data.json"
    _write_json(data_cfg_path, data_cfg)

    # Copy and patch train config
    base_train_cfg = _read_json(spec["train_config"])
    train_cfg = _deep_update(
        dict(base_train_cfg),
        {
            "device": device,
            "seed": seed,
            "save_dir": str(model_dir / "checkpoints"),
            "data_dir": base_train_cfg.get("data_dir", ""),
        },
    )
    train_cfg_path = config_dir / Path(spec["train_config"]).name
    _write_json(train_cfg_path, train_cfg)

    result: dict[str, Any] = {"model": model_name, "seed": seed}

    # Step 1: process
    process_done_flag = model_dir / "process.done"
    if not skip_process and (force_process or not process_done_flag.exists()):
        _log(f"[{model_name}/seed={seed}] processing data ...")
        out = _run_command(
            [
                python,
                "-m",
                spec["package"] + ".process",
                "--config",
                str(data_cfg_path),
            ],
            cwd=repo_root,
            log_path=model_dir / "process.log",
        )
        process_done_flag.write_text("done\n", encoding="utf-8")
    else:
        _log(f"[{model_name}/seed={seed}] skip process (already done)")

    # Step 2: train
    checkpoint_path = Path(train_cfg["save_dir"]) / "best.pth"
    if not skip_train and (force_train or not checkpoint_path.exists()):
        _log(f"[{model_name}/seed={seed}] training ...")
        out = _run_command(
            [python, "-m", spec["package"] + ".train", "--config", str(train_cfg_path)],
            cwd=repo_root,
            log_path=model_dir / "train.log",
        )
        result["train_output_tail"] = out[-500:]
    else:
        _log(
            f"[{model_name}/seed={seed}] skip train (checkpoint exists or --skip-train)"
        )

    # Step 3: evaluate on each split
    for split in eval_splits:
        infer_cfg = {
            "data_dir": train_cfg.get("data_dir", ""),
            "save_dir": train_cfg["save_dir"],
            "checkpoint": str(checkpoint_path),
            "device": device,
            "seed": seed,
            "training": {
                "batch_size": train_cfg.get("training", {}).get("batch_size", 1024)
            },
            "eval_split": split,
        }
        infer_cfg_path = config_dir / f"infer_{split}.json"
        _write_json(infer_cfg_path, infer_cfg)

        _log(f"[{model_name}/seed={seed}] evaluating split={split} ...")
        out = _run_command(
            [
                python,
                "-m",
                spec["package"] + ".evaluate",
                "--config",
                str(infer_cfg_path),
            ],
            cwd=repo_root,
            log_path=model_dir / f"eval_{split}.log",
        )
        metrics = _extract_metrics(out)
        for k, v in metrics.items():
            result[f"{split}_{k}"] = v
        _log(f"[{model_name}/seed={seed}] {split} metrics: {metrics}")

    return result


def _write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return

    metric_keys = sorted(
        {
            k
            for r in results
            for k in r
            if k not in {"model", "seed", "train_output_tail"}
        }
    )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "seed"] + metric_keys, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(results)
    _log(f"wrote {csv_path}")

    # Aggregate by model across seeds
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    agg_rows = []
    for model_name, rows in sorted(by_model.items()):
        agg: dict[str, Any] = {"model": model_name, "num_seeds": len(rows)}
        for mk in metric_keys:
            vals = [r[mk] for r in rows if mk in r]
            if vals:
                agg[f"{mk}_mean"] = round(mean(vals), 6)
                if len(vals) > 1:
                    agg[f"{mk}_std"] = round(pstdev(vals), 6)
        agg_rows.append(agg)

    agg_path = output_dir / "summary_aggregated.json"
    agg_path.write_text(
        json.dumps(agg_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _log(f"wrote {agg_path}")

    print("\n=== Benchmark Results ===")
    for row in agg_rows:
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    repo_root = PROJECT_DIR.parent.parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"repo_root={repo_root}, output_dir={output_dir}, device={args.device}")
    _log(f"models={args.models}, seeds={args.seeds}")

    all_results: list[dict[str, Any]] = []
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        for seed in args.seeds:
            try:
                result = _run_model(
                    model_name=model_name,
                    spec=spec,
                    seed=seed,
                    device=args.device,
                    output_dir=output_dir,
                    repo_root=repo_root,
                    python=args.python,
                    eval_splits=args.eval_splits,
                    force_process=args.force_process,
                    skip_process=args.skip_process,
                    force_train=args.force_train,
                    skip_train=args.skip_train,
                )
                all_results.append(result)
            except Exception as exc:
                _log(f"[{model_name}/seed={seed}] FAILED: {exc}")
                all_results.append(
                    {"model": model_name, "seed": seed, "error": str(exc)}
                )

    _write_summary(all_results, output_dir)
    _log("all done.")


if __name__ == "__main__":
    main()
