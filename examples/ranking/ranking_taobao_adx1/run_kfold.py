from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "kfold"

MODEL_SPECS = {
    "dcn_v2": {
        "package": "reckit.ranking.dcn_v2",
        "train_config": "train_dcn_v2.json",
    },
    "rankmixer": {
        "package": "reckit.ranking.rankmixer",
        "train_config": "train_rankmixer.json",
    },
    "onetrans": {
        "package": "reckit.ranking.onetrans",
        "train_config": "train_onetrans.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blocked time K-fold TaobaoAd_x1 ranking experiments."
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_SPECS), default=list(MODEL_SPECS)
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of contiguous time blocks used as folds.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--valid-policy",
        choices=["adjacent", "next", "prev", "test"],
        default="adjacent",
        help="How to choose the validation fold relative to the held-out test fold.",
    )
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-splits", nargs="+", default=["valid", "test"])
    parser.add_argument(
        "--override-json",
        default=None,
        help="Optional JSON file with per-model config overrides.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[run_kfold_taobaoad_x1] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_overrides(args: argparse.Namespace) -> dict[str, Any]:
    if not args.override_json:
        return {}
    path = Path(args.override_json)
    if not path.is_absolute():
        path = Path.cwd() / path
    return read_json(path)


def blocked_fold_ids(n_samples: int, k: int) -> list[int]:
    if k < 3:
        raise ValueError("k must be >= 3 so train/valid/test are all non-empty")
    if n_samples < k:
        raise ValueError(f"sample count {n_samples} is smaller than k={k}")
    base = n_samples // k
    remainder = n_samples % k
    fold_ids: list[int] = []
    for fold in range(k):
        size = base + (1 if fold < remainder else 0)
        fold_ids.extend([fold] * size)
    return fold_ids


def resolve_valid_fold(fold: int, k: int, valid_policy: str) -> int:
    if valid_policy == "test":
        return fold
    if valid_policy == "next":
        return (fold + 1) % k
    if valid_policy == "prev":
        return (fold - 1 + k) % k
    if fold == k - 1:
        return k - 2
    return fold + 1


def make_time_block_splitter(k: int, fold: int, valid_policy: str):
    def split_samples(samples: list[dict[str, Any]], config: dict[str, Any]):
        if not samples:
            raise ValueError("no samples available for K-fold split")
        fold_ids = blocked_fold_ids(len(samples), k)
        valid_fold = resolve_valid_fold(fold, k, valid_policy)
        train: list[dict[str, Any]] = []
        valid: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        for sample, fold_id in zip(samples, fold_ids):
            if fold_id == fold:
                test.append(sample)
            elif fold_id == valid_fold:
                valid.append(sample)
            else:
                train.append(sample)
        if min(len(train), len(valid), len(test)) == 0:
            raise ValueError(
                f"empty blocked K-fold split produced: train={len(train)} valid={len(valid)} test={len(test)}"
            )
        return train, valid, test

    return split_samples


def prepare_fold_configs(
    model_name: str, fold: int, args: argparse.Namespace, output_dir: Path
) -> Path:
    spec = MODEL_SPECS[model_name]
    base_config_dir = PROJECT_DIR / model_name / "configs"
    fold_root = output_dir / model_name / f"fold_{fold:02d}"
    config_dir = fold_root / "configs"
    data_dir = fold_root / "data"
    checkpoint_dir = fold_root / "checkpoints"

    data_cfg = read_json(base_config_dir / "data.json")
    data_cfg["output_dir"] = str(data_dir)
    data_cfg.setdefault("sample_builder", {})["split"] = {
        "type": "time_block_kfold",
        "k": int(args.k),
        "fold": int(fold),
        "valid_fold": int(
            resolve_valid_fold(fold, int(args.k), str(args.valid_policy))
        ),
        "valid_policy": str(args.valid_policy),
        "seed": int(args.seed),
    }

    train_cfg = read_json(base_config_dir / str(spec["train_config"]))
    overrides = getattr(args, "config_overrides", {})
    if model_name in overrides:
        deep_update(train_cfg, overrides[model_name])
    train_cfg["data_dir"] = str(data_dir)
    train_cfg["save_dir"] = str(checkpoint_dir)
    train_cfg["device"] = args.device
    train_cfg["seed"] = int(args.seed)

    infer_cfg = read_json(base_config_dir / "infer.json")
    infer_cfg["data_dir"] = str(data_dir)
    infer_cfg["save_dir"] = str(checkpoint_dir)
    infer_cfg["checkpoint"] = str(checkpoint_dir / "best.pth")
    infer_cfg["device"] = args.device
    infer_cfg["seed"] = int(args.seed)

    write_json(config_dir / "data.json", data_cfg)
    write_json(config_dir / str(spec["train_config"]), train_cfg)
    write_json(config_dir / "infer.json", infer_cfg)
    _log(
        f"prepared fold configs: model={model_name}, fold={fold}, config_dir={config_dir}"
    )
    return config_dir


def process_fold(
    model_name: str, fold: int, args: argparse.Namespace, config_dir: Path
) -> None:
    data_cfg = read_json(config_dir / "data.json")
    data_dir = Path(str(data_cfg["output_dir"]))
    meta_path = data_dir / "meta.json"
    if meta_path.exists() and not args.force_process:
        _log(
            f"reuse processed fold data: model={model_name}, fold={fold}, data_dir={data_dir}"
        )
        return
    if data_dir.exists():
        shutil.rmtree(data_dir)

    package = str(MODEL_SPECS[model_name]["package"])
    process_module = importlib.import_module(f"{package}.process")
    original_split = process_module._split_samples
    process_module._split_samples = make_time_block_splitter(
        k=int(args.k),
        fold=int(fold),
        valid_policy=str(args.valid_policy),
    )
    try:
        _log(f"start process fold: model={model_name}, fold={fold}")
        process_module.build_ranking_data(data_cfg)
        _log(f"finished process fold: model={model_name}, fold={fold}")
    finally:
        process_module._split_samples = original_split


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> str:
    _log(f"run command: {' '.join(cmd)}")
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\nSee log: {log_path}"
        )
    return proc.stdout


def train_fold(
    model_name: str, fold: int, args: argparse.Namespace, config_dir: Path
) -> Path:
    spec = MODEL_SPECS[model_name]
    train_cfg = read_json(config_dir / str(spec["train_config"]))
    save_dir = train_cfg.get("save_dir")
    if not save_dir:
        raise ValueError(f"Missing save_dir in fold train config: {config_dir}")
    checkpoint = Path(str(save_dir)) / "best.pth"
    if args.skip_train:
        _log(f"skip train fold: model={model_name}, fold={fold}")
        return checkpoint
    if checkpoint.exists() and not args.force_train:
        _log(
            f"reuse fold checkpoint: model={model_name}, fold={fold}, checkpoint={checkpoint}"
        )
        return checkpoint

    cmd = [
        args.python,
        "-m",
        f"{spec['package']}.train",
        "--config",
        str(config_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    run_command(
        cmd,
        cwd=PROJECT_DIR.parent.parent,
        log_path=config_dir.parent / "logs" / "train.log",
    )
    _log(
        f"finished train fold: model={model_name}, fold={fold}, checkpoint={checkpoint}"
    )
    return checkpoint


def extract_json(output: str) -> dict[str, Any]:
    start = output.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in output:\n{output}")
    return json.loads(output[start:])


def evaluate_fold(
    model_name: str, fold: int, args: argparse.Namespace, config_dir: Path, split: str
) -> dict[str, Any]:
    package = str(MODEL_SPECS[model_name]["package"])
    cmd = [
        args.python,
        "-m",
        f"{package}.evaluate",
        "--config",
        str(config_dir),
        "--split",
        split,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    output = run_command(
        cmd,
        cwd=PROJECT_DIR.parent.parent,
        log_path=config_dir.parent / "logs" / f"evaluate_{split}.log",
    )
    payload = extract_json(output)
    _log(f"finished evaluate fold: model={model_name}, fold={fold}, split={split}")
    return payload["metrics"]


def evaluate_splits(
    model_name: str, fold: int, args: argparse.Namespace, config_dir: Path
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in args.eval_splits:
        split_metrics = evaluate_fold(model_name, fold, args, config_dir, str(split))
        for key, value in split_metrics.items():
            metrics[f"{split}_{key}"] = float(value)
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    summary: dict[str, Any] = {}
    for model, items in sorted(by_model.items()):
        metric_names = sorted(
            key for key in items[0].keys() if key not in {"model", "fold"}
        )
        summary[model] = {
            metric: {
                "mean": mean(float(item[metric]) for item in items),
                "std": pstdev(float(item[metric]) for item in items),
            }
            for metric in metric_names
        }
    return summary


def write_results(
    output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "results.json", {"folds": rows, "summary": summary})

    metric_names = (
        sorted(key for key in rows[0] if key not in {"model", "fold"}) if rows else []
    )
    with (output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fold", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "mean", "std"])
        writer.writeheader()
        for model, metrics in summary.items():
            for metric, values in metrics.items():
                writer.writerow({"model": model, "metric": metric, **values})

    lines = ["# TaobaoAd_x1 Ranking K-fold Results", ""]
    for model, metrics in summary.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| metric | mean | std |")
        lines.append("| --- | ---: | ---: |")
        for metric, values in metrics.items():
            lines.append(f"| {metric} | {values['mean']:.6f} | {values['std']:.6f} |")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.config_overrides = load_overrides(args)
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    _log(
        f"start blocked K-fold benchmark: models={args.models}, k={args.k}, device={args.device}, "
        f"valid_policy={args.valid_policy}, output_dir={output_dir}"
    )

    for model_name in args.models:
        for fold in range(int(args.k)):
            _log(f"start fold loop: model={model_name}, fold={fold}")
            config_dir = prepare_fold_configs(model_name, int(fold), args, output_dir)
            process_fold(model_name, int(fold), args, config_dir)
            checkpoint = train_fold(model_name, int(fold), args, config_dir)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint missing for model={model_name}, fold={fold}: {checkpoint}"
                )
            metrics = evaluate_splits(model_name, int(fold), args, config_dir)
            row = {"model": model_name, "fold": int(fold), **metrics}
            rows.append(row)
            write_results(output_dir, rows, summarize(rows))
            _log(
                f"finished fold loop: model={model_name}, fold={fold}, metrics={metrics}"
            )
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

    summary = summarize(rows)
    write_results(output_dir, rows, summary)
    _log(f"finished blocked K-fold benchmark: summary={output_dir / 'results.json'}")
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
