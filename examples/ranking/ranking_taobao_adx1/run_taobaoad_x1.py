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
        description="Run TaobaoAd_x1 ranking experiments for DCNv2/RankMixer/OneTrans."
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


def _log(message: str) -> None:
    print(f"[run_taobaoad_x1] {message}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> str:
    _log(f"run command: {' '.join(cmd)}")
    _log(f"write command log: {log_path}")
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
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}"
        )
    _log(f"command finished successfully: {log_path}")
    return proc.stdout


def _extract_json_object(output: str) -> dict[str, Any]:
    start = output.find("{")
    if start < 0:
        raise ValueError("No JSON object found in command output")
    return json.loads(output[start:])


def _model_train_checkpoint_path(model: str) -> Path:
    spec = MODEL_SPECS[model]
    train_cfg = _read_json(Path(spec["train_config"]))
    save_dir = train_cfg.get("save_dir")
    if not save_dir:
        raise ValueError(f"Missing save_dir in train config for model={model}")
    return Path(str(save_dir)) / "best.pth"


def _process_model(model: str, args: argparse.Namespace, output_dir: Path) -> None:
    spec = MODEL_SPECS[model]
    data_cfg = _read_json(Path(spec["data_config"]))
    model_data_dir = Path(str(data_cfg["output_dir"]))
    meta_path = model_data_dir / "meta.json"

    if args.skip_process:
        _log(f"skip process for model={model} due to --skip-process")
        return
    if model_data_dir.exists() and args.force_process:
        _log(f"remove existing processed data for model={model}: {model_data_dir}")
        shutil.rmtree(model_data_dir)
    if meta_path.exists() and not args.force_process:
        _log(f"reuse existing processed data for model={model}: {model_data_dir}")
        return

    _log(f"start process for model={model}")
    cmd = [
        args.python,
        "-m",
        f"{spec['package']}.process",
        "--config",
        str(spec["data_config"]),
    ]
    log_path = output_dir / model / "process.log"
    _run_command(cmd, cwd=PROJECT_DIR.parent.parent, log_path=log_path)
    _log(f"finished process for model={model}")


def _train_model(
    model: str, seed: int, args: argparse.Namespace, output_dir: Path
) -> Path:
    spec = MODEL_SPECS[model]
    checkpoint = _model_train_checkpoint_path(model)
    if args.skip_train:
        _log(f"skip train for model={model}, seed={seed} due to --skip-train")
        return checkpoint
    if checkpoint.exists() and not args.force_train:
        _log(f"reuse existing checkpoint for model={model}, seed={seed}: {checkpoint}")
        return checkpoint

    _log(f"start train for model={model}, seed={seed}")
    cmd = [
        args.python,
        "-m",
        f"{spec['package']}.train",
        "--config",
        str(spec["config_dir"]),
        "--device",
        args.device,
        "--seed",
        str(seed),
    ]
    log_path = output_dir / model / f"seed_{seed}" / "train.log"
    _run_command(cmd, cwd=PROJECT_DIR.parent.parent, log_path=log_path)
    _log(f"finished train for model={model}, seed={seed}, checkpoint={checkpoint}")
    return checkpoint


def _evaluate_model(
    model: str,
    seed: int,
    split: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    spec = MODEL_SPECS[model]
    _log(f"start evaluate for model={model}, seed={seed}, split={split}")
    cmd = [
        args.python,
        "-m",
        f"{spec['package']}.evaluate",
        "--config",
        str(spec["config_dir"]),
        "--split",
        split,
        "--device",
        args.device,
        "--seed",
        str(seed),
    ]
    log_path = output_dir / model / f"seed_{seed}" / f"evaluate_{split}.log"
    output = _run_command(cmd, cwd=PROJECT_DIR.parent.parent, log_path=log_path)
    payload = _extract_json_object(output)
    metrics = {key: float(value) for key, value in payload.get("metrics", {}).items()}
    _log(
        f"finished evaluate for model={model}, seed={seed}, split={split}, metrics={metrics}"
    )
    return metrics


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["model"]), str(row["split"])), []).append(row)

    summary: dict[str, Any] = {}
    for (model, split), items in sorted(by_key.items()):
        metric_names = sorted(
            [key for key in items[0].keys() if key not in {"model", "seed", "split"}]
        )
        model_summary = summary.setdefault(model, {})
        split_summary: dict[str, Any] = {}
        for metric in metric_names:
            values = [float(item[metric]) for item in items]
            split_summary[metric] = {
                "mean": mean(values),
                "std": pstdev(values),
            }
        model_summary[split] = split_summary
    return summary


def _write_results(
    output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"write summary files to {output_dir} (rows={len(rows)})")

    payload = {"rows": rows, "summary": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if rows:
        headers = ["model", "seed", "split"] + sorted(
            [key for key in rows[0].keys() if key not in {"model", "seed", "split"}]
        )
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# TaobaoAd_x1 Ranking Summary", ""]
    for model, split_map in summary.items():
        lines.append(f"## {model}")
        lines.append("")
        for split, metric_map in split_map.items():
            lines.append(f"### split={split}")
            lines.append("")
            lines.append("| metric | mean | std |")
            lines.append("| --- | ---: | ---: |")
            for metric, stat in metric_map.items():
                lines.append(f"| {metric} | {stat['mean']:.6f} | {stat['std']:.6f} |")
            lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    _log(
        f"start benchmark: models={args.models}, seeds={args.seeds}, device={args.device}, "
        f"output_dir={output_dir}"
    )

    for model in args.models:
        _log(f"start model loop: model={model}")
        _process_model(model, args, output_dir)
        for seed in args.seeds:
            _log(f"start seed loop: model={model}, seed={seed}")
            checkpoint = _train_model(model, int(seed), args, output_dir)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Training checkpoint missing for model={model}, seed={seed}: {checkpoint}"
                )
            for split in args.eval_splits:
                metrics = _evaluate_model(
                    model, int(seed), str(split), args, output_dir
                )
                rows.append(
                    {"model": model, "seed": int(seed), "split": str(split), **metrics}
                )
                _write_results(output_dir, rows, _summarize(rows))
            _log(f"finished seed loop: model={model}, seed={seed}")
        _log(f"finished model loop: model={model}")

    summary = _summarize(rows)
    _write_results(output_dir, rows, summary)
    _log(
        f"finished benchmark: total_rows={len(rows)}, summary_file={output_dir / 'summary.json'}"
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "rows": len(rows), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
