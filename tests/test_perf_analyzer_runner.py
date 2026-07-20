import argparse
import asyncio
import csv
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def mb_to_gb(value_mb: Optional[float]) -> Optional[float]:
    if value_mb is None:
        return None
    return round(value_mb / 1024.0, 2)


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def render_row(row: List[Any]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    split_line = "-+-".join("-" * width for width in widths)
    print(render_row(headers))
    print(split_line)
    for row in rows:
        print(render_row(row))


@dataclass
class NpuMetric:
    ts_ms: int
    device_id: str
    used_mb: Optional[int]
    total_mb: Optional[int]
    utilization_pct: Optional[int]
    raw: str


class AscendNpuSampler:
    """
    采集单张 Ascend NPU 卡的显存占用。

    默认走：npu-smi info -t usages -i <device_id>
    """

    def __init__(
        self,
        device_id: str,
        interval_sec: float,
        enabled: bool = True,
        command: Optional[str] = None,
    ) -> None:
        self.device_id = device_id
        self.interval_sec = interval_sec
        self.enabled = enabled
        self.command = command or "npu-smi info"
        self.records: List[NpuMetric] = []
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _command_name(self) -> str:
        return self.command.strip().split()[0]

    def _command_exists(self) -> bool:
        return shutil.which(self._command_name()) is not None

    def _should_append_device_args(self) -> bool:
        stripped_command = self.command.strip()
        if not stripped_command.lower().startswith("npu-smi info"):
            return False
        if re.search(r"(^|\s)-i(\s|$)", stripped_command):
            return False
        if re.search(r"(^|\s)-t(\s|$)", stripped_command):
            return False
        return True

    def _build_sample_command(self) -> str:
        if self._should_append_device_args():
            return f"{self.command} -t usages -i {self.device_id}"
        return self.command

    def _extract_memory_pair(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        hbm_capacity_match = re.search(
            r"HBM\s+Capacity\(MB\)\s*:\s*(\d+)", text, re.IGNORECASE
        )
        hbm_usage_match = re.search(
            r"HBM\s+Usage\s+Rate\(%\)\s*:\s*(\d+)", text, re.IGNORECASE
        )
        if hbm_capacity_match and hbm_usage_match:
            total_mb = int(hbm_capacity_match.group(1))
            usage_rate = int(hbm_usage_match.group(1))
            used_mb = round(total_mb * usage_rate / 100.0)
            if 0 <= used_mb <= total_mb and total_mb >= 1024:
                return used_mb, total_mb

        patterns = [
            r"(\d+)\s*/\s*(\d+)\s*MB",
            r"(\d+)\s*/\s*(\d+)\s*MiB",
            r"(\d+)\s*/\s*(\d+)",
        ]
        for pattern in patterns:
            candidates = re.findall(pattern, text, re.IGNORECASE)
            for used_str, total_str in candidates:
                used_mb = int(used_str)
                total_mb = int(total_str)
                if 0 <= used_mb <= total_mb and total_mb >= 1024:
                    return used_mb, total_mb

        return None, None

    def _parse_output(
        self, raw: str
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        used_mb, total_mb = self._extract_memory_pair(raw)
        utilization_pct = None
        hbm_usage_match = re.search(
            r"HBM\s+Usage\s+Rate\(%\)\s*:\s*(\d+)", raw, re.IGNORECASE
        )
        if hbm_usage_match:
            utilization_pct = int(hbm_usage_match.group(1))
        else:
            util_candidates = re.findall(r"(\d+)\s*%", raw)
            if util_candidates:
                utilization_pct = int(util_candidates[0])
        return used_mb, total_mb, utilization_pct

    async def run(self) -> None:
        if not self.enabled:
            return

        if not self._command_exists():
            self.records.append(
                NpuMetric(
                    ts_ms=now_ms(),
                    device_id=self.device_id,
                    used_mb=None,
                    total_mb=None,
                    utilization_pct=None,
                    raw=f"sampler_error: command_not_found: {self._command_name()}",
                )
            )
            return

        while not self._stop:
            ts_ms = now_ms()
            raw = ""
            sample_command = self._build_sample_command()
            try:
                proc = await asyncio.create_subprocess_shell(
                    sample_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                raw = (stdout or b"").decode("utf-8", errors="ignore")
                err = (stderr or b"").decode("utf-8", errors="ignore")
                if err.strip():
                    raw = raw + "\n" + err

                used_mb, total_mb, utilization_pct = self._parse_output(raw)
                self.records.append(
                    NpuMetric(
                        ts_ms=ts_ms,
                        device_id=self.device_id,
                        used_mb=used_mb,
                        total_mb=total_mb,
                        utilization_pct=utilization_pct,
                        raw=raw.strip(),
                    )
                )
            except Exception as exc:
                self.records.append(
                    NpuMetric(
                        ts_ms=ts_ms,
                        device_id=self.device_id,
                        used_mb=None,
                        total_mb=None,
                        utilization_pct=None,
                        raw=f"sampler_error: {exc}",
                    )
                )

            await asyncio.sleep(self.interval_sec)


class PerfAnalyzerInputBuilder:
    """
    生成 perf_analyzer 所需的 input_data JSON。

    tensor_config_file 的格式示例：
    {
      "inputs": [
        {
          "name": "INPUT_IDS",
          "dtype": "INT64",
          "content_type": "int",
          "shape": [128],
          "min": 1,
          "max": 10000
        },
        {
          "name": "ATTN_MASK",
          "dtype": "INT64",
          "content_type": "int",
          "shape": [128],
          "min": 0,
          "max": 1
        }
      ]
    }
    """

    def __init__(
        self,
        tensor_config: Dict[str, Any],
        sample_count: int,
        seed: int,
    ) -> None:
        self.tensor_config = tensor_config
        self.sample_count = sample_count
        self.seed = seed
        random.seed(seed)

    def _shape_size(self, shape: List[int]) -> int:
        size = 1
        for dim in shape:
            size *= dim
        return size

    def _build_scalar_list(self, spec: Dict[str, Any]) -> List[Any]:
        shape = spec.get("shape", [])
        total_size = self._shape_size(shape)
        content_type = str(spec.get("content_type", "int")).lower()
        min_value = spec.get("min", 0)
        max_value = spec.get("max", 100)

        if content_type == "float":
            return [
                round(random.uniform(float(min_value), float(max_value)), 6)
                for _ in range(total_size)
            ]
        if content_type == "bool":
            return [random.choice([0, 1]) for _ in range(total_size)]
        if content_type == "string":
            prefix = str(spec.get("string_prefix", spec["name"]))
            return [
                f"{prefix}_{random.randint(int(min_value), int(max_value))}"
                for _ in range(total_size)
            ]
        return [
            random.randint(int(min_value), int(max_value)) for _ in range(total_size)
        ]

    def build(self) -> Dict[str, Any]:
        inputs = self.tensor_config.get("inputs", [])
        data: List[Dict[str, Any]] = []

        for _ in range(self.sample_count):
            item: Dict[str, Any] = {}
            for spec in inputs:
                tensor_name = spec["name"]
                tensor_content = self._build_scalar_list(spec)
                if spec.get("shape"):
                    item[tensor_name] = {
                        "content": tensor_content,
                        "shape": spec["shape"],
                    }
                else:
                    item[tensor_name] = tensor_content
            data.append(item)

        return {"data": data}


class PerfAnalyzerRunner:
    def __init__(
        self,
        perf_analyzer_bin: str,
        model_name: str,
        server_url: str,
        protocol: str,
        input_data_file: Path,
        output_dir: Path,
        batch_size: int,
        concurrency_range: str,
        measurement_interval_ms: int,
        percentile: int,
        warmup_request_count: int,
        extra_shapes: List[str],
        extra_headers: List[str],
        request_count: int,
        collect_metrics: bool,
        metrics_url: str,
    ) -> None:
        self.perf_analyzer_bin = perf_analyzer_bin
        self.model_name = model_name
        self.server_url = server_url
        self.protocol = protocol
        self.input_data_file = input_data_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.concurrency_range = concurrency_range
        self.measurement_interval_ms = measurement_interval_ms
        self.percentile = percentile
        self.warmup_request_count = warmup_request_count
        self.extra_shapes = extra_shapes
        self.extra_headers = extra_headers
        self.request_count = request_count
        self.collect_metrics = collect_metrics
        self.metrics_url = metrics_url

    def _command_exists(self) -> bool:
        return shutil.which(self.perf_analyzer_bin) is not None

    def build_command(self) -> List[str]:
        report_file = self.output_dir / "perf_analyzer_report.csv"
        command = [
            self.perf_analyzer_bin,
            "-m",
            self.model_name,
            "-u",
            self.server_url,
            "-i",
            self.protocol,
            "-b",
            str(self.batch_size),
            "--input-data",
            str(self.input_data_file),
            "--measurement-interval",
            str(self.measurement_interval_ms),
            "--concurrency-range",
            self.concurrency_range,
            "--percentile",
            str(self.percentile),
            "--warmup-request-count",
            str(self.warmup_request_count),
            "-f",
            str(report_file),
            "--verbose-csv",
        ]

        if self.request_count > 0:
            command.extend(["--request-count", str(self.request_count)])

        if self.collect_metrics:
            command.extend(["--collect-metrics", "--metrics-url", self.metrics_url])

        for shape in self.extra_shapes:
            command.extend(["--shape", shape])

        for header in self.extra_headers:
            command.extend(["-H", header])

        return command

    def run(self) -> Tuple[int, str, str, Path]:
        ensure_dir(self.output_dir)
        report_file = self.output_dir / "perf_analyzer_report.csv"

        if not self._command_exists():
            raise FileNotFoundError(
                f"perf_analyzer not found: {self.perf_analyzer_bin}"
            )

        command = self.build_command()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        return result.returncode, result.stdout, result.stderr, report_file


def load_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path: Path, payload: Dict[str, Any]) -> None:
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def summarize_perf_analyzer_csv(report_file: Path) -> Dict[str, Any]:
    if not report_file.exists():
        return {
            "row_count": 0,
            "best_concurrency": None,
            "best_infer_per_sec": None,
            "best_p95_latency_us": None,
        }

    with report_file.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return {
            "row_count": 0,
            "best_concurrency": None,
            "best_infer_per_sec": None,
            "best_p95_latency_us": None,
        }

    def safe_float(row: Dict[str, str], key: str) -> float:
        try:
            return float(row.get(key, "0") or 0)
        except ValueError:
            return 0.0

    best_row = max(rows, key=lambda row: safe_float(row, "Inferences/Second"))
    return {
        "row_count": len(rows),
        "best_concurrency": best_row.get("Concurrency"),
        "best_infer_per_sec": safe_float(best_row, "Inferences/Second"),
        "best_p95_latency_us": safe_float(best_row, "p95 latency"),
    }


def summarize_npu_metrics(npu_metrics: List[NpuMetric]) -> Dict[str, Any]:
    used_values = [item.used_mb for item in npu_metrics if item.used_mb is not None]
    total_values = [item.total_mb for item in npu_metrics if item.total_mb is not None]

    used_mb_min = min(used_values) if used_values else None
    used_mb_avg = round(statistics.mean(used_values), 2) if used_values else None
    used_mb_max = max(used_values) if used_values else None
    total_mb = max(total_values) if total_values else None

    return {
        "npu_sample_count": len(npu_metrics),
        "npu_valid_sample_count": len(used_values),
        "npu_used_mb_min": used_mb_min,
        "npu_used_mb_avg": used_mb_avg,
        "npu_used_mb_max": used_mb_max,
        "npu_total_mb": total_mb,
        "npu_used_gb_min": mb_to_gb(used_mb_min),
        "npu_used_gb_avg": mb_to_gb(used_mb_avg),
        "npu_used_gb_max": mb_to_gb(used_mb_max),
        "npu_total_gb": mb_to_gb(total_mb),
    }


def write_npu_csv(file_path: Path, npu_metrics: List[NpuMetric]) -> None:
    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ts_ms",
                "device_id",
                "used_mb",
                "total_mb",
                "utilization_pct",
                "raw",
            ],
        )
        writer.writeheader()
        for item in npu_metrics:
            writer.writerow(asdict(item))


def print_summary(perf_summary: Dict[str, Any], npu_summary: Dict[str, Any]) -> None:
    rows = [
        ["perf_rows", perf_summary["row_count"]],
        ["best_concurrency", perf_summary["best_concurrency"]],
        ["best_infer_per_sec", perf_summary["best_infer_per_sec"]],
        ["best_p95_latency_us", perf_summary["best_p95_latency_us"]],
        ["npu_sample_count", npu_summary["npu_sample_count"]],
        ["npu_valid_sample_count", npu_summary["npu_valid_sample_count"]],
        ["npu_used_mb_min", npu_summary["npu_used_mb_min"]],
        ["npu_used_mb_avg", npu_summary["npu_used_mb_avg"]],
        ["npu_used_mb_max", npu_summary["npu_used_mb_max"]],
        ["npu_total_mb", npu_summary["npu_total_mb"]],
        ["npu_used_gb_min", npu_summary["npu_used_gb_min"]],
        ["npu_used_gb_avg", npu_summary["npu_used_gb_avg"]],
        ["npu_used_gb_max", npu_summary["npu_used_gb_max"]],
        ["npu_total_gb", npu_summary["npu_total_gb"]],
    ]
    print("\n=== Perf Analyzer Summary ===")
    print_table(["metric", "value"], rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triton perf_analyzer benchmark wrapper"
    )
    parser.add_argument("--model-name", required=True, help="Triton model name")
    parser.add_argument(
        "--server-url", default="127.0.0.1:8000", help="Triton server address"
    )
    parser.add_argument(
        "--protocol", choices=["http", "grpc"], default="http", help="Triton protocol"
    )
    parser.add_argument(
        "--perf-analyzer-bin",
        default="perf_analyzer",
        help="perf_analyzer executable path",
    )
    parser.add_argument(
        "--output-dir", default="./perf_analyzer_output", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Request batch size")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=30,
        help="Number of input samples in input_data.json",
    )
    parser.add_argument("--seed", type=int, default=20260720, help="Random seed")
    parser.add_argument(
        "--concurrency-range", default="1:16:1", help="perf_analyzer concurrency range"
    )
    parser.add_argument(
        "--measurement-interval-ms",
        type=int,
        default=5000,
        help="Measurement interval in ms",
    )
    parser.add_argument(
        "--percentile", type=int, default=95, help="Percentile used by perf_analyzer"
    )
    parser.add_argument(
        "--warmup-request-count", type=int, default=20, help="Warmup request count"
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=0,
        help="Fixed request count for perf_analyzer; 0 means use stability mode",
    )
    parser.add_argument(
        "--tensor-config-file",
        required=True,
        help="JSON file that describes Triton model inputs",
    )
    parser.add_argument(
        "--extra-shape",
        action="append",
        default=[],
        help="Additional --shape option, for example INPUT_IDS:128",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional HTTP header, for example Authorization:Bearer xxx",
    )
    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        help="Enable perf_analyzer Prometheus metrics collection",
    )
    parser.add_argument(
        "--metrics-url",
        default="localhost:8002/metrics",
        help="Prometheus metrics endpoint used by perf_analyzer",
    )
    parser.add_argument(
        "--disable-npu-sampler", action="store_true", help="Disable Ascend NPU sampling"
    )
    parser.add_argument("--npu-device-id", default="0", help="Ascend device id")
    parser.add_argument(
        "--npu-sample-interval-sec",
        type=float,
        default=1.0,
        help="NPU sample interval in seconds",
    )
    parser.add_argument(
        "--npu-command", default="npu-smi info", help="NPU sampling command"
    )
    return parser


async def main_async(args: argparse.Namespace) -> None:
    base_dir = Path(args.output_dir).resolve()
    inputs_dir = base_dir / "inputs"
    reports_dir = base_dir / "reports"
    ensure_dir(base_dir)
    ensure_dir(inputs_dir)
    ensure_dir(reports_dir)

    tensor_config = load_json(Path(args.tensor_config_file))
    input_builder = PerfAnalyzerInputBuilder(
        tensor_config=tensor_config,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    input_data = input_builder.build()
    input_data_file = inputs_dir / "perf_analyzer_input_data.json"
    write_json(input_data_file, input_data)

    sampler = AscendNpuSampler(
        device_id=args.npu_device_id,
        interval_sec=args.npu_sample_interval_sec,
        enabled=not args.disable_npu_sampler,
        command=args.npu_command,
    )
    sampler_task = (
        asyncio.create_task(sampler.run()) if not args.disable_npu_sampler else None
    )

    runner = PerfAnalyzerRunner(
        perf_analyzer_bin=args.perf_analyzer_bin,
        model_name=args.model_name,
        server_url=args.server_url,
        protocol=args.protocol,
        input_data_file=input_data_file,
        output_dir=reports_dir,
        batch_size=args.batch_size,
        concurrency_range=args.concurrency_range,
        measurement_interval_ms=args.measurement_interval_ms,
        percentile=args.percentile,
        warmup_request_count=args.warmup_request_count,
        extra_shapes=args.extra_shape,
        extra_headers=args.header,
        request_count=args.request_count,
        collect_metrics=args.collect_metrics,
        metrics_url=args.metrics_url,
    )

    try:
        return_code, stdout_text, stderr_text, report_file = await asyncio.to_thread(
            runner.run
        )
    finally:
        sampler.stop()
        if sampler_task is not None:
            await sampler_task

    command_file = reports_dir / "perf_analyzer_stdout.txt"
    command_file.write_text(
        stdout_text + "\n\nSTDERR:\n" + stderr_text, encoding="utf-8"
    )

    npu_csv = reports_dir / "npu_metrics.csv"
    write_npu_csv(npu_csv, sampler.records)

    perf_summary = summarize_perf_analyzer_csv(report_file)
    npu_summary = summarize_npu_metrics(sampler.records)
    combined_summary = {
        "return_code": return_code,
        **perf_summary,
        **npu_summary,
    }

    summary_file = reports_dir / "summary.json"
    write_json(summary_file, combined_summary)
    print_summary(perf_summary, npu_summary)

    print("\n=== Output Files ===")
    print(f"input_data_json: {input_data_file}")
    print(f"perf_report_csv: {report_file}")
    print(f"perf_stdout_txt: {command_file}")
    print(f"npu_metrics_csv: {npu_csv}")
    print(f"summary_json: {summary_file}")

    if return_code != 0:
        raise SystemExit(return_code)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    """示例：
    python test_perf_analyzer_runner.py \
      --model-name my_triton_model \
      --server-url 127.0.0.1:8000 \
      --protocol http \
      --tensor-config-file ./tensor_config.json \
      --batch-size 1 \
      --sample-count 50 \
      --concurrency-range 1:32:1 \
      --measurement-interval-ms 5000 \
      --percentile 95 \
      --output-dir ./perf_analyzer_run
    """
    main()
