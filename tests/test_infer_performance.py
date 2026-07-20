import argparse
import asyncio
import csv
import json
import math
import random
import re
import shutil
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# =========================
# 配置区：默认参数与目录工具
# =========================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def mb_to_gb(value_mb: Optional[float]) -> Optional[float]:
    if value_mb is None:
        return None
    return round(value_mb / 1024.0, 2)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    low_value = ordered[lower]
    high_value = ordered[upper]
    return low_value + (high_value - low_value) * (rank - lower)


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


# =========================
# 数据结构区：压测结果对象
# =========================


@dataclass
class RequestMetric:
    request_index: int
    scenario_name: str
    uid: str
    history_len: int
    start_ts_ms: int
    end_ts_ms: int
    latency_ms: float
    status: int
    transport_success: int
    http_success: int
    business_success: int
    final_success: int
    request_bytes: int
    response_bytes: int
    response_code: str
    response_message: str
    response_file: str
    error: str


@dataclass
class NpuMetric:
    ts_ms: int
    device_id: str
    used_mb: Optional[int]
    total_mb: Optional[int]
    utilization_pct: Optional[int]
    raw: str


# =========================
# 核心逻辑区 1：请求生成
# =========================


class RequestGenerator:
    """
    生成批量 request.json 请求文件。

    设计目标：
    1. 方便重复复用同一批请求做版本对比
    2. 方便针对不同 history 长度分场景压测
    3. 后续容易替换成真实样本回放
    """

    def __init__(
        self,
        output_dir: Path,
        total_requests: int,
        seed: int,
        metadata_template: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.total_requests = total_requests
        self.seed = seed
        self.metadata_template = metadata_template or {}
        random.seed(seed)

    def scenario_pool(self) -> List[Dict[str, Any]]:
        """
        当前按照 history 长度定义 3 类场景。
        如果后续要扩展更多请求变量，优先改这里。
        """
        return [
            {"scenario_name": "light", "history_min": 5, "history_max": 10},
            {"scenario_name": "medium", "history_min": 20, "history_max": 50},
            {"scenario_name": "heavy", "history_min": 100, "history_max": 200},
        ]

    def build_history(
        self, history_len: int, base_timestamp: int
    ) -> List[Dict[str, Any]]:
        """
        生成 history 行为序列。

        约束：
        1. action 只取 0 或 1
        2. rating 只取 1 到 5
        3. timestamp 递增，模拟真实行为时间线
        """
        history = []
        timestamp = base_timestamp

        for _ in range(history_len):
            timestamp += random.randint(1, 30)
            history.append(
                {
                    "iid": str(random.randint(100, 999999)),
                    "timestamp": timestamp,
                    "action": random.choice([0, 1]),
                    "rating": random.randint(1, 5),
                }
            )
        return history

    def build_request(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成单条请求。

        如果你的真实接口后续新增字段，优先改这个函数。
        """
        history_len = random.randint(scenario["history_min"], scenario["history_max"])
        base_timestamp = 978300000 + random.randint(0, 50000)

        return {
            "uid": str(random.randint(1, 10000000)),
            "history": self.build_history(history_len, base_timestamp),
            "query_context": {
                "action": random.choice([0, 1]),
                "rating": random.randint(1, 5),
            },
            "_metadata": self.metadata_template,
        }

    def generate(self) -> List[Path]:
        """
        生成请求文件并保存到 requests 目录。

        每条请求单独保存，便于：
        1. 定位异常样本
        2. 重放同一批请求
        3. 多版本服务横向对比
        """
        ensure_dir(self.output_dir)
        files: List[Path] = []
        scenarios = self.scenario_pool()

        for idx in range(self.total_requests):
            scenario = scenarios[idx % len(scenarios)]
            payload = self.build_request(scenario)
            file_path = (
                self.output_dir / f"request_{idx:05d}_{scenario['scenario_name']}.json"
            )
            with file_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            files.append(file_path)

        return files


# =========================
# 核心逻辑区 2：Ascend NPU 采样
# =========================


class AscendNpuSampler:
    """
    周期采样 Ascend NPU 指标。

    默认通过 npu-smi info 获取数据。
    如果后续你们环境里有更稳定的采集方式，替换这里即可。
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
        """
        默认命令是 npu-smi info 时，自动补上 -t usages -i <device_id>。

        这样可以直接让 npu-smi 只返回目标卡信息，
        避免多卡输出时再靠文本匹配导致误判。
        如果用户已经手动传了 -i 或 -t，则尊重用户自定义命令。
        """
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
        """
        提取显存占用。

        兼容几种常见格式：
        1. 27162 / 32768 MB
        2. 27162/32768MB
        3. 27162 / 32768（部分 npu-smi 版本不带 MB）
        """
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
        """
        从单卡采样命令输出中提取显存和利用率。

        这里不再做按 device_id 的文本筛选，也不再对全量多卡输出做兜底。
        当前假设调用命令已经通过 -i <device_id> 限定到了单张卡。
        """
        used_mb, total_mb = self._extract_memory_pair(raw)
        utilization_pct = None
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


# =========================
# 核心逻辑区 3：并发压测执行
# =========================


class LoadTester:
    """
    并发发送请求并记录性能指标。

    这是压测主路径：
    1. 读取请求文件
    2. 并发发送 HTTP 请求
    3. 记录请求级别的结果
    """

    def __init__(
        self,
        endpoint: str,
        headers: Dict[str, str],
        concurrency: int,
        timeout_sec: float,
        requests_dir: Path,
        responses_dir: Path,
    ) -> None:
        self.endpoint = endpoint
        self.headers = headers
        self.concurrency = concurrency
        self.timeout_sec = timeout_sec
        self.requests_dir = requests_dir
        self.responses_dir = responses_dir
        self.metrics: List[RequestMetric] = []

    def load_requests(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        requests = []
        files = sorted(self.requests_dir.glob("request_*.json"))

        for idx, path in enumerate(files):
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            requests.append((idx, path.name, payload))

        return requests

    def detect_scenario(self, file_name: str) -> str:
        if "light" in file_name:
            return "light"
        if "medium" in file_name:
            return "medium"
        if "heavy" in file_name:
            return "heavy"
        return "unknown"

    def parse_business_result(
        self,
        status: int,
        response_text: str,
    ) -> Tuple[int, int, int, str, str, Any]:
        """
        解析业务成功状态。

        成功分三层：
        1. transport_success: 成功拿到响应，没有连接或超时异常
        2. http_success: HTTP 状态码在 2xx
        3. business_success: 响应体中的业务字段判定成功

        当前兼容规则：
        1. 有 success 字段时，以 success 为准
        2. 有 code 字段时，0/OK/SUCCESS 视为成功
        3. 都没有时，视为业务失败，避免把仅有 HTTP 200 的响应误判为成功
        """
        transport_success = 1
        http_success = 1 if 200 <= status < 300 else 0
        business_success = 0
        response_code = ""
        response_message = ""

        try:
            parsed_body: Any = json.loads(response_text)
        except json.JSONDecodeError:
            parsed_body = response_text

        if isinstance(parsed_body, dict):
            if "success" in parsed_body:
                business_success = 1 if bool(parsed_body.get("success")) else 0
                response_code = str(parsed_body.get("code", ""))
                response_message = str(
                    parsed_body.get("message", parsed_body.get("msg", ""))
                )
            elif "code" in parsed_body:
                code_value = parsed_body.get("code")
                response_code = str(code_value)
                response_message = str(
                    parsed_body.get("message", parsed_body.get("msg", ""))
                )
                business_success = (
                    1 if str(code_value).upper() in {"0", "OK", "SUCCESS"} else 0
                )
            else:
                business_success = 0
        else:
            business_success = 0

        return (
            transport_success,
            http_success,
            business_success,
            response_code,
            response_message,
            parsed_body,
        )

    def save_response(
        self,
        request_index: int,
        scenario_name: str,
        payload: Dict[str, Any],
        status: int,
        parsed_body: Any,
        error: str,
    ) -> str:
        """
        保存完整响应，便于后续排查异常样本和核对业务结果。
        """
        ensure_dir(self.responses_dir)
        response_path = (
            self.responses_dir / f"response_{request_index:05d}_{scenario_name}.json"
        )
        record = {
            "request_index": request_index,
            "scenario_name": scenario_name,
            "status": status,
            "error": error,
            "request": payload,
            "response": parsed_body,
        }
        with response_path.open("w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)
        return str(response_path)

    async def send_one(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        request_index: int,
        payload: Dict[str, Any],
        scenario_name: str,
    ) -> None:
        """
        发送单条请求。

        后续最常改的点一般在这里：
        1. 请求头注入
        2. trace_id 注入
        3. 响应体校验
        4. 错误码分类统计
        """
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        uid = str(payload.get("uid", ""))
        history_len = len(payload.get("history", []))
        status = 0
        transport_success = 0
        http_success = 0
        business_success = 0
        final_success = 0
        response_bytes = 0
        response_code = ""
        response_message = ""
        error = ""
        parsed_body: Any = None

        start_ts = now_ms()

        async with semaphore:
            try:
                async with session.post(
                    self.endpoint,
                    data=request_body,
                    headers=self.headers,
                ) as response:
                    content = await response.read()
                    response_text = content.decode("utf-8", errors="ignore")
                    status = response.status
                    response_bytes = len(content)
                    (
                        transport_success,
                        http_success,
                        business_success,
                        response_code,
                        response_message,
                        parsed_body,
                    ) = self.parse_business_result(status, response_text)
                    final_success = (
                        1
                        if transport_success and http_success and business_success
                        else 0
                    )
            except Exception as exc:
                error = str(exc)

        end_ts = now_ms()
        response_file = self.save_response(
            request_index=request_index,
            scenario_name=scenario_name,
            payload=payload,
            status=status,
            parsed_body=parsed_body,
            error=error,
        )

        self.metrics.append(
            RequestMetric(
                request_index=request_index,
                scenario_name=scenario_name,
                uid=uid,
                history_len=history_len,
                start_ts_ms=start_ts,
                end_ts_ms=end_ts,
                latency_ms=float(end_ts - start_ts),
                status=status,
                transport_success=transport_success,
                http_success=http_success,
                business_success=business_success,
                final_success=final_success,
                request_bytes=len(request_body),
                response_bytes=response_bytes,
                response_code=response_code,
                response_message=response_message[:200],
                response_file=response_file,
                error=error,
            )
        )

    async def run(self) -> List[RequestMetric]:
        loaded_requests = self.load_requests()
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        connector = aiohttp.TCPConnector(limit=0, ssl=False)
        semaphore = asyncio.Semaphore(self.concurrency)

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            tasks = []
            for idx, file_name, payload in loaded_requests:
                scenario_name = self.detect_scenario(file_name)
                tasks.append(
                    self.send_one(
                        session=session,
                        semaphore=semaphore,
                        request_index=idx,
                        payload=payload,
                        scenario_name=scenario_name,
                    )
                )
            await asyncio.gather(*tasks)

        self.metrics.sort(key=lambda item: item.request_index)
        return self.metrics


# =========================
# 核心逻辑区 4：结果统计与输出
# =========================


class ReportWriter:
    """
    输出压测结果。

    输出内容：
    1. request_metrics.csv：每条请求明细
    2. npu_metrics.csv：NPU 采样明细
    3. summary.csv：总体汇总
    """

    def __init__(
        self,
        output_dir: Path,
        request_metrics: List[RequestMetric],
        npu_metrics: List[NpuMetric],
    ) -> None:
        self.output_dir = output_dir
        self.request_metrics = request_metrics
        self.npu_metrics = npu_metrics

    def write_csv(self) -> None:
        ensure_dir(self.output_dir)

        request_csv = self.output_dir / "request_metrics.csv"
        with request_csv.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "request_index",
                "scenario_name",
                "uid",
                "history_len",
                "start_ts_ms",
                "end_ts_ms",
                "latency_ms",
                "status",
                "transport_success",
                "http_success",
                "business_success",
                "final_success",
                "request_bytes",
                "response_bytes",
                "response_code",
                "response_message",
                "response_file",
                "error",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.request_metrics:
                writer.writerow(asdict(item))

        npu_csv = self.output_dir / "npu_metrics.csv"
        with npu_csv.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "ts_ms",
                "device_id",
                "used_mb",
                "total_mb",
                "utilization_pct",
                "raw",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.npu_metrics:
                writer.writerow(asdict(item))

    def summarize(self) -> Dict[str, Any]:
        """
        计算总体指标。

        当前重点包括：
        1. 吞吐量 QPS
        2. 平均延迟和 P50/P90/P95/P99
        3. 成功率
        4. 平均请求大小
        5. NPU 显存占用区间
        """
        if not self.request_metrics:
            return {
                "total_requests": 0,
                "transport_success_rate_pct": 0.0,
                "http_success_rate_pct": 0.0,
                "business_success_rate_pct": 0.0,
                "success_rate_pct": 0.0,
                "npu_sample_count": 0,
                "npu_valid_sample_count": 0,
                "throughput_qps": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p90_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "avg_history_len": 0.0,
                "avg_request_bytes": 0.0,
                "avg_response_bytes": 0.0,
                "npu_used_mb_min": None,
                "npu_used_mb_avg": None,
                "npu_used_mb_max": None,
                "npu_total_mb": None,
                "npu_used_gb_min": None,
                "npu_used_gb_avg": None,
                "npu_used_gb_max": None,
                "npu_total_gb": None,
            }

        latencies = [item.latency_ms for item in self.request_metrics]
        histories = [item.history_len for item in self.request_metrics]
        transport_successes = [item.transport_success for item in self.request_metrics]
        http_successes = [item.http_success for item in self.request_metrics]
        business_successes = [item.business_success for item in self.request_metrics]
        final_successes = [item.final_success for item in self.request_metrics]
        request_sizes = [item.request_bytes for item in self.request_metrics]
        response_sizes = [item.response_bytes for item in self.request_metrics]

        start_ts = min(item.start_ts_ms for item in self.request_metrics)
        end_ts = max(item.end_ts_ms for item in self.request_metrics)
        wall_time_sec = max((end_ts - start_ts) / 1000.0, 0.001)

        npu_used_values = [
            item.used_mb for item in self.npu_metrics if item.used_mb is not None
        ]
        npu_total_values = [
            item.total_mb for item in self.npu_metrics if item.total_mb is not None
        ]

        npu_used_mb_min = min(npu_used_values) if npu_used_values else None
        npu_used_mb_avg = (
            round(statistics.mean(npu_used_values), 2) if npu_used_values else None
        )
        npu_used_mb_max = max(npu_used_values) if npu_used_values else None
        npu_total_mb = max(npu_total_values) if npu_total_values else None

        return {
            "total_requests": len(self.request_metrics),
            "transport_success_rate_pct": round(
                sum(transport_successes) / len(transport_successes) * 100, 2
            ),
            "http_success_rate_pct": round(
                sum(http_successes) / len(http_successes) * 100, 2
            ),
            "business_success_rate_pct": round(
                sum(business_successes) / len(business_successes) * 100, 2
            ),
            "success_rate_pct": round(
                sum(final_successes) / len(final_successes) * 100, 2
            ),
            "npu_sample_count": len(self.npu_metrics),
            "npu_valid_sample_count": len(npu_used_values),
            "throughput_qps": round(len(self.request_metrics) / wall_time_sec, 2),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p50_latency_ms": round(percentile(latencies, 50), 2),
            "p90_latency_ms": round(percentile(latencies, 90), 2),
            "p95_latency_ms": round(percentile(latencies, 95), 2),
            "p99_latency_ms": round(percentile(latencies, 99), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_history_len": round(statistics.mean(histories), 2),
            "avg_request_bytes": round(statistics.mean(request_sizes), 2),
            "avg_response_bytes": round(statistics.mean(response_sizes), 2),
            "npu_used_mb_min": npu_used_mb_min,
            "npu_used_mb_avg": npu_used_mb_avg,
            "npu_used_mb_max": npu_used_mb_max,
            "npu_total_mb": npu_total_mb,
            "npu_used_gb_min": mb_to_gb(npu_used_mb_min),
            "npu_used_gb_avg": mb_to_gb(npu_used_mb_avg),
            "npu_used_gb_max": mb_to_gb(npu_used_mb_max),
            "npu_total_gb": mb_to_gb(npu_total_mb),
        }

    def summarize_by_scenario(self) -> List[Dict[str, Any]]:
        """
        按场景分组统计，便于对比不同 history 长度带来的性能差异。
        """
        grouped: Dict[str, List[RequestMetric]] = {}

        for item in self.request_metrics:
            grouped.setdefault(item.scenario_name, []).append(item)

        scenario_rows: List[Dict[str, Any]] = []
        for scenario_name, items in grouped.items():
            latencies = [item.latency_ms for item in items]
            final_successes = [item.final_success for item in items]
            histories = [item.history_len for item in items]

            start_ts = min(item.start_ts_ms for item in items)
            end_ts = max(item.end_ts_ms for item in items)
            wall_time_sec = max((end_ts - start_ts) / 1000.0, 0.001)

            scenario_rows.append(
                {
                    "scenario_name": scenario_name,
                    "total_requests": len(items),
                    "success_rate_pct": round(
                        sum(final_successes) / len(final_successes) * 100, 2
                    ),
                    "throughput_qps": round(len(items) / wall_time_sec, 2),
                    "avg_latency_ms": round(statistics.mean(latencies), 2),
                    "p95_latency_ms": round(percentile(latencies, 95), 2),
                    "p99_latency_ms": round(percentile(latencies, 99), 2),
                    "avg_history_len": round(statistics.mean(histories), 2),
                }
            )

        scenario_rows.sort(key=lambda item: item["scenario_name"])
        return scenario_rows

    def write_summary_csv(self, summary: Dict[str, Any]) -> None:
        summary_csv = self.output_dir / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

    def write_scenario_summary_csv(self, rows: List[Dict[str, Any]]) -> None:
        scenario_csv = self.output_dir / "scenario_summary.csv"
        with scenario_csv.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "scenario_name",
                "total_requests",
                "success_rate_pct",
                "throughput_qps",
                "avg_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "avg_history_len",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def print_summary(
        self, summary: Dict[str, Any], scenario_rows: List[Dict[str, Any]]
    ) -> None:
        total_rows = [
            ["total_requests", summary["total_requests"]],
            ["transport_success_rate_pct", summary["transport_success_rate_pct"]],
            ["http_success_rate_pct", summary["http_success_rate_pct"]],
            ["business_success_rate_pct", summary["business_success_rate_pct"]],
            ["success_rate_pct", summary["success_rate_pct"]],
            ["npu_sample_count", summary["npu_sample_count"]],
            ["npu_valid_sample_count", summary["npu_valid_sample_count"]],
            ["throughput_qps", summary["throughput_qps"]],
            ["avg_latency_ms", summary["avg_latency_ms"]],
            ["p50_latency_ms", summary["p50_latency_ms"]],
            ["p90_latency_ms", summary["p90_latency_ms"]],
            ["p95_latency_ms", summary["p95_latency_ms"]],
            ["p99_latency_ms", summary["p99_latency_ms"]],
            ["max_latency_ms", summary["max_latency_ms"]],
            ["avg_history_len", summary["avg_history_len"]],
            ["avg_request_bytes", summary["avg_request_bytes"]],
            ["avg_response_bytes", summary["avg_response_bytes"]],
            ["npu_used_mb_min", summary["npu_used_mb_min"]],
            ["npu_used_mb_avg", summary["npu_used_mb_avg"]],
            ["npu_used_mb_max", summary["npu_used_mb_max"]],
            ["npu_total_mb", summary["npu_total_mb"]],
            ["npu_used_gb_min", summary["npu_used_gb_min"]],
            ["npu_used_gb_avg", summary["npu_used_gb_avg"]],
            ["npu_used_gb_max", summary["npu_used_gb_max"]],
            ["npu_total_gb", summary["npu_total_gb"]],
        ]

        print("\n=== Overall Summary ===")
        print_table(["metric", "value"], total_rows)

        if scenario_rows:
            print("\n=== Scenario Summary ===")
            rows = []
            for item in scenario_rows:
                rows.append(
                    [
                        item["scenario_name"],
                        item["total_requests"],
                        item["success_rate_pct"],
                        item["throughput_qps"],
                        item["avg_latency_ms"],
                        item["p95_latency_ms"],
                        item["p99_latency_ms"],
                        item["avg_history_len"],
                    ]
                )
            print_table(
                [
                    "scenario",
                    "total",
                    "success_rate_pct",
                    "qps",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "p99_latency_ms",
                    "avg_history_len",
                ],
                rows,
            )


def print_metric_explanations() -> None:
    """
    输出指标口径说明，避免不同人对统计结果的理解不一致。
    """
    rows = [
        ["total_requests", "本次压测总请求数"],
        [
            "transport_success_rate_pct",
            "成功拿到响应的请求占比，不含连接失败和超时异常",
        ],
        ["http_success_rate_pct", "HTTP 状态码为 2xx 的请求占比"],
        [
            "business_success_rate_pct",
            "按响应体 success/code 字段明确判断为业务成功的请求占比；仅 HTTP 200 不算成功",
        ],
        [
            "success_rate_pct",
            "最终成功率，只有成功拿到业务成功返回才算成功；仅 HTTP 200 不算成功",
        ],
        ["npu_sample_count", "NPU 采样总次数，用于判断采样流程是否真正执行"],
        [
            "npu_valid_sample_count",
            "成功解析出显存占用的采样次数；为 0 时，npu_used_mb_* 通常会是 None",
        ],
        ["throughput_qps", "每秒处理请求数，反映整体吞吐能力"],
        ["avg_latency_ms", "平均响应延迟，单位毫秒"],
        ["p50_latency_ms", "50 分位延迟，中位数"],
        ["p90_latency_ms", "90 分位延迟，90% 请求不超过该值"],
        ["p95_latency_ms", "95 分位延迟，常用核心性能指标"],
        ["p99_latency_ms", "99 分位延迟，反映长尾请求性能"],
        ["max_latency_ms", "最大延迟，反映最慢请求"],
        ["avg_history_len", "平均历史行为长度，反映请求负载规模"],
        ["avg_request_bytes", "平均请求体大小，单位字节"],
        ["avg_response_bytes", "平均响应体大小，单位字节"],
        ["npu_used_mb_min", "压测期间 NPU 显存最小占用，单位 MB"],
        ["npu_used_mb_avg", "压测期间 NPU 显存平均占用，单位 MB"],
        ["npu_used_mb_max", "压测期间 NPU 显存最大占用，单位 MB"],
        ["npu_used_gb_min", "压测期间 NPU 显存最小占用，单位 GB"],
        ["npu_used_gb_avg", "压测期间 NPU 显存平均占用，单位 GB"],
        ["npu_used_gb_max", "压测期间 NPU 显存最大占用，单位 GB"],
        [
            "npu_total_mb",
            "NPU 总显存容量，单位 MB；如果采样命令不可用或解析失败会显示 None",
        ],
        ["npu_total_gb", "NPU 总显存容量，单位 GB"],
        [
            "npu-device-id",
            "指定采集哪张 NPU 卡；默认会拼接为 npu-smi info -t usages -i <device_id>",
        ],
        ["response_file", "每次请求对应的完整响应落盘文件，便于排查"],
    ]
    print("\n=== Metric Explanations ===")
    print_table(["metric", "meaning"], rows)


# =========================
# 主流程编排区
# =========================


async def main_async(args: argparse.Namespace) -> None:
    """
    主流程：
    1. 生成请求
    2. 启动 NPU 采样
    3. 发起并发压测
    4. 输出总体结果和分场景结果
    """
    base_dir = Path(args.output_dir).resolve()
    requests_dir = base_dir / "requests"
    responses_dir = base_dir / "responses"
    reports_dir = base_dir / "reports"

    ensure_dir(base_dir)
    ensure_dir(requests_dir)
    ensure_dir(responses_dir)
    ensure_dir(reports_dir)

    metadata_template = {}
    if args.metadata_file:
        with open(args.metadata_file, "r", encoding="utf-8") as file:
            metadata_template = json.load(file)

    generator = RequestGenerator(
        output_dir=requests_dir,
        total_requests=args.total_requests,
        seed=args.seed,
        metadata_template=metadata_template,
    )
    generated_files = generator.generate()
    print(f"generated_requests={len(generated_files)}")
    print(f"requests_dir={requests_dir}")

    headers = {"Content-Type": "application/json"}
    if args.auth_token:
        headers["Authorization"] = f"Bearer {args.auth_token}"

    sampler = AscendNpuSampler(
        device_id=args.npu_device_id,
        interval_sec=args.npu_sample_interval_sec,
        enabled=not args.disable_npu_sampler,
        command=args.npu_command,
    )

    tester = LoadTester(
        endpoint=args.endpoint,
        headers=headers,
        concurrency=args.concurrency,
        timeout_sec=args.timeout_sec,
        requests_dir=requests_dir,
        responses_dir=responses_dir,
    )

    sampler_task = (
        asyncio.create_task(sampler.run()) if not args.disable_npu_sampler else None
    )

    try:
        request_metrics = await tester.run()
    finally:
        sampler.stop()
        if sampler_task is not None:
            await sampler_task

    reporter = ReportWriter(
        output_dir=reports_dir,
        request_metrics=request_metrics,
        npu_metrics=sampler.records,
    )
    reporter.write_csv()

    summary = reporter.summarize()
    scenario_rows = reporter.summarize_by_scenario()

    reporter.write_summary_csv(summary)
    reporter.write_scenario_summary_csv(scenario_rows)
    reporter.print_summary(summary, scenario_rows)
    print_metric_explanations()

    print("\n=== Output Files ===")
    print(f"request_metrics_csv: {reports_dir / 'request_metrics.csv'}")
    print(f"responses_dir:         {responses_dir}")
    print(f"npu_metrics_csv:     {reports_dir / 'npu_metrics.csv'}")
    print(f"summary_csv:         {reports_dir / 'summary.csv'}")
    print(f"scenario_summary:    {reports_dir / 'scenario_summary.csv'}")


# =========================
# 参数入口区
# =========================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference service performance test tool"
    )

    parser.add_argument("--endpoint", required=True, help="外层推理服务 HTTP 地址")
    parser.add_argument("--output-dir", default="./perf_output", help="输出目录")
    parser.add_argument(
        "--total-requests", type=int, default=300, help="生成并发送的总请求数"
    )
    parser.add_argument("--concurrency", type=int, default=20, help="并发请求数")
    parser.add_argument(
        "--timeout-sec", type=float, default=10.0, help="HTTP 超时时间，单位秒"
    )
    parser.add_argument(
        "--seed", type=int, default=20260720, help="随机种子，便于复现实验"
    )
    parser.add_argument("--auth-token", default="", help="可选 Bearer Token")
    parser.add_argument("--metadata-file", default="", help="固定 _metadata 模板文件")

    parser.add_argument(
        "--disable-npu-sampler", action="store_true", help="关闭 Ascend NPU 指标采样"
    )
    parser.add_argument("--npu-device-id", default="0", help="Ascend 设备号")
    parser.add_argument(
        "--npu-sample-interval-sec", type=float, default=1.0, help="NPU 采样周期"
    )
    parser.add_argument(
        "--npu-command",
        default="npu-smi info",
        help="NPU 采样命令；默认自动补成 npu-smi info -t usages -i <device_id>",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    """python test_infer_performance.py \
  --endpoint http://127.0.0.1:8080/infer \
  --total-requests 600 \
  --concurrency 50 \
  --output-dir ./perf_run_01"""
    main()
