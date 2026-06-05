import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return round(ordered[low], 2)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 2)


def _extract_token_usage(payload: Any) -> Tuple[int, int, int]:
    """
    尽力从不同模型返回结构中提取 token 使用量。
    """
    if not isinstance(payload, dict):
        return 0, 0, 0

    def _from_map(data: Dict[str, Any]) -> Tuple[int, int, int]:
        input_keys = ["input_tokens", "prompt_tokens", "prompt_token_count"]
        output_keys = ["output_tokens", "completion_tokens", "candidates_token_count"]
        total_keys = ["total_tokens", "total_token_count"]
        input_v = next((int(data[k]) for k in input_keys if k in data and str(data[k]).isdigit()), 0)
        output_v = next((int(data[k]) for k in output_keys if k in data and str(data[k]).isdigit()), 0)
        total_v = next((int(data[k]) for k in total_keys if k in data and str(data[k]).isdigit()), 0)
        if total_v == 0:
            total_v = input_v + output_v
        return input_v, output_v, total_v

    # 常见位置：usage_metadata / token_usage / usage
    for key in ["usage_metadata", "token_usage", "usage"]:
        usage = payload.get(key)
        if isinstance(usage, dict):
            i, o, t = _from_map(usage)
            if t > 0:
                return i, o, t

    # 一些框架把 usage 放在 response_metadata 里
    response_metadata = payload.get("response_metadata")
    if isinstance(response_metadata, dict):
        for key in ["token_usage", "usage_metadata", "usage"]:
            usage = response_metadata.get(key)
            if isinstance(usage, dict):
                i, o, t = _from_map(usage)
                if t > 0:
                    return i, o, t

    return 0, 0, 0


class MetricsCollector:
    """
    轻量指标收集器（进程内聚合 + 按日 jsonl 落盘）
    """

    TRACKED_NODES = {"Reader", "Analysis", "Modeling", "Coder", "Review", "Writing", "Export", "Supervisor"}

    def __init__(self, base_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._base_dir = base_dir or (Path(__file__).resolve().parents[2] / "logs" / "metrics")
        self._node_stats: Dict[str, Dict[str, Any]] = {}
        self._run_state: Dict[str, Dict[str, Any]] = {}

    def _stats_key(self, node: str, model_id: str) -> str:
        return f"{node}::{model_id}"

    def _new_stat(self, node: str, model_id: str) -> Dict[str, Any]:
        return {
            "node": node,
            "model_id": model_id,
            "invocations": 0,
            "failures": 0,
            "retries": 0,
            "token_input": 0,
            "token_output": 0,
            "token_total": 0,
            "durations_ms": [],
        }

    def _ensure_node(self, node: str, model_id: str) -> Dict[str, Any]:
        key = self._stats_key(node, model_id)
        node_stats = self._node_stats.get(key)
        if node_stats is None:
            node_stats = self._new_stat(node, model_id)
            self._node_stats[key] = node_stats
        return node_stats

    def _append_event(self, event: Dict[str, Any]) -> None:
        day_dir = self._base_dir / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / "events.jsonl"
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def start_run(self, run_id: str, thread_id: str, model_id: str) -> None:
        with self._lock:
            self._run_state[run_id] = {
                "thread_id": thread_id,
                "model_id": model_id,
                "nodes": {},
                "started_at": time.time(),
            }
            self._append_event(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "kind": "run_start",
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "model_id": model_id,
                }
            )

    def start_node(self, run_id: str, node: str) -> None:
        if node not in self.TRACKED_NODES:
            return
        with self._lock:
            run = self._run_state.get(run_id)
            if not run:
                return
            node_state = run["nodes"].get(node) or {"starts": 0, "last_start": None}
            node_state["starts"] += 1
            node_state["last_start"] = time.time()
            run["nodes"][node] = node_state

            model_id = str(run.get("model_id", "unknown_model"))
            stats = self._ensure_node(node, model_id)
            stats["invocations"] += 1
            retried = node_state["starts"] > 1
            if retried:
                stats["retries"] += 1

            self._append_event(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "kind": "node_start",
                    "run_id": run_id,
                    "thread_id": run["thread_id"],
                    "model_id": run["model_id"],
                    "node": node,
                    "retried": retried,
                }
            )

    def end_node(self, run_id: str, node: str, success: bool, error: str = "") -> None:
        if node not in self.TRACKED_NODES:
            return
        with self._lock:
            run = self._run_state.get(run_id)
            if not run:
                return
            node_state = run["nodes"].get(node) or {}
            start = node_state.get("last_start")
            duration_ms = round((time.time() - start) * 1000, 2) if start else 0.0

            model_id = str(run.get("model_id", "unknown_model"))
            stats = self._ensure_node(node, model_id)
            if not success:
                stats["failures"] += 1
            if duration_ms > 0:
                stats["durations_ms"].append(duration_ms)
                # 保持轻量内存占用
                if len(stats["durations_ms"]) > 2000:
                    stats["durations_ms"] = stats["durations_ms"][-2000:]

            self._append_event(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "kind": "node_end",
                    "run_id": run_id,
                    "thread_id": run["thread_id"],
                    "model_id": run["model_id"],
                    "node": node,
                    "success": success,
                    "duration_ms": duration_ms,
                    "error": error[:500] if error else "",
                }
            )

    def record_token_usage(self, run_id: str, node: str, usage_payload: Any) -> None:
        if node not in self.TRACKED_NODES:
            return
        i, o, t = _extract_token_usage(usage_payload if isinstance(usage_payload, dict) else {})
        if t <= 0:
            return
        with self._lock:
            run = self._run_state.get(run_id)
            if not run:
                return
            model_id = str(run.get("model_id", "unknown_model"))
            stats = self._ensure_node(node, model_id)
            stats["token_input"] += i
            stats["token_output"] += o
            stats["token_total"] += t
            self._append_event(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "kind": "token_usage",
                    "run_id": run_id,
                    "thread_id": run["thread_id"],
                    "model_id": run["model_id"],
                    "node": node,
                    "token_input": i,
                    "token_output": o,
                    "token_total": t,
                }
            )

    def finish_run(self, run_id: str, status: str, stage: str = "") -> None:
        with self._lock:
            run = self._run_state.pop(run_id, None)
            if not run:
                return
            self._append_event(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "kind": "run_end",
                    "run_id": run_id,
                    "thread_id": run["thread_id"],
                    "model_id": run["model_id"],
                    "status": status,
                    "stage": stage,
                    "duration_ms": round((time.time() - run["started_at"]) * 1000, 2),
                }
            )

    def _iter_event_files(self) -> List[Path]:
        if not self._base_dir.exists():
            return []
        event_files: List[Path] = []
        for day_dir in sorted(self._base_dir.iterdir(), key=lambda p: p.name):
            if not day_dir.is_dir():
                continue
            event_file = day_dir / "events.jsonl"
            if event_file.is_file():
                event_files.append(event_file)
        return event_files

    def _aggregate_from_logs(self) -> Dict[str, Dict[str, Any]]:
        stats_map: Dict[str, Dict[str, Any]] = {}
        for event_file in self._iter_event_files():
            try:
                with event_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                        except Exception:
                            continue
                        kind = str(event.get("kind", "")).strip()
                        node = str(event.get("node", "")).strip()
                        if node not in self.TRACKED_NODES:
                            continue
                        model_id = str(event.get("model_id", "unknown_model") or "unknown_model")
                        key = self._stats_key(node, model_id)
                        stat = stats_map.get(key)
                        if stat is None:
                            stat = self._new_stat(node, model_id)
                            stats_map[key] = stat

                        if kind == "node_start":
                            stat["invocations"] += 1
                            if bool(event.get("retried")):
                                stat["retries"] += 1
                        elif kind == "node_end":
                            if not bool(event.get("success", True)):
                                stat["failures"] += 1
                            duration = event.get("duration_ms")
                            if isinstance(duration, (int, float)) and duration > 0:
                                stat["durations_ms"].append(float(duration))
                                if len(stat["durations_ms"]) > 5000:
                                    stat["durations_ms"] = stat["durations_ms"][-5000:]
                        elif kind == "token_usage":
                            for k, target in (
                                ("token_input", "token_input"),
                                ("token_output", "token_output"),
                                ("token_total", "token_total"),
                            ):
                                val = event.get(k)
                                if isinstance(val, (int, float)):
                                    stat[target] += int(val)
            except OSError:
                continue
        return stats_map

    def _snapshot_memory_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            snap: Dict[str, Dict[str, Any]] = {}
            for key, stat in self._node_stats.items():
                snap[key] = {
                    "node": str(stat.get("node", "")),
                    "model_id": str(stat.get("model_id", "unknown_model")),
                    "invocations": int(stat.get("invocations", 0)),
                    "failures": int(stat.get("failures", 0)),
                    "retries": int(stat.get("retries", 0)),
                    "token_input": int(stat.get("token_input", 0)),
                    "token_output": int(stat.get("token_output", 0)),
                    "token_total": int(stat.get("token_total", 0)),
                    "durations_ms": list(stat.get("durations_ms", [])),
                }
            return snap

    def _build_dashboard(self, stats_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        total_invocations = 0
        total_failures = 0
        total_retries = 0
        total_tokens = 0

        model_rollup: Dict[str, Dict[str, Any]] = {}

        for _, stat in sorted(stats_map.items(), key=lambda item: (item[1]["model_id"], item[1]["node"])):
            node = str(stat["node"])
            model_id = str(stat["model_id"])
            invocations = int(stat["invocations"])
            failures = int(stat["failures"])
            retries = int(stat["retries"])
            durations = list(stat["durations_ms"])
            token_input = int(stat["token_input"])
            token_output = int(stat["token_output"])
            token_total = int(stat["token_total"])

            total_invocations += invocations
            total_failures += failures
            total_retries += retries
            total_tokens += token_total

            failure_rate = round((failures / invocations) if invocations else 0.0, 4)
            retry_rate = round((retries / invocations) if invocations else 0.0, 4)

            rows.append(
                {
                    "node": node,
                    "model_id": model_id,
                    "invocations": invocations,
                    "failures": failures,
                    "failure_rate": failure_rate,
                    "retries": retries,
                    "retry_rate": retry_rate,
                    "latency_avg_ms": round((sum(durations) / len(durations)) if durations else 0.0, 2),
                    "latency_p50_ms": _percentile(durations, 0.50),
                    "latency_p95_ms": _percentile(durations, 0.95),
                    "token_input": token_input,
                    "token_output": token_output,
                    "token_total": token_total,
                    "avg_token_per_invocation": round((token_total / invocations) if invocations else 0.0, 2),
                }
            )

            bucket = model_rollup.get(model_id)
            if bucket is None:
                bucket = {
                    "model_id": model_id,
                    "invocations": 0,
                    "failures": 0,
                    "retries": 0,
                    "token_total": 0,
                }
                model_rollup[model_id] = bucket
            bucket["invocations"] += invocations
            bucket["failures"] += failures
            bucket["retries"] += retries
            bucket["token_total"] += token_total

        model_rows: List[Dict[str, Any]] = []
        for model_id, bucket in sorted(model_rollup.items(), key=lambda item: item[0]):
            invocations = int(bucket["invocations"])
            failures = int(bucket["failures"])
            retries = int(bucket["retries"])
            token_total = int(bucket["token_total"])
            model_rows.append(
                {
                    "model_id": model_id,
                    "invocations": invocations,
                    "failures": failures,
                    "retries": retries,
                    "failure_rate": round((failures / invocations) if invocations else 0.0, 4),
                    "retry_rate": round((retries / invocations) if invocations else 0.0, 4),
                    "token_total": token_total,
                }
            )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "nodes": len(rows),
                "models": len(model_rows),
                "invocations": total_invocations,
                "failures": total_failures,
                "retries": total_retries,
                "token_total": total_tokens,
                "failure_rate": round((total_failures / total_invocations) if total_invocations else 0.0, 4),
                "retry_rate": round((total_retries / total_invocations) if total_invocations else 0.0, 4),
            },
            "models": model_rows,
            "nodes": rows,
        }

    def dashboard(self) -> Dict[str, Any]:
        history_stats = self._aggregate_from_logs()
        if history_stats:
            return self._build_dashboard(history_stats)
        return self._build_dashboard(self._snapshot_memory_stats())


metrics_collector = MetricsCollector()
