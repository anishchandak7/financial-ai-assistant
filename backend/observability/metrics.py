from typing import Dict, Any, List
from threading import Lock

class MetricsRegistry:
    """Thread-safe in-memory metrics collector and Prometheus-compatible exporter."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._http_requests_total: Dict[str, int] = {}
        self._agent_invocations_total: Dict[str, int] = {}
        self._agent_duration_seconds: Dict[str, List[float]] = {}
        self._guardrail_blocks_total: Dict[str, int] = {}
        self._tool_executions_total: Dict[str, int] = {}
        self._tool_failures_total: Dict[str, int] = {}
        self._llm_tokens: Dict[str, Dict[str, int]] = {
            "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        self._rag_ingestions_total: Dict[str, int] = {}
        self._rag_retrievals_total: Dict[str, int] = {}
        self._rag_eval_scores: Dict[str, List[float]] = {}

    def record_http_request(self, method: str, path: str, status_code: int, duration_ms: float):
        with self._lock:
            key = f"{method} {path} {status_code}"
            self._http_requests_total[key] = self._http_requests_total.get(key, 0) + 1

    def record_agent_execution(self, agent_name: str, duration_sec: float, success: bool = True):
        with self._lock:
            self._agent_invocations_total[agent_name] = self._agent_invocations_total.get(agent_name, 0) + 1
            if agent_name not in self._agent_duration_seconds:
                self._agent_duration_seconds[agent_name] = []
            # Keep rolling window of last 500 execution durations
            if len(self._agent_duration_seconds[agent_name]) >= 500:
                self._agent_duration_seconds[agent_name].pop(0)
            self._agent_duration_seconds[agent_name].append(duration_sec)

    def record_guardrail_block(self, category: str):
        with self._lock:
            self._guardrail_blocks_total[category] = self._guardrail_blocks_total.get(category, 0) + 1

    def record_tool_call(self, tool_name: str, duration_sec: float, success: bool = True):
        with self._lock:
            self._tool_executions_total[tool_name] = self._tool_executions_total.get(tool_name, 0) + 1
            if not success:
                self._tool_failures_total[tool_name] = self._tool_failures_total.get(tool_name, 0) + 1

    def record_token_usage(self, model_name: str, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            if model_name not in self._llm_tokens:
                self._llm_tokens[model_name] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._llm_tokens[model_name]["prompt_tokens"] += prompt_tokens
            self._llm_tokens[model_name]["completion_tokens"] += completion_tokens
            self._llm_tokens[model_name]["total_tokens"] += (prompt_tokens + completion_tokens)
            self._llm_tokens["total"]["prompt_tokens"] += prompt_tokens
            self._llm_tokens["total"]["completion_tokens"] += completion_tokens
            self._llm_tokens["total"]["total_tokens"] += (prompt_tokens + completion_tokens)

    def record_rag_ingestion(self, collection: str, file_type: str, chunk_count: int, success: bool = True):
        with self._lock:
            key = f"{collection} {file_type} {'success' if success else 'failed'}"
            self._rag_ingestions_total[key] = self._rag_ingestions_total.get(key, 0) + chunk_count

    def record_rag_retrieval(self, collection: str, chunk_count: int):
        with self._lock:
            self._rag_retrievals_total[collection] = self._rag_retrievals_total.get(collection, 0) + chunk_count

    def record_rag_eval(self, metric_name: str, score: float):
        with self._lock:
            if metric_name not in self._rag_eval_scores:
                self._rag_eval_scores[metric_name] = []
            if len(self._rag_eval_scores[metric_name]) >= 200:
                self._rag_eval_scores[metric_name].pop(0)
            self._rag_eval_scores[metric_name].append(score)

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured JSON summary of metrics."""
        with self._lock:
            agent_latencies = {}
            for agent, durations in self._agent_duration_seconds.items():
                if durations:
                    sorted_durations = sorted(durations)
                    p50_idx = int(len(sorted_durations) * 0.5)
                    p95_idx = min(int(len(sorted_durations) * 0.95), len(sorted_durations) - 1)
                    agent_latencies[agent] = {
                        "count": len(durations),
                        "avg_ms": round((sum(durations) / len(durations)) * 1000, 2),
                        "p50_ms": round(sorted_durations[p50_idx] * 1000, 2),
                        "p95_ms": round(sorted_durations[p95_idx] * 1000, 2)
                    }

            return {
                "http_requests": self._http_requests_total,
                "agent_invocations": self._agent_invocations_total,
                "agent_latencies": agent_latencies,
                "guardrail_blocks": self._guardrail_blocks_total,
                "tool_executions": self._tool_executions_total,
                "tool_failures": self._tool_failures_total,
                "llm_token_usage": self._llm_tokens
            }

    def export_prometheus(self) -> str:
        """Exports metrics in standard Prometheus text format."""
        lines = []
        with self._lock:
            lines.append("# HELP financial_ai_http_requests_total Total number of HTTP requests")
            lines.append("# TYPE financial_ai_http_requests_total counter")
            for key, val in self._http_requests_total.items():
                parts = key.split(" ")
                if len(parts) == 3:
                    lines.append(f'financial_ai_http_requests_total{{method="{parts[0]}",path="{parts[1]}",status="{parts[2]}"}} {val}')

            lines.append("\n# HELP financial_ai_agent_invocations_total Total executions per agent")
            lines.append("# TYPE financial_ai_agent_invocations_total counter")
            for agent, count in self._agent_invocations_total.items():
                lines.append(f'financial_ai_agent_invocations_total{{agent="{agent}"}} {count}')

            lines.append("\n# HELP financial_ai_guardrail_blocks_total Intercepted malicious or out-of-bounds inputs")
            lines.append("# TYPE financial_ai_guardrail_blocks_total counter")
            for cat, count in self._guardrail_blocks_total.items():
                lines.append(f'financial_ai_guardrail_blocks_total{{category="{cat}"}} {count}')

            lines.append("\n# HELP financial_ai_tool_executions_total Tool calls executed")
            lines.append("# TYPE financial_ai_tool_executions_total counter")
            for tool, count in self._tool_executions_total.items():
                lines.append(f'financial_ai_tool_executions_total{{tool="{tool}"}} {count}')

            lines.append("\n# HELP financial_ai_llm_tokens_total Token usage by model")
            lines.append("# TYPE financial_ai_llm_tokens_total counter")
            for model, usage in self._llm_tokens.items():
                lines.append(f'financial_ai_llm_tokens_total{{model="{model}",type="prompt"}} {usage["prompt_tokens"]}')
                lines.append(f'financial_ai_llm_tokens_total{{model="{model}",type="completion"}} {usage["completion_tokens"]}')

        return "\n".join(lines) + "\n"

metrics = MetricsRegistry()
