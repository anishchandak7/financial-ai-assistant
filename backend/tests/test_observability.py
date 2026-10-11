import unittest
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from observability.metrics import MetricsRegistry
from observability.tracer import trace_span

class TestObservability(unittest.TestCase):

    def test_metrics_registry(self):
        """Verify metrics collector records and aggregates measurements."""
        registry = MetricsRegistry()

        # Record HTTP requests
        registry.record_http_request("POST", "/api/chat", 200, 150.5)
        registry.record_http_request("POST", "/api/chat", 200, 200.0)
        registry.record_http_request("GET", "/api/health", 200, 5.0)

        # Record Agent executions
        registry.record_agent_execution("Market Analysis Agent", 1.2, success=True)
        registry.record_agent_execution("Stock Analyzer Agent", 0.8, success=True)

        # Record Guardrail block
        registry.record_guardrail_block("prompt_injection")

        # Record Tool executions
        registry.record_tool_call("fetch_realtime_market_data", 0.35, success=True)
        registry.record_tool_call("document_lookup", 0.12, success=False)

        # Record Token usage
        registry.record_token_usage("gemini-3.8-flash", 500, 250)

        summary = registry.get_summary()
        self.assertEqual(summary["http_requests"]["POST /api/chat 200"], 2)
        self.assertEqual(summary["guardrail_blocks"]["prompt_injection"], 1)
        self.assertEqual(summary["agent_invocations"]["Market Analysis Agent"], 1)
        self.assertEqual(summary["tool_failures"]["document_lookup"], 1)
        self.assertEqual(summary["llm_token_usage"]["gemini-3.8-flash"]["total_tokens"], 750)

        # Verify Prometheus export format
        prom_text = registry.export_prometheus()
        self.assertIn("financial_ai_http_requests_total", prom_text)
        self.assertIn("financial_ai_agent_invocations_total", prom_text)
        self.assertIn("financial_ai_guardrail_blocks_total", prom_text)
        self.assertIn("financial_ai_llm_tokens_total", prom_text)

    def test_trace_span(self):
        """Verify execution spans complete and calculate elapsed time."""
        with trace_span("test_span", {"custom_attr": "value"}) as span:
            self.assertEqual(span.name, "test_span")
            self.assertEqual(span.attributes.get("custom_attr"), "value")

        self.assertIsNotNone(span.end_time)
        self.assertGreaterEqual(span.duration_ms, 0)
        self.assertIsNone(span.error)

    def test_langsmith_status_and_tracer(self):
        """Verify LangSmith status helper and tracer generation."""
        from observability.langsmith_tracer import check_langsmith_status, get_langsmith_tracer

        status = check_langsmith_status()
        self.assertIn("enabled", status)
        self.assertIn("project", status)
        self.assertIn("endpoint", status)
        self.assertEqual(status["project"], "financial-ai-assistant")
        self.assertEqual(status["endpoint"], "https://api.smith.langchain.com")

if __name__ == "__main__":
    unittest.main()
