"""Integration tests for multi-turn session memory workflows and API endpoints."""

import unittest
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from server import app
from memory.manager import agent_memory_manager
from agents.orchestrator.router import triage_and_route
from agents.orchestrator.state import GraphState


class TestMultiTurnMemory(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.session_id = "integration_test_session_99"
        agent_memory_manager.reset_session(self.session_id)

    def test_router_resolves_pronoun_from_session_memory(self):
        """Verify the triage router uses session memory to resolve pronouns like 'its' when analyzing a stock."""
        # Establish session memory that the user analyzed NVDA previously
        agent_memory_manager.add_turn(
            session_id=self.session_id,
            agent_name="Stock Analyzer Agent",
            user_query="Analyze NVDA momentum",
            agent_response="NVDA RSI is 74.2.",
            metadata={"ticker": "NVDA"}
        )

        state: GraphState = {
            "session_id": self.session_id,
            "input_query": "What is its 200-day moving average and support level?"
        }

        res = triage_and_route(state)
        self.assertFalse(res.get("is_blocked"))
        self.assertEqual(res.get("current_intent"), "Stock Analyzer Agent")

    def test_api_chat_multi_turn_and_memory_inspection(self):
        """Verify /api/chat preserves session_id and per-agent memory is inspectable via API."""
        # Turn 1: Conversational greeting with user name
        payload1 = {
            "prompt": "Hello! My name is Arthur and I am an equity investor.",
            "session_id": self.session_id
        }
        res1 = self.client.post("/api/chat", json=payload1)
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertTrue(data1["success"])
        self.assertEqual(data1["session_id"], self.session_id)

        # Inspect session memory via endpoint
        mem_res = self.client.get(f"/api/chat/session/{self.session_id}/memory")
        self.assertEqual(mem_res.status_code, 200)
        mem_data = mem_res.json()
        self.assertEqual(mem_data["session_id"], self.session_id)
        self.assertIn("Conversational Agent", mem_data["agents"])
        self.assertEqual(mem_data["agents"]["Conversational Agent"]["turns_count"], 1)

        # Reset session endpoint
        reset_res = self.client.post(f"/api/chat/session/reset?session_id={self.session_id}")
        self.assertEqual(reset_res.status_code, 200)
        reset_data = reset_res.json()
        self.assertTrue(reset_data["cleared"])

        # Verify session memory is now empty
        mem_res_after = self.client.get(f"/api/chat/session/{self.session_id}/memory")
        self.assertEqual(mem_res_after.json()["agents"], {})


if __name__ == "__main__":
    unittest.main()
