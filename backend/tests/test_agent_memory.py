"""Unit and Integration Tests for Per-Agent Session Memory."""

import unittest
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from memory.manager import AgentMemoryManager, agent_memory_manager
from memory.models import MemoryTurn
from agents.tools.memory_tools import refer_agent_memory, set_current_session_id
from agents.specialists.conversational import ConversationalAgent


class TestAgentMemoryManager(unittest.TestCase):

    def setUp(self):
        self.manager = AgentMemoryManager(default_max_turns=3)
        self.session_id = "test_session_123"

    def test_session_and_agent_isolation(self):
        """Verify that different agents and sessions maintain strictly separated memories."""
        # Session 1: Stock Analyzer
        self.manager.add_turn(
            session_id=self.session_id,
            agent_name="Stock Analyzer Agent",
            user_query="What is NVDA RSI?",
            agent_response="NVDA RSI is 74.2.",
            metadata={"ticker": "NVDA"}
        )

        # Session 1: Finance Q&A
        self.manager.add_turn(
            session_id=self.session_id,
            agent_name="Finance Q&A Agent",
            user_query="What is a 401(k)?",
            agent_response="A 401(k) is an employer-sponsored retirement savings plan."
        )

        # Session 2: Stock Analyzer (different session)
        self.manager.add_turn(
            session_id="different_session_456",
            agent_name="Stock Analyzer Agent",
            user_query="What is TSLA MACD?",
            agent_response="TSLA MACD is bullish crossover."
        )

        # Assert isolated memories for Session 1
        stock_mem = self.manager.get_agent_memory(self.session_id, "Stock Analyzer Agent")
        qa_mem = self.manager.get_agent_memory(self.session_id, "Finance Q&A Agent")
        diff_stock_mem = self.manager.get_agent_memory("different_session_456", "Stock Analyzer Agent")

        self.assertEqual(len(stock_mem.turns), 1)
        self.assertEqual(stock_mem.turns[0].user_query, "What is NVDA RSI?")
        self.assertEqual(len(qa_mem.turns), 1)
        self.assertEqual(qa_mem.turns[0].user_query, "What is a 401(k)?")
        self.assertEqual(len(diff_stock_mem.turns), 1)
        self.assertEqual(diff_stock_mem.turns[0].user_query, "What is TSLA MACD?")

    def test_sliding_window_eviction(self):
        """Verify that memories strictly enforce the sliding-window turn limit to prevent prompt bloat."""
        max_turns = 3
        manager = AgentMemoryManager(default_max_turns=max_turns)
        session_id = "eviction_session"

        for i in range(5):
            manager.add_turn(
                session_id=session_id,
                agent_name="Conversational Agent",
                user_query=f"Query {i}",
                agent_response=f"Response {i}"
            )

        agent_mem = manager.get_agent_memory(session_id, "Conversational Agent")
        self.assertEqual(len(agent_mem.turns), max_turns)
        self.assertEqual(agent_mem.turns[0].user_query, "Query 2")
        self.assertEqual(agent_mem.turns[-1].user_query, "Query 4")

    def test_working_memory_scratchpad(self):
        """Verify structured working memory can be recorded, summarized, and updated."""
        session_id = "scratchpad_session"
        self.manager.update_working_memory(
            session_id=session_id,
            agent_name="Stock Analyzer Agent",
            key="latest_indicators",
            value={"symbol": "NVDA", "rsi": 74.2, "support": 118.5}
        )

        summary = self.manager.get_working_memory_summary(session_id, "Stock Analyzer Agent")
        self.assertIn("NVDA", summary)
        self.assertIn("74.2", summary)

        # Router summary should reflect active ticker
        router_info = self.manager.get_session_summary_for_router(session_id)
        self.assertEqual(router_info.get("active_ticker"), "NVDA")

    def test_cross_agent_referral(self):
        """Verify that an agent can look up facts recorded by another agent in the same session."""
        session_id = "referral_session"
        self.manager.add_turn(
            session_id=session_id,
            agent_name="Stock Analyzer Agent",
            user_query="NVDA Technicals",
            agent_response="NVDA RSI is 74.2 and primary resistance is at $125.",
            metadata={"ticker": "NVDA"}
        )
        self.manager.update_working_memory(
            session_id=session_id,
            agent_name="Stock Analyzer Agent",
            key="resistance_levels",
            value={"R1": 125.0, "R2": 130.0}
        )

        # Refer memory from another agent
        data = self.manager.refer_agent_memory(
            session_id=session_id,
            target_agent="Stock Analyzer Agent"
        )
        self.assertTrue(data["found"])
        self.assertEqual(data["active_ticker"], "NVDA")
        self.assertIn("resistance_levels", data["working_memory"])

    def test_refer_agent_memory_tool(self):
        """Verify LangChain refer_agent_memory tool operates with ContextVar session binding."""
        session_id = "tool_test_session"
        agent_memory_manager.add_turn(
            session_id=session_id,
            agent_name="Stock Analyzer Agent",
            user_query="Check AAPL",
            agent_response="AAPL is currently testing $220 support.",
            metadata={"ticker": "AAPL"}
        )

        # Bind session to current context
        set_current_session_id(session_id)
        tool_result = refer_agent_memory.invoke({"agent_name": "Stock Analyzer Agent"})
        self.assertIn("Stock Analyzer Agent", tool_result)
        self.assertIn("AAPL is currently testing $220 support", tool_result)

    def test_session_reset(self):
        """Verify that resetting a session completely wipes all agent memory partitions."""
        session_id = "reset_session"
        self.manager.add_turn(
            session_id=session_id,
            agent_name="Finance Q&A Agent",
            user_query="Explain capital gains tax",
            agent_response="Capital gains tax applies on asset profits."
        )

        self.assertTrue(self.manager.reset_session(session_id))
        mem = self.manager.get_agent_memory(session_id, "Finance Q&A Agent")
        self.assertEqual(len(mem.turns), 0)


if __name__ == "__main__":
    unittest.main()
