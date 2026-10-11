"""Thread-safe Agent Memory Manager.

Manages isolated per-agent memory partitions across chat sessions:
- Dialogue history buffering with sliding-window truncation.
- Structured working memory scratchpads for quantitative/RAG facts.
- Cross-agent memory referral mechanisms.
- Router-level context summaries for pronoun and coreference resolution.
"""

import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from memory.base import BaseAgentMemoryStore
from memory.models import SessionMemoryState, AgentMemoryState, MemoryTurn
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)


class AgentMemoryManager(BaseAgentMemoryStore):
    """Thread-safe in-memory session manager for isolated agent memories."""

    def __init__(self, default_max_turns: int = 10):
        self._sessions: Dict[str, SessionMemoryState] = {}
        self._lock = threading.RLock()
        self.default_max_turns = default_max_turns

    def get_or_create_session(self, session_id: str) -> SessionMemoryState:
        with self._lock:
            if session_id not in self._sessions:
                logger.info(f"Initializing new session memory partition for session_id: {session_id}")
                self._sessions[session_id] = SessionMemoryState(session_id=session_id)
            else:
                self._sessions[session_id].last_accessed = datetime.now(timezone.utc)
            return self._sessions[session_id]

    def get_agent_memory(self, session_id: str, agent_name: str) -> AgentMemoryState:
        with self._lock:
            session = self.get_or_create_session(session_id)
            if agent_name not in session.agents:
                session.agents[agent_name] = AgentMemoryState(agent_name=agent_name)
            return session.agents[agent_name]

    def add_turn(
        self,
        session_id: str,
        agent_name: str,
        user_query: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None
    ) -> MemoryTurn:
        with self._lock:
            session = self.get_or_create_session(session_id)
            agent_mem = self.get_agent_memory(session_id, agent_name)

            turn = MemoryTurn(
                user_query=user_query,
                agent_response=agent_response,
                metadata=metadata or {}
            )
            agent_mem.turns.append(turn)

            limit = max_turns or self.default_max_turns
            if len(agent_mem.turns) > limit:
                # Sliding window eviction to prevent prompt bloat
                agent_mem.turns = agent_mem.turns[-limit:]

            # Update session-level active context
            session.last_active_agent = agent_name
            extracted_ticker = (metadata or {}).get("ticker")
            if extracted_ticker:
                session.active_ticker = str(extracted_ticker).upper()

            logger.info(
                f"Recorded turn in {agent_name} memory (Session: {session_id[:8]}..., "
                f"Total turns: {len(agent_mem.turns)}, Ticker: {session.active_ticker})"
            )
            return turn

    def update_working_memory(
        self,
        session_id: str,
        agent_name: str,
        key: str,
        value: Any
    ) -> None:
        with self._lock:
            agent_mem = self.get_agent_memory(session_id, agent_name)
            agent_mem.working_memory[key] = value

            # If an indicator snapshot is stored with a ticker, reflect on session
            if key == "latest_indicators" and isinstance(value, dict) and "symbol" in value:
                self._sessions[session_id].active_ticker = str(value["symbol"]).upper()

            logger.debug(f"Updated working memory for {agent_name} [{key}] (Session: {session_id[:8]}...)")

    def refer_agent_memory(
        self,
        session_id: str,
        target_agent: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Allows an agent to query or refer back to another agent's memory in the same session."""
        with self._lock:
            session = self.get_or_create_session(session_id)
            if target_agent not in session.agents:
                return {
                    "found": False,
                    "message": f"No memory found for {target_agent} in session {session_id}."
                }

            mem = session.agents[target_agent]
            recent_turns = [
                {
                    "query": t.user_query,
                    "response": t.agent_response[:300] + ("..." if len(t.agent_response) > 300 else ""),
                    "metadata": t.metadata
                }
                for t in mem.turns[-3:]
            ]

            return {
                "found": True,
                "agent_name": target_agent,
                "recent_turns": recent_turns,
                "working_memory": mem.working_memory,
                "active_ticker": session.active_ticker
            }

    def get_messages_for_agent(
        self,
        session_id: str,
        agent_name: str,
        max_turns: Optional[int] = None
    ) -> List[BaseMessage]:
        """Formats the agent's memory turns into standard LangChain BaseMessage objects."""
        with self._lock:
            agent_mem = self.get_agent_memory(session_id, agent_name)
            limit = max_turns or self.default_max_turns
            selected_turns = agent_mem.turns[-limit:]

            messages: List[BaseMessage] = []
            for t in selected_turns:
                messages.append(HumanMessage(content=t.user_query))
                messages.append(AIMessage(content=t.agent_response))
            return messages

    def get_working_memory_summary(self, session_id: str, agent_name: str) -> str:
        """Produces a concise text summary of the agent's structured working memory."""
        with self._lock:
            agent_mem = self.get_agent_memory(session_id, agent_name)
            if not agent_mem.working_memory:
                return ""

            summary_lines = [f"### Session Memory Scratchpad for {agent_name}:"]
            for k, v in agent_mem.working_memory.items():
                if isinstance(v, dict):
                    inner = ", ".join(f"{sub_k}: {sub_v}" for sub_k, sub_v in list(v.items())[:6])
                    summary_lines.append(f"- **{k}**: {inner}")
                elif isinstance(v, list):
                    items = ", ".join(str(x) for x in v[:5])
                    summary_lines.append(f"- **{k}**: [{items}]")
                else:
                    summary_lines.append(f"- **{k}**: {v}")
            return "\n".join(summary_lines)

    def get_session_summary_for_router(self, session_id: str) -> Dict[str, Any]:
        """Provides the triage router with recent session context for disambiguation."""
        with self._lock:
            session = self.get_or_create_session(session_id)
            return {
                "active_ticker": session.active_ticker,
                "last_active_agent": session.last_active_agent,
                "total_agents_engaged": len(session.agents),
            }

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Reset and cleared session memory for session_id: {session_id}")
                return True
            return False


# Global singleton instance
agent_memory_manager = AgentMemoryManager()
