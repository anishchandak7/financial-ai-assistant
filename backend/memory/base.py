"""Abstract base classes for agent memory storage."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from memory.models import SessionMemoryState, AgentMemoryState, MemoryTurn


class BaseAgentMemoryStore(ABC):
    """Abstract interface for session-scoped per-agent memory storage."""

    @abstractmethod
    def get_or_create_session(self, session_id: str) -> SessionMemoryState:
        """Retrieves an existing session or initializes a new one."""
        pass

    @abstractmethod
    def get_agent_memory(self, session_id: str, agent_name: str) -> AgentMemoryState:
        """Retrieves the isolated memory partition for a specific agent."""
        pass

    @abstractmethod
    def add_turn(
        self,
        session_id: str,
        agent_name: str,
        user_query: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_turns: int = 10
    ) -> MemoryTurn:
        """Appends a new turn to the agent's memory with sliding window eviction."""
        pass

    @abstractmethod
    def update_working_memory(
        self,
        session_id: str,
        agent_name: str,
        key: str,
        value: Any
    ) -> None:
        """Sets a domain fact or metric into the agent's working scratchpad."""
        pass

    @abstractmethod
    def refer_agent_memory(
        self,
        session_id: str,
        target_agent: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Allows one agent to inspect another agent's memory within the same session."""
        pass

    @abstractmethod
    def reset_session(self, session_id: str) -> bool:
        """Clears all agent memories for the given session."""
        pass
