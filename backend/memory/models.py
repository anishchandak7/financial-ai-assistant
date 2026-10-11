"""Data models for per-agent session memory.

Defines schemas for:
- MemoryTurn: A single interaction turn between a user and a specialist agent.
- AgentMemoryState: Scoped memory partition for an individual agent (dialogue turns + working memory scratchpad).
- SessionMemoryState: The complete session-level state holding multiple isolated agent memory partitions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MemoryTurn(BaseModel):
    """Represents a single query-response turn for a specific agent."""
    user_query: str = Field(..., description="User prompt or sub-agent input query")
    agent_response: str = Field(..., description="Assistant response text")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when the turn occurred"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific metadata (e.g. ticker, citations, indicators computed)"
    )


class AgentMemoryState(BaseModel):
    """Isolated memory partition for a specific agent in a session."""
    agent_name: str = Field(..., description="Canonical name of the specialist agent")
    turns: List[MemoryTurn] = Field(
        default_factory=list,
        description="Chronological interaction history for this agent"
    )
    working_memory: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured domain scratchpad (e.g. calculated indicators, cited docs, topics)"
    )


class SessionMemoryState(BaseModel):
    """Session container aggregating isolated agent partitions and top-level session metadata."""
    session_id: str = Field(..., description="Unique conversation session identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation timestamp"
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last interaction timestamp"
    )
    agents: Dict[str, AgentMemoryState] = Field(
        default_factory=dict,
        description="Per-agent isolated memory partitions keyed by agent name"
    )
    active_ticker: Optional[str] = Field(
        None,
        description="Most recently analyzed or discussed stock ticker in this session"
    )
    last_active_agent: Optional[str] = Field(
        None,
        description="Name of the most recently executed specialist agent in this session"
    )
