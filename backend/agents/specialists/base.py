from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSpecialistAgent(ABC):
    """Abstract base class for all domain specialist agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name of the specialist agent."""
        pass

    @abstractmethod
    def run(self, input_query: str, **kwargs) -> Dict[str, Any]:
        """Executes the agent logic on the user input query."""
        pass
