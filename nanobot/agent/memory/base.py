"""Abstract base interface for memory stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.store import Session


class MemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    def __init__(self, workspace: Path) -> None:
        """Initialize the memory store."""
        pass

    @abstractmethod
    def read_long_term(self) -> str:
        """Read long-term memory."""
        pass

    @abstractmethod
    def write_long_term(self, content: str) -> None:
        """Write long-term memory."""
        pass

    @abstractmethod
    def append_history(self, entry: str) -> None:
        """Append to history log."""
        pass

    @abstractmethod
    def get_memory_context(self) -> str:
        """Get formatted memory context for LLM."""
        pass

    @abstractmethod
    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into memory."""
        pass
