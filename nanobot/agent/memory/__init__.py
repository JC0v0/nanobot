"""Memory system module for nanobot.

This module provides memory management for the agent, including:
- LegacyMemoryStore: Original two-layer memory (MEMORY.md + HISTORY.md)
- MemoryGraph: Structured entity-relationship graph
- Timeline: Chronological event timeline
- GraphMemoryStore: Enhanced memory store combining all of the above
- EntityExtractor: LLM-based entity/relationship extraction

For backward compatibility, the default export is still LegacyMemoryStore
under the name 'MemoryStore'.
"""

from nanobot.agent.memory.base import MemoryStore
from nanobot.agent.memory.extractor import EntityExtractor, _EXTRACT_ENTITIES_TOOL
from nanobot.agent.memory.graph import (
    MemoryEntity,
    MemoryFact,
    MemoryGraph,
    MemoryRelationship,
)
from nanobot.agent.memory.graph_store import GraphMemoryStore
from nanobot.agent.memory.store import LegacyMemoryStore, _SAVE_MEMORY_TOOL
from nanobot.agent.memory.timeline import Timeline, TimelineEntry

# Backward compatibility: export LegacyMemoryStore as MemoryStore
# This ensures existing code continues to work without modification
MemoryStore = LegacyMemoryStore

__all__ = [
    # Base interface
    "MemoryStore",
    # Legacy implementation (backward compatible default)
    "LegacyMemoryStore",
    # Enhanced memory store with graph
    "GraphMemoryStore",
    # Entity extractor
    "EntityExtractor",
    # Graph components
    "MemoryGraph",
    "MemoryEntity",
    "MemoryRelationship",
    "MemoryFact",
    # Timeline components
    "Timeline",
    "TimelineEntry",
    # Internal
    "_SAVE_MEMORY_TOOL",
    "_EXTRACT_ENTITIES_TOOL",
]
