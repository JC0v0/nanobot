"""Memory retriever for retrieving relevant memory entries.

Provides abstract base class and implementations:
- MemoryRetriever: Abstract base interface
- RuleBasedRetriever: Rule-based retrieval (keyword matching, time-based)
- NoopRetriever: Empty implementation for disabling retrieval
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory.graph import MemoryEntity, MemoryGraph
from nanobot.agent.memory.timeline import Timeline, TimelineEntry


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    type: str  # entity|relationship|fact|timeline
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MemoryRetriever(ABC):
    """Abstract base class for memory retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memory entries for the given query.

        Args:
            query: The query string to search for
            limit: Maximum number of results to return

        Returns:
            List of retrieval results, ordered by relevance
        """
        pass

    @abstractmethod
    def retrieve_entities(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryEntity]:
        """Retrieve relevant entities for the given query.

        Args:
            query: The query string to search for
            limit: Maximum number of results to return

        Returns:
            List of relevant entities
        """
        pass


class NoopRetriever(MemoryRetriever):
    """No-op retriever that returns nothing (for disabling retrieval)."""

    def retrieve(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        return []

    def retrieve_entities(self, query: str, limit: int = 10) -> list[MemoryEntity]:
        return []


class RuleBasedRetriever(MemoryRetriever):
    """Rule-based memory retriever.

    Uses simple rules for retrieval:
    - Keyword matching in entity names and descriptions
    - Time-based ordering (recent first)
    - Type filtering
    - One-hop relationship expansion
    """

    def __init__(self, graph: MemoryGraph | None, timeline: Timeline | None):
        self.graph = graph
        self.timeline = timeline

    def retrieve(self, query: str, limit: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant memory entries using rule-based approach."""
        results: list[RetrievalResult] = []

        # Retrieve entities
        entities = self.retrieve_entities(query, limit=limit // 2)
        for entity in entities:
            results.append(RetrievalResult(
                type="entity",
                content=f"`{entity.id}`: {entity.description}",
                score=1.0,
                metadata={"entity_id": entity.id, "entity_type": entity.type}
            ))

        # Retrieve timeline entries (most recent first)
        if self.timeline:
            recent = self.timeline.get_recent(limit=limit // 2)
            for entry in recent:
                score = 0.8 - (len(results) * 0.05)  # Time-based score decay
                results.append(RetrievalResult(
                    type="timeline",
                    content=f"[{entry.date_str} {entry.time_str}] {entry.title}",
                    score=max(0.1, score),
                    metadata={"timestamp": entry.timestamp.isoformat()}
                ))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def retrieve_entities(self, query: str, limit: int = 10) -> list[MemoryEntity]:
        """Retrieve relevant entities using rule-based approach."""
        if not self.graph:
            return []

        results: list[tuple[MemoryEntity, float]] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for entity in self.graph.entities.values():
            score = self._score_entity(entity, query_lower, query_words)
            if score > 0:
                results.append((entity, score))

        # Add one-hop related entities
        related_ids: set[str] = set()
        for entity, _ in results[:limit]:
            related = self.graph.find_related_entities(entity.id, max_depth=1)
            for rel_entity in related:
                if rel_entity.id not in related_ids and rel_entity.id not in self.graph.entities:
                    related_ids.add(rel_entity.id)
                    results.append((rel_entity, 0.3))  # Lower score for related

        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in results[:limit]]

    def _score_entity(self, entity: MemoryEntity, query_lower: str, query_words: set[str]) -> float:
        """Score an entity against the query."""
        score = 0.0

        # Exact match in ID
        if entity.id.lower() == query_lower:
            score += 2.0
        elif query_lower in entity.id.lower():
            score += 1.0

        # Exact match in name
        if entity.name.lower() == query_lower:
            score += 1.5
        elif query_lower in entity.name.lower():
            score += 0.8

        # Match in description
        if query_lower in entity.description.lower():
            score += 0.5

        # Word matches
        entity_text = f"{entity.id} {entity.name} {entity.description}".lower()
        word_matches = sum(1 for word in query_words if word in entity_text and len(word) > 2)
        score += word_matches * 0.2

        # Metadata matches
        for key, value in entity.metadata.items():
            if isinstance(value, str) and query_lower in value.lower():
                score += 0.3
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and query_lower in v.lower():
                        score += 0.2

        return score


__all__ = [
    "MemoryRetriever",
    "NoopRetriever",
    "RuleBasedRetriever",
    "RetrievalResult",
]
