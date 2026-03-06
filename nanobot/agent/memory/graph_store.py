"""Enhanced memory store with graph and timeline support.

This module provides GraphMemoryStore which uses:
- MemoryGraph (entities + relationships + facts)
- Timeline (chronological events)
- EntityExtractor (LLM-based extraction)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.memory.extractor import EntityExtractor, _EXTRACT_ENTITIES_TOOL
from nanobot.agent.memory.graph import (
    MemoryEntity,
    MemoryFact,
    MemoryGraph,
    MemoryRelationship,
)
from nanobot.agent.memory.timeline import Timeline, TimelineEntry

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.store import Session


class GraphMemoryStore:
    """Enhanced memory store using graph-based memory.

    Stores entities, relationships, facts, and timeline events.
    No longer uses the legacy MEMORY.md + HISTORY.md approach.
    """

    def __init__(self, workspace: Path, *, enable_graph: bool = True):
        self.workspace = workspace
        self.enable_graph = enable_graph

        self._graph: MemoryGraph | None = None
        self._timeline: Timeline | None = None
        self._extractor: EntityExtractor | None = None

        if enable_graph:
            self._graph = MemoryGraph(workspace)
            self._timeline = Timeline(workspace)
            self._extractor = EntityExtractor(workspace)
            self._graph.load()
            self._timeline.load()

    @property
    def graph(self) -> MemoryGraph | None:
        return self._graph

    @property
    def timeline(self) -> Timeline | None:
        return self._timeline

    def read_long_term(self) -> str:
        return ""

    def write_long_term(self, content: str) -> None:
        pass

    def append_history(self, entry: str) -> None:
        pass

    def get_memory_context(self) -> str:
        """Get formatted memory context including graph and timeline."""
        if not self._graph and not self._timeline:
            return ""

        parts = []

        if self._graph and self._graph.entities:
            graph_parts = []

            entities = list(self._graph.entities.values())[:10]
            if entities:
                graph_parts.append("### Related Entities")
                for entity in entities:
                    graph_parts.append(f"- `{entity.id}`: {entity.description}")

            if self._timeline:
                recent = self._timeline.get_recent(5)
                if recent:
                    graph_parts.append("")
                    graph_parts.append("### Recent Timeline")
                    for entry in recent:
                        graph_parts.append(f"[{entry.date_str} {entry.time_str}] {entry.title}")

            if graph_parts:
                parts.append("## Memory Graph\n\n" + "\n".join(graph_parts))

        return "\n\n---\n\n".join(parts) if parts else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 10,
    ) -> bool:
        """Consolidate old messages into memory graph.

        Extracts entities, relationships, facts, and timeline entries
        from the conversation using LLM.
        """
        if not self.enable_graph or not self._graph or not self._timeline or not self._extractor:
            return True

        try:
            if archive_all:
                old_messages = session.messages
            else:
                keep_count = memory_window // 2
                old_messages = (
                    session.messages[session.last_consolidated : -keep_count]
                    if keep_count > 0
                    else []
                )

            if old_messages:
                await self._extract_and_update_graph(old_messages, provider, model)
        except Exception as e:
            logger.warning("Failed to update memory graph: {}", e)
            return False

        return True

    async def _extract_and_update_graph(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> None:
        """Extract entities from messages and update graph."""
        if not self._extractor or not self._graph or not self._timeline:
            return

        result = await self._extractor.extract_from_messages(messages, provider, model)

        if not result:
            return

        for entity_data in result.get("entities", []):
            entity = MemoryEntity(
                id=entity_data.get("id", ""),
                type=entity_data.get("type", "concept"),
                name=entity_data.get("name", entity_data.get("id", "")),
                description=entity_data.get("description", ""),
                metadata=entity_data.get("metadata", {}),
            )
            if entity.id:
                self._graph.add_entity(entity)

        for rel_data in result.get("relationships", []):
            rel = MemoryRelationship(
                source=rel_data.get("source", ""),
                type=rel_data.get("type", "RELATED_TO"),
                target=rel_data.get("target", ""),
                description=rel_data.get("description"),
            )
            if rel.source and rel.target:
                self._graph.add_relationship(rel)

        for fact_data in result.get("facts", []):
            fact = MemoryFact(
                statement=fact_data.get("statement", ""),
                confidence=fact_data.get("confidence", 0.9),
                source=fact_data.get("source"),
            )
            if fact.statement:
                self._graph.add_fact(fact)

        timeline_data = result.get("timeline_entry", {})
        if timeline_data:
            self._timeline.add_entry(
                title=timeline_data.get("summary", "Conversation")[:100],
                entry_type=timeline_data.get("type", "discussion"),
                summary=timeline_data.get("summary"),
                tags=timeline_data.get("tags", []),
            )

        self._graph.save()
        self._timeline.save()

        logger.info(
            "Memory graph updated: {} entities, {} relationships, {} facts",
            len(self._graph.entities),
            len(self._graph.relationships),
            len(self._graph.facts),
        )

    def save(self) -> None:
        """Save graph and timeline to disk."""
        if self._graph:
            self._graph.save()
        if self._timeline:
            self._timeline.save()

    def find_similar_entities(self, threshold: float = 0.8) -> list[tuple]:
        """Find similar entities that might be duplicates."""
        if not self._graph:
            return []
        return self._graph.find_similar_entities(threshold)

    def merge_entities(self, from_id: str, to_id: str) -> None:
        """Merge one entity into another."""
        if not self._graph:
            return
        self._graph.merge_entity(from_id, to_id)
        self._graph.save()

    def detect_conflicts(self) -> dict[str, list[dict]]:
        """Detect potential conflicts in the graph."""
        if not self._graph:
            return {}
        return self._graph.detect_conflicts()

    def cleanup_graph(self) -> dict[str, int]:
        """Clean up the graph by removing orphaned relationships and duplicates."""
        if not self._graph:
            return {}
        result = self._graph.cleanup()
        self._graph.save()
        return result
