"""Enhanced memory store with graph and timeline support.

This module provides GraphMemoryStore which combines:
- LegacyMemoryStore (MEMORY.md + HISTORY.md)
- MemoryGraph (entities + relationships + facts)
- Timeline (chronological events)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.memory.base import MemoryStore
from nanobot.agent.memory.extractor import EntityExtractor, _EXTRACT_ENTITIES_TOOL
from nanobot.agent.memory.graph import (
    MemoryEntity,
    MemoryFact,
    MemoryGraph,
    MemoryRelationship,
)
from nanobot.agent.memory.store import LegacyMemoryStore, _SAVE_MEMORY_TOOL
from nanobot.agent.memory.timeline import Timeline, TimelineEntry

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


class GraphMemoryStore(MemoryStore):
    """Enhanced memory store combining legacy memory, graph, and timeline.

    This class maintains backward compatibility with MemoryStore while
    adding graph-based memory capabilities.
    """

    def __init__(self, workspace: Path, *, enable_graph: bool = True):
        self.workspace = workspace
        self.enable_graph = enable_graph

        # Legacy memory (always present for backward compatibility)
        self._legacy = LegacyMemoryStore(workspace)

        # Graph and timeline (optional)
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
        """Access the memory graph (if enabled)."""
        return self._graph

    @property
    def timeline(self) -> Timeline | None:
        """Access the timeline (if enabled)."""
        return self._timeline

    def read_long_term(self) -> str:
        """Read long-term memory from legacy store."""
        return self._legacy.read_long_term()

    def write_long_term(self, content: str) -> None:
        """Write long-term memory to legacy store."""
        self._legacy.write_long_term(content)

    def append_history(self, entry: str) -> None:
        """Append to history log in legacy store."""
        self._legacy.append_history(entry)

    def get_memory_context(self) -> str:
        """Get formatted memory context including graph and timeline."""
        parts = []

        # Legacy memory first
        legacy_context = self._legacy.get_memory_context()
        if legacy_context:
            parts.append(legacy_context)

        # Add graph context if enabled
        if self.enable_graph and self._graph and self._graph.entities:
            graph_parts = []

            # Add related entities (limit to 10 most recent/relevant)
            entities = list(self._graph.entities.values())[:10]
            if entities:
                graph_parts.append("### Related Entities")
                for entity in entities:
                    graph_parts.append(f"- `{entity.id}`: {entity.description}")

            # Add recent timeline entries
            if self._timeline:
                recent = self._timeline.get_recent(5)
                if recent:
                    graph_parts.append("")
                    graph_parts.append("### Recent Timeline")
                    for entry in recent:
                        graph_parts.append(
                            f"[{entry.date_str} {entry.time_str}] {entry.title}"
                        )

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
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into memory.

        This runs the legacy consolidation first, then optionally extracts
        entities and updates the graph and timeline.
        """
        # First run legacy consolidation
        legacy_success = await self._legacy.consolidate(
            session, provider, model,
            archive_all=archive_all,
            memory_window=memory_window,
        )

        # If graph is disabled, we're done
        if not self.enable_graph or not self._graph or not self._timeline or not self._extractor:
            return legacy_success

        # Extract entities if we have messages to process
        try:
            if archive_all:
                old_messages = session.messages
            else:
                keep_count = memory_window // 2
                old_messages = session.messages[session.last_consolidated : -keep_count] if keep_count > 0 else []

            if old_messages:
                await self._extract_and_update_graph(
                    old_messages, provider, model
                )
        except Exception as e:
            logger.warning("Failed to update memory graph: {}", e)
            # Don't fail the whole consolidation if graph update fails

        return legacy_success

    async def _extract_and_update_graph(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> None:
        """Extract entities from messages and update graph."""
        if not self._extractor or not self._graph or not self._timeline:
            return

        # Use the extractor
        result = await self._extractor.extract_from_messages(
            messages, provider, model
        )

        if not result:
            return

        # Parse and add entities
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

        # Parse and add relationships
        for rel_data in result.get("relationships", []):
            rel = MemoryRelationship(
                source=rel_data.get("source", ""),
                type=rel_data.get("type", "RELATED_TO"),
                target=rel_data.get("target", ""),
                description=rel_data.get("description"),
            )
            if rel.source and rel.target:
                self._graph.add_relationship(rel)

        # Parse and add facts
        for fact_data in result.get("facts", []):
            fact = MemoryFact(
                statement=fact_data.get("statement", ""),
                confidence=fact_data.get("confidence", 0.9),
                source=fact_data.get("source"),
            )
            if fact.statement:
                self._graph.add_fact(fact)

        # Parse and add timeline entry
        timeline_data = result.get("timeline_entry", {})
        if timeline_data:
            self._timeline.add_entry(
                title=timeline_data.get("summary", "Conversation")[:100],
                entry_type=timeline_data.get("type", "discussion"),
                summary=timeline_data.get("summary"),
                tags=timeline_data.get("tags", []),
            )

        # Save changes
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
