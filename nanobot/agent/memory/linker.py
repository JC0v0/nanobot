"""Context linker for building graph context.

Extracts relevant information from the graph and timeline
and builds formatted context strings for LLM consumption.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory.graph import MemoryGraph
from nanobot.agent.memory.retriever import (
    MemoryRetriever,
    RetrievalResult,
    RuleBasedRetriever,
)
from nanobot.agent.memory.timeline import Timeline


class ContextLinker:
    """Links current query to memory graph and builds context."""

    def __init__(
        self,
        graph: MemoryGraph | None,
        timeline: Timeline | None,
    ):
        self.graph = graph
        self.timeline = timeline
        self.retriever: MemoryRetriever = RuleBasedRetriever(graph, timeline)

    def set_retriever(self, retriever: MemoryRetriever) -> None:
        """Set a custom retriever."""
        self.retriever = retriever

    def extract_query_entities(self, query: str) -> list[str]:
        """Extract potential entity keywords from query.

        Uses simple heuristics to find potential entity references.
        """
        keywords: set[str] = set()

        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.update(quoted)
        quoted = re.findall(r"'([^']+)'", query)
        keywords.update(quoted)

        # Extract backtick terms
        backtick = re.findall(r"`([^`]+)`", query)
        keywords.update(backtick)

        # Extract file-like paths
        files = re.findall(r'[\w/-]+\.(?:py|md|txt|json|yaml|yml)', query)
        keywords.update(files)

        # Extract capitalized terms (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z0-9_-]+\b', query)
        keywords.update(capitalized)

        # If no specific terms found, use the whole query
        if not keywords:
            keywords.add(query)

        return list(keywords)

    def build_graph_context(
        self,
        query: str,
        entity_limit: int = 10,
    ) -> str:
        """Build graph context section for the query.

        Returns a Markdown-formatted string with relevant entities
        and relationships.
        """
        if not self.graph:
            return ""

        parts: list[str] = []

        # Get relevant entities
        entities = self.retriever.retrieve_entities(query, limit=entity_limit)

        if entities:
            parts.append("### Related Entities")
            for entity in entities:
                parts.append(f"- `{entity.id}`: {entity.description}")

        # Get recent timeline entries
        if self.timeline:
            recent = self.timeline.get_recent(5)
            if recent:
                if parts:
                    parts.append("")
                parts.append("### Recent Timeline")
                for entry in recent:
                    parts.append(f"[{entry.date_str} {entry.time_str}] {entry.title}")

        return "\n".join(parts) if parts else ""

    def build_timeline_context(
        self,
        limit: int = 5,
    ) -> str:
        """Build timeline context section.

        Returns a Markdown-formatted string with recent timeline entries.
        """
        if not self.timeline:
            return ""

        recent = self.timeline.get_recent(limit)
        if not recent:
            return ""

        parts = ["### Recent Timeline"]
        for entry in recent:
            parts.append(f"[{entry.date_str} {entry.time_str}] {entry.title}")

        return "\n".join(parts)

    def rank_results(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Rank and filter retrieval results.

        Orders by score and applies diversity filtering.
        """
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply diversity: don't show too many of the same type
        type_counts: dict[str, int] = {}
        max_per_type = 5
        filtered: list[RetrievalResult] = []

        for result in results:
            count = type_counts.get(result.type, 0)
            if count < max_per_type:
                filtered.append(result)
                type_counts[result.type] = count + 1

        return filtered

    def build_full_context(
        self,
        query: str,
        include_entities: bool = True,
        include_timeline: bool = True,
        entity_limit: int = 10,
        timeline_limit: int = 5,
    ) -> str:
        """Build full graph context section.

        Returns a complete Markdown section with both entities
        and timeline (if enabled).
        """
        if not self.graph and not self.timeline:
            return ""

        parts: list[str] = []

        if include_entities and self.graph:
            entity_ctx = self.build_graph_context(query, entity_limit=entity_limit)
            if entity_ctx:
                parts.append(entity_ctx)

        if include_timeline and self.timeline and not include_entities:
            timeline_ctx = self.build_timeline_context(limit=timeline_limit)
            if timeline_ctx:
                parts.append(timeline_ctx)

        if parts:
            return "## Memory Graph\n\n" + "\n".join(parts)
        return ""


__all__ = ["ContextLinker"]
