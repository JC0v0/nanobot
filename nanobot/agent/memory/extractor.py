"""Entity extraction for memory graph.

Extracts entities, relationships, facts, and timeline entries
from conversation messages using LLM tool calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


# 实体提取工具定义
_EXTRACT_ENTITIES_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "extract_memory_entities",
            "description": "Extract entities, relationships, facts, and timeline entry from conversation for memory graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique identifier for the entity (lowercase, no spaces, use underscores)"},
                                "type": {"type": "string", "description": "Entity type: project|file|feature|concept|person|tool"},
                                "name": {"type": "string", "description": "Human-readable name"},
                                "description": {"type": "string", "description": "1-2 sentence description"},
                                "metadata": {"type": "object", "description": "Additional metadata (tags, path, location, part_of, etc.)"},
                            },
                            "required": ["id", "type", "name", "description"],
                        },
                        "description": "List of entities extracted from the conversation.",
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "description": "Source entity ID (must match an entity id)"},
                                "type": {"type": "string", "description": "Relationship type: HAS_FILE|USES|IMPLEMENTS|BLOCKED_BY|RELATED_TO|PART_OF|IMPLEMENTED_IN|HAS_FEATURE"},
                                "target": {"type": "string", "description": "Target entity ID (must match an entity id)"},
                                "description": {"type": "string", "description": "Optional description of the relationship"},
                            },
                            "required": ["source", "type", "target"],
                        },
                        "description": "List of relationships between entities.",
                    },
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string", "description": "The confirmed fact statement"},
                                "confidence": {"type": "number", "description": "Confidence score 0-1 (0.9 for user statements, 0.7 for inferences)"},
                                "source": {"type": "string", "description": "Source of the fact (e.g., 'user said', 'inferred from code', 'conversation')"},
                            },
                            "required": ["statement"],
                        },
                        "description": "List of confirmed facts from the conversation.",
                    },
                    "timeline_entry": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "2-5 sentence summary of what happened in this conversation"},
                            "type": {"type": "string", "description": "Entry type: discussion|code_change|decision|learning|task"},
                            "tags": {"type": "array", "items": {"type": "string"}, "description": "Relevant tags (keywords, entity ids, topics)"},
                        },
                        "description": "Timeline entry for this conversation.",
                    },
                },
                "required": ["entities", "relationships", "facts", "timeline_entry"],
            },
        },
    }
]


class EntityExtractor:
    """Extract entities and relationships from conversation messages."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    @staticmethod
    def format_messages_for_extraction(messages: list[dict]) -> str:
        """Format messages for entity extraction."""
        lines = []
        for m in messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            timestamp = m.get("timestamp", "?")[:16]
            role = m["role"].upper()
            content = m.get("content", "")
            lines.append(f"[{timestamp}] {role}{tools}: {content}")
        return "\n".join(lines)

    async def extract_from_messages(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> dict | None:
        """Extract entities, relationships, facts, and timeline entry from messages.

        Returns a dict with:
            - entities: list of entity dicts
            - relationships: list of relationship dicts
            - facts: list of fact dicts
            - timeline_entry: timeline entry dict
        """
        if not messages:
            return None

        formatted = self.format_messages_for_extraction(messages)
        if not formatted.strip():
            return None

        prompt = f"""Analyze this conversation and extract entities, relationships, facts, and a timeline entry.

Call the extract_memory_entities tool with your analysis.

## Conversation
{formatted}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory extraction assistant. Analyze the conversation and call extract_memory_entities."},
                    {"role": "user", "content": prompt},
                ],
                tools=_EXTRACT_ENTITIES_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.debug("Entity extraction: LLM did not call tool")
                return None

            args = response.tool_calls[0].arguments
            parsed = self.parse_extraction_result(args)
            return parsed

        except Exception:
            logger.exception("Entity extraction failed")
            return None

    @staticmethod
    def parse_extraction_result(args: dict | str) -> dict | None:
        """Parse extraction result from LLM tool call."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                logger.warning("Failed to parse extraction result JSON")
                return None

        if not isinstance(args, dict):
            logger.warning("Extraction result is not a dict: {}", type(args).__name__)
            return None

        return {
            "entities": args.get("entities", []),
            "relationships": args.get("relationships", []),
            "facts": args.get("facts", []),
            "timeline_entry": args.get("timeline_entry", {}),
        }


__all__ = ["EntityExtractor", "_EXTRACT_ENTITIES_TOOL"]
