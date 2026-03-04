"""Memory graph implementation for structured memory storage.

Stores entities, relationships, and confirmed facts in MEMORY_GRAPH.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

ENTITY_TYPES = {"project", "file", "feature", "concept", "person", "tool"}
RELATIONSHIP_TYPES = {
    "HAS_FILE", "USES", "IMPLEMENTS", "BLOCKED_BY", "RELATED_TO",
    "PART_OF", "IMPLEMENTED_IN", "HAS_FEATURE", "DISCUSSED"
}


@dataclass
class MemoryEntity:
    """An entity in the memory graph."""
    id: str
    type: str  # project|file|feature|concept|person|tool
    name: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self, indent: int = 0) -> str:
        """Convert entity to Markdown list item."""
        prefix = "  " * indent
        lines = [f"{prefix}- `{self.id}`: {self.description}"]
        for key, value in self.metadata.items():
            if isinstance(value, list):
                lines.append(f"{prefix}  - {key}: {', '.join(value)}")
            else:
                lines.append(f"{prefix}  - {key}: {value}")
        return "\n".join(lines)


@dataclass
class MemoryRelationship:
    """A relationship between two entities."""
    source: str
    type: str  # HAS_FILE|USES|IMPLEMENTS|...
    target: str
    description: str | None = None

    def to_markdown(self) -> str:
        """Convert relationship to Markdown line."""
        if self.description:
            return f"- {self.source} {self.type} {self.target} — {self.description}"
        return f"- {self.source} {self.type} {self.target}"


@dataclass
class MemoryFact:
    """A confirmed fact."""
    statement: str
    confidence: float = 0.9
    source: str | None = None

    def to_markdown(self) -> str:
        """Convert fact to Markdown list item."""
        if self.source:
            return f"- {self.statement} (source: {self.source}, confidence: {self.confidence})"
        return f"- {self.statement}"


class MemoryGraph:
    """Memory graph for storing entities, relationships, and facts."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.graph_file = self.memory_dir / "MEMORY_GRAPH.md"

        self.entities: dict[str, MemoryEntity] = {}
        self.relationships: list[MemoryRelationship] = []
        self.facts: list[MemoryFact] = []
        self.version: str = "1.0"
        self.last_updated: datetime = datetime.now()

    def load(self) -> None:
        """Load from MEMORY_GRAPH.md."""
        if not self.graph_file.exists():
            logger.debug("Memory graph file not found, starting empty")
            return

        try:
            content = self.graph_file.read_text(encoding="utf-8")
            self._parse_markdown(content)
            logger.debug("Loaded memory graph: {} entities, {} relationships, {} facts",
                        len(self.entities), len(self.relationships), len(self.facts))
        except Exception as e:
            logger.warning("Failed to parse memory graph, starting empty: {}", e)
            self.entities = {}
            self.relationships = []
            self.facts = []

    def _parse_markdown(self, content: str) -> None:
        """Parse Markdown content into graph structure."""
        lines = content.split("\n")
        section = None
        entity_type = None
        current_entity: MemoryEntity | None = None
        in_frontmatter = False
        seen_content = False

        for i, line in enumerate(lines):
            line = line.rstrip()

            # Frontmatter handling (only at beginning of file)
            if line == "---" and not seen_content:
                in_frontmatter = not in_frontmatter
                continue

            # If we see a non-empty, non-frontmatter line, we've passed the header
            if line.strip() and not in_frontmatter:
                seen_content = True

            if in_frontmatter:
                if line.startswith("version:"):
                    self.version = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("last_updated:"):
                    ts = line.split(":", 1)[1].strip().strip('"')
                    try:
                        self.last_updated = datetime.fromisoformat(ts)
                    except ValueError:
                        pass
                continue

            # Section headers
            if line.startswith("## "):
                section = line[3:].strip()
                entity_type = None
                if current_entity:
                    self.entities[current_entity.id] = current_entity
                    current_entity = None
            elif line.startswith("### "):
                entity_type = line[4:].strip().lower().rstrip("s")

            # Entity
            elif line.strip().startswith("- `") and section == "Entities":
                if current_entity:
                    self.entities[current_entity.id] = current_entity
                match = re.match(r"\s*-\s*`([^`]+)`:\s*(.*)", line)
                if match:
                    eid, desc = match.groups()
                    current_entity = MemoryEntity(
                        id=eid,
                        type=entity_type or "concept",
                        name=eid,
                        description=desc
                    )

            # Entity metadata
            elif line.strip().startswith("- ") and current_entity and ":" in line:
                match = re.match(r"\s*-\s*([^:]+):\s*(.*)", line.strip())
                if match:
                    key, value = match.groups()
                    if ", " in value:
                        current_entity.metadata[key] = [v.strip() for v in value.split(", ")]
                    else:
                        current_entity.metadata[key] = value

            # Relationship
            elif line.strip().startswith("- ") and section == "Relationships":
                parts = line.strip()[2:].split(" — ", 1)
                main = parts[0]
                desc = parts[1] if len(parts) > 1 else None
                # Parse "source TYPE target"
                words = main.split()
                if len(words) >= 3:
                    rel_type = None
                    for t in RELATIONSHIP_TYPES:
                        if t in main:
                            rel_type = t
                            break
                    if rel_type:
                        source = main[:main.index(rel_type)].strip()
                        target = main[main.index(rel_type) + len(rel_type):].strip()
                        self.relationships.append(MemoryRelationship(
                            source=source,
                            type=rel_type,
                            target=target,
                            description=desc
                        ))

            # Fact
            elif line.strip().startswith("- ") and section == "Confirmed Facts":
                stmt = line.strip()[2:]
                source = None
                confidence = 0.9
                if "(source:" in stmt:
                    stmt, meta_part = stmt.split("(source:", 1)
                    stmt = stmt.strip()
                    if ", confidence:" in meta_part:
                        source_part, conf_part = meta_part.split(", confidence:", 1)
                        source = source_part.strip().rstrip(")")
                        try:
                            confidence = float(conf_part.strip().rstrip(")"))
                        except ValueError:
                            pass
                    else:
                        source = meta_part.strip().rstrip(")")
                self.facts.append(MemoryFact(
                    statement=stmt,
                    confidence=confidence,
                    source=source
                ))

        if current_entity:
            self.entities[current_entity.id] = current_entity

    def save(self) -> None:
        """Save to MEMORY_GRAPH.md."""
        self.last_updated = datetime.now()
        content = self._to_markdown()
        self.graph_file.write_text(content, encoding="utf-8")
        logger.debug("Saved memory graph: {} entities, {} relationships, {} facts",
                    len(self.entities), len(self.relationships), len(self.facts))

    def _to_markdown(self) -> str:
        """Convert graph to Markdown."""
        lines = [
            "---",
            f'version: "{self.version}"',
            f'last_updated: "{self.last_updated.isoformat()}"',
            "---",
            "",
            "# Memory Graph",
            "",
        ]

        # Entities grouped by type
        if self.entities:
            lines.append("## Entities")
            # Group entities by type
            entities_by_type: dict[str, list[MemoryEntity]] = {}
            for entity in self.entities.values():
                entities_by_type.setdefault(entity.type, []).append(entity)
            # Write each type section
            type_names = {
                "project": "Projects",
                "file": "Files",
                "feature": "Features",
                "concept": "Concepts",
                "person": "People",
                "tool": "Tools",
            }
            for entity_type, entities in sorted(entities_by_type.items()):
                section_name = type_names.get(entity_type, entity_type.title() + "s")
                lines.append(f"### {section_name}")
                for entity in sorted(entities, key=lambda e: e.id):
                    lines.append(entity.to_markdown())
                lines.append("")

        # Relationships
        if self.relationships:
            lines.append("---")
            lines.append("")
            lines.append("## Relationships")
            for rel in self.relationships:
                lines.append(rel.to_markdown())
            lines.append("")

        # Facts
        if self.facts:
            lines.append("---")
            lines.append("")
            lines.append("## Confirmed Facts")
            for fact in self.facts:
                lines.append(fact.to_markdown())
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def add_entity(self, entity: MemoryEntity) -> None:
        """Add or update an entity."""
        self.entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> MemoryEntity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def add_relationship(self, relationship: MemoryRelationship) -> None:
        """Add a relationship (avoiding duplicates)."""
        # Check for duplicate
        for existing in self.relationships:
            if (existing.source == relationship.source and
                existing.type == relationship.type and
                existing.target == relationship.target):
                return
        self.relationships.append(relationship)

    def add_fact(self, fact: MemoryFact) -> None:
        """Add a fact."""
        self.facts.append(fact)

    def find_related_entities(self, entity_id: str, max_depth: int = 1) -> list[MemoryEntity]:
        """Find entities related to the given entity."""
        result: list[MemoryEntity] = []
        seen: set[str] = {entity_id}

        for _ in range(max_depth):
            new_seen: set[str] = set()
            for rel in self.relationships:
                if rel.source in seen and rel.target not in seen:
                    if entity := self.entities.get(rel.target):
                        result.append(entity)
                        new_seen.add(rel.target)
                if rel.target in seen and rel.source not in seen:
                    if entity := self.entities.get(rel.source):
                        result.append(entity)
                        new_seen.add(rel.source)
            seen.update(new_seen)

        return result

    def find_entities_by_keyword(self, keyword: str) -> list[MemoryEntity]:
        """Find entities by keyword in name or description."""
        keyword_lower = keyword.lower()
        result: list[MemoryEntity] = []
        for entity in self.entities.values():
            if (keyword_lower in entity.id.lower() or
                keyword_lower in entity.name.lower() or
                keyword_lower in entity.description.lower()):
                result.append(entity)
        return result

    def merge_entity(self, from_id: str, to_id: str) -> None:
        """Merge one entity into another, redirecting all relationships."""
        if from_id not in self.entities or to_id not in self.entities:
            return

        # Update relationships
        for rel in self.relationships:
            if rel.source == from_id:
                rel.source = to_id
            if rel.target == from_id:
                rel.target = to_id

        # Remove old entity
        del self.entities[from_id]
