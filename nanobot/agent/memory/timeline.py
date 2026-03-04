"""Timeline manager for event history.

Stores chronological events in TIMELINE.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

TIMELINE_TYPES = {"discussion", "code_change", "decision", "learning", "task"}


@dataclass
class TimelineEntry:
    """A single entry in the timeline."""
    timestamp: datetime
    title: str
    type: str = "discussion"  # discussion|code_change|decision|learning|task
    summary: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def date_str(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M")

    def to_markdown(self) -> str:
        """Convert entry to Markdown."""
        lines = [f"### {self.time_str} - {self.title}"]
        if self.type != "discussion":
            lines = [f"### {self.time_str} - [{self.type.replace('_', ' ').title()}] {self.title}"]

        if self.summary:
            lines.append(f"- Summary: {self.summary}")

        for key, value in sorted(self.details.items()):
            if isinstance(value, list):
                lines.append(f"- {key.replace('_', ' ').title()}: {', '.join(value)}")
            elif value:
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)


class Timeline:
    """Timeline manager for chronological events."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.timeline_file = self.memory_dir / "TIMELINE.md"

        self.entries: list[TimelineEntry] = []

    def load(self) -> None:
        """Load from TIMELINE.md."""
        if not self.timeline_file.exists():
            logger.debug("Timeline file not found, starting empty")
            return

        try:
            content = self.timeline_file.read_text(encoding="utf-8")
            self._parse_markdown(content)
            logger.debug("Loaded timeline: {} entries", len(self.entries))
        except Exception as e:
            logger.warning("Failed to parse timeline, starting empty: {}", e)
            self.entries = []

    def _parse_markdown(self, content: str) -> None:
        """Parse Markdown content into timeline entries."""
        lines = content.split("\n")
        current_date: str | None = None
        current_entry: TimelineEntry | None = None

        for line in lines:
            line = line.rstrip()

            # Date section
            if line.startswith("## "):
                current_date = line[3:].strip()
                current_entry = None

            # Entry header
            elif line.startswith("### "):
                if current_entry:
                    self.entries.append(current_entry)

                header = line[4:].strip()

                # Parse "[Type] Title" format
                entry_type = "discussion"
                title = header

                type_match = re.match(r"\[([^\]]+)\]\s*(.*)", header)
                if type_match:
                    type_str, title = type_match.groups()
                    entry_type = type_str.lower().replace(" ", "_")

                # Parse time from start
                time_match = re.match(r"(\d{2}:\d{2})\s*-\s*(.*)", title)
                if time_match:
                    time_str, title = time_match.groups()
                else:
                    time_str = "00:00"

                if current_date:
                    try:
                        ts = datetime.strptime(f"{current_date} {time_str}", "%Y-%m-%d %H:%M")
                        current_entry = TimelineEntry(
                            timestamp=ts,
                            title=title,
                            type=entry_type
                        )
                    except ValueError:
                        current_entry = None
                else:
                    current_entry = None

            # Entry details (list items)
            elif line.strip().startswith("- ") and current_entry:
                detail_line = line.strip()[2:]
                if ": " in detail_line:
                    key, value = detail_line.split(": ", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()
                    if key == "summary":
                        current_entry.summary = value
                    else:
                        if ", " in value:
                            current_entry.details[key] = [v.strip() for v in value.split(", ")]
                        else:
                            current_entry.details[key] = value
                else:
                    # Free-form line, append to summary
                    if current_entry.summary:
                        current_entry.summary += " " + detail_line.strip()
                    else:
                        current_entry.summary = detail_line.strip()

        if current_entry:
            self.entries.append(current_entry)

        # Sort by timestamp
        self.entries.sort(key=lambda e: e.timestamp)

    def save(self) -> None:
        """Save to TIMELINE.md."""
        content = self._to_markdown()
        self.timeline_file.write_text(content, encoding="utf-8")
        logger.debug("Saved timeline: {} entries", len(self.entries))

    def _to_markdown(self) -> str:
        """Convert timeline to Markdown."""
        lines = ["# Timeline", ""]

        if not self.entries:
            return "\n".join(lines) + "\n"

        # Group entries by date
        entries_by_date: dict[str, list[TimelineEntry]] = {}
        for entry in self.entries:
            entries_by_date.setdefault(entry.date_str, []).append(entry)

        # Write each date section (newest first)
        for date_str in sorted(entries_by_date.keys(), reverse=True):
            lines.append(f"## {date_str}")
            for entry in sorted(entries_by_date[date_str], key=lambda e: e.timestamp, reverse=True):
                lines.append("")
                lines.append(entry.to_markdown())
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def add_entry(
        self,
        title: str,
        entry_type: str = "discussion",
        summary: str | None = None,
        timestamp: datetime | None = None,
        **details: Any
    ) -> TimelineEntry:
        """Add a new entry to the timeline."""
        entry = TimelineEntry(
            timestamp=timestamp or datetime.now(),
            title=title,
            type=entry_type if entry_type in TIMELINE_TYPES else "discussion",
            summary=summary,
            details=details
        )
        self.entries.append(entry)
        # Keep sorted
        self.entries.sort(key=lambda e: e.timestamp)
        return entry

    def get_recent(self, limit: int = 10) -> list[TimelineEntry]:
        """Get the most recent N entries."""
        return list(reversed(self.entries[-limit:]))

    def get_by_date(self, date: datetime | str) -> list[TimelineEntry]:
        """Get entries for a specific date."""
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")
        return [e for e in self.entries if e.date_str == date_str]

    def get_by_type(self, entry_type: str) -> list[TimelineEntry]:
        """Get entries of a specific type."""
        return [e for e in self.entries if e.type == entry_type]
