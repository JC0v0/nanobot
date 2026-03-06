"""Skill management tool for workspace skill evolution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


class SkillManagerTool(Tool):
    """Safely manage skills in workspace/skills only."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.skills_dir = workspace / "skills"

    @property
    def name(self) -> str:
        return "skill_manager"

    @property
    def description(self) -> str:
        return (
            "Manage workspace skills safely. "
            "Actions: list, read, create, update, deprecate. "
            "Never touches builtin skills."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "create", "update", "deprecate"],
                },
                "name": {
                    "type": "string",
                    "description": "Skill name (lowercase kebab-case)",
                },
                "description": {
                    "type": "string",
                    "description": "Skill description for frontmatter",
                },
                "content": {
                    "type": "string",
                    "description": "Skill markdown body without frontmatter",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        description: str | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            return self._list_skills()
        if action == "read":
            return self._read_skill(name)
        if action == "create":
            return self._create_skill(name, description, content)
        if action == "update":
            return self._update_skill(name, description, content)
        if action == "deprecate":
            return self._deprecate_skill(name)
        return f"Error: unsupported action '{action}'"

    def _list_skills(self) -> str:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        skill_files = sorted(self.skills_dir.glob("*/SKILL.md"))
        if not skill_files:
            return "No workspace skills found."

        lines = ["Workspace skills:"]
        for p in skill_files:
            meta = self._parse_frontmatter(p.read_text(encoding="utf-8"))
            desc = meta.get("description", "")
            deprecated = meta.get("deprecated", "false").lower() == "true"
            status = "deprecated" if deprecated else "active"
            lines.append(f"- {p.parent.name} ({status}) - {desc}")
        return "\n".join(lines)

    def _read_skill(self, name: str | None) -> str:
        skill_path = self._skill_path(name)
        if isinstance(skill_path, str):
            return skill_path
        if not skill_path.exists():
            return f"Error: skill '{name}' not found in workspace"
        return skill_path.read_text(encoding="utf-8")

    def _create_skill(
        self,
        name: str | None,
        description: str | None,
        content: str | None,
    ) -> str:
        skill_path = self._skill_path(name)
        if isinstance(skill_path, str):
            return skill_path
        if not description:
            return "Error: description is required for create"
        if not content:
            return "Error: content is required for create"
        if skill_path.exists():
            return f"Error: skill '{name}' already exists in workspace"

        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(
            self._render_skill_doc(name or "", description, content), encoding="utf-8"
        )
        return f"Created workspace skill: {skill_path.parent.name}"

    def _update_skill(
        self,
        name: str | None,
        description: str | None,
        content: str | None,
    ) -> str:
        skill_path = self._skill_path(name)
        if isinstance(skill_path, str):
            return skill_path
        if not skill_path.exists():
            return f"Error: skill '{name}' not found in workspace"

        current = skill_path.read_text(encoding="utf-8")
        meta = self._parse_frontmatter(current)
        body = self._strip_frontmatter(current)

        if description:
            meta["description"] = description
        if content is not None:
            body = content

        rendered = self._render_with_meta(meta, body)
        skill_path.write_text(rendered, encoding="utf-8")
        return f"Updated workspace skill: {skill_path.parent.name}"

    def _deprecate_skill(self, name: str | None) -> str:
        skill_path = self._skill_path(name)
        if isinstance(skill_path, str):
            return skill_path
        if not skill_path.exists():
            return f"Error: skill '{name}' not found in workspace"

        current = skill_path.read_text(encoding="utf-8")
        meta = self._parse_frontmatter(current)
        body = self._strip_frontmatter(current)
        meta["deprecated"] = "true"
        if "description" not in meta:
            meta["description"] = skill_path.parent.name

        skill_path.write_text(self._render_with_meta(meta, body), encoding="utf-8")
        return f"Deprecated workspace skill: {skill_path.parent.name}"

    def _skill_path(self, name: str | None) -> Path | str:
        if not name:
            return "Error: name is required"
        if not _SKILL_NAME_RE.match(name):
            return "Error: invalid name, use lowercase kebab-case (2-63 chars)"
        return self.skills_dir / name / "SKILL.md"

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, str]:
        if not content.startswith("---\n"):
            return {}
        end = content.find("\n---\n", 4)
        if end < 0:
            return {}
        raw = content[4:end]
        meta: dict[str, str] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("\"'")
        return meta

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if not content.startswith("---\n"):
            return content.strip()
        end = content.find("\n---\n", 4)
        if end < 0:
            return content.strip()
        return content[end + 5 :].strip()

    def _render_skill_doc(self, name: str, description: str, body: str) -> str:
        meta = {"name": name, "description": description, "deprecated": "false"}
        return self._render_with_meta(meta, body)

    @staticmethod
    def _render_with_meta(meta: dict[str, str], body: str) -> str:
        lines = ["---"]
        for key in ("name", "description", "deprecated", "metadata"):
            if key in meta and meta[key] != "":
                lines.append(f'{key}: "{meta[key]}"')
        lines.append("---")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
        return "\n".join(lines)
