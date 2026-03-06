"""Tool manager for creating and evolving workspace Python tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from nanobot.agent.tools.base import Tool


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ToolManagerTool(Tool):
    """Manage workspace tools under workspace/tools."""

    def __init__(
        self, workspace: Path, reload_callback: Callable[[], dict[str, list[str]]]
    ):
        self.workspace = workspace
        self.tools_dir = workspace / "tools"
        self._reload = reload_callback

    @property
    def name(self) -> str:
        return "tool_manager"

    @property
    def description(self) -> str:
        return (
            "Manage workspace Python tools. Actions: list, read, create, update, "
            "deprecate, reload. Files are limited to workspace/tools/."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "create", "update", "deprecate", "reload"],
                },
                "name": {"type": "string", "description": "Tool file name without .py"},
                "content": {
                    "type": "string",
                    "description": "Full Python source code for create/update",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            return self._list()
        if action == "read":
            return self._read(name)
        if action == "create":
            return self._create(name, content)
        if action == "update":
            return self._update(name, content)
        if action == "deprecate":
            return self._deprecate(name)
        if action == "reload":
            return self._reload_report()
        return f"Error: unsupported action '{action}'"

    def _list(self) -> str:
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            f
            for f in self.tools_dir.glob("*.py")
            if not f.name.endswith(".disabled.py")
        )
        disabled = sorted(self.tools_dir.glob("*.disabled.py"))
        if not files and not disabled:
            return "No workspace tools found."
        lines = ["Workspace tools:"]
        for f in files:
            lines.append(f"- {f.name} (active)")
        for f in disabled:
            lines.append(f"- {f.name} (deprecated)")
        return "\n".join(lines)

    def _read(self, name: str | None) -> str:
        path = self._tool_path(name)
        if isinstance(path, str):
            return path
        if path.exists():
            return path.read_text(encoding="utf-8")
        disabled = self._disabled_path(path)
        if disabled.exists():
            return disabled.read_text(encoding="utf-8")
        return f"Error: tool '{name}' not found"

    def _create(self, name: str | None, content: str | None) -> str:
        path = self._tool_path(name)
        if isinstance(path, str):
            return path
        if not content:
            content = self._template(name or "tool_name")
        if path.exists() or self._disabled_path(path).exists():
            return f"Error: tool '{name}' already exists"

        self.tools_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        result = self._reload_report()
        return f"Created tool file: {path.name}\n{result}"

    def _update(self, name: str | None, content: str | None) -> str:
        path = self._tool_path(name)
        if isinstance(path, str):
            return path
        if not path.exists():
            return f"Error: tool '{name}' not found"
        if not content:
            return "Error: content is required for update"
        path.write_text(content, encoding="utf-8")
        result = self._reload_report()
        return f"Updated tool file: {path.name}\n{result}"

    def _deprecate(self, name: str | None) -> str:
        path = self._tool_path(name)
        if isinstance(path, str):
            return path
        if not path.exists():
            return f"Error: tool '{name}' not found"
        disabled = self._disabled_path(path)
        path.rename(disabled)
        result = self._reload_report()
        return f"Deprecated tool file: {disabled.name}\n{result}"

    def _reload_report(self) -> str:
        report = self._reload()
        loaded = report.get("loaded") or []
        errors = report.get("errors") or []
        lines = [f"Reloaded. loaded={len(loaded)}, errors={len(errors)}"]
        if loaded:
            lines.append("Loaded files: " + ", ".join(loaded))
        if errors:
            lines.append("Errors: " + " | ".join(errors))
        return "\n".join(lines)

    def _tool_path(self, name: str | None) -> Path | str:
        if not name:
            return "Error: name is required"
        if not _NAME_RE.match(name):
            return "Error: invalid name, use snake_case (2-64 chars)"
        return self.tools_dir / f"{name}.py"

    @staticmethod
    def _disabled_path(path: Path) -> Path:
        return path.with_name(path.name.replace(".py", ".disabled.py"))

    @staticmethod
    def _template(name: str) -> str:
        tool_name = name
        class_name = "".join(part.capitalize() for part in name.split("_")) + "Tool"
        return f'''"""Workspace tool: {tool_name}."""

from typing import Any

from nanobot.agent.tools.base import Tool


class {class_name}(Tool):
    @property
    def name(self) -> str:
        return "{tool_name}"

    @property
    def description(self) -> str:
        return "Describe what this tool does."

    @property
    def parameters(self) -> dict[str, Any]:
        return {{
            "type": "object",
            "properties": {{
                "input": {{"type": "string", "description": "Input text"}}
            }},
            "required": ["input"]
        }}

    async def execute(self, input: str, **kwargs: Any) -> str:
        return f"received: {{input}}"
'''
