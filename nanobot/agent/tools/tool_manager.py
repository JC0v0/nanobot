"""Tool manager for creating and evolving workspace Python tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import yaml
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.schema import MCPServerConfig


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ToolManagerTool(Tool):
    """Manage workspace tools under workspace/tools and MCP servers."""

    def __init__(
        self,
        workspace: Path,
        reload_callback: Callable[[], dict[str, list[str]]],
        reload_mcp_callback: Callable[[], dict[str, Any]] | None = None,
    ):
        self.workspace = workspace
        self.tools_dir = workspace / "tools"
        self.mcp_config_path = workspace / "mcp_servers.yaml"
        self._reload = reload_callback
        self._reload_mcp = reload_mcp_callback

    @property
    def name(self) -> str:
        return "tool_manager"

    @property
    def description(self) -> str:
        return (
            "Manage workspace Python tools and MCP servers. Tool actions: list, read, create, "
            "update, deprecate, reload. MCP actions: mcp_list, mcp_add, mcp_remove, mcp_reload. "
            "Files are limited to workspace/tools/. MCP config stored in workspace/mcp_servers.yaml."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list",
                        "read",
                        "create",
                        "update",
                        "deprecate",
                        "reload",
                        "mcp_list",
                        "mcp_add",
                        "mcp_remove",
                        "mcp_reload",
                    ],
                },
                "name": {"type": "string", "description": "Tool file name or MCP server name"},
                "content": {
                    "type": "string",
                    "description": "Full Python source code for create/update, or MCP config YAML",
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
        if action == "mcp_list":
            return self._mcp_list()
        if action == "mcp_add":
            return self._mcp_add(name, content)
        if action == "mcp_remove":
            return self._mcp_remove(name)
        if action == "mcp_reload":
            return self._mcp_reload()
        return f"Error: unsupported action '{action}'"

    def _list(self) -> str:
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            f for f in self.tools_dir.glob("*.py") if not f.name.endswith(".disabled.py")
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

    def _load_mcp_config(self) -> dict[str, MCPServerConfig]:
        if not self.mcp_config_path.exists():
            return {}
        try:
            data = yaml.safe_load(self.mcp_config_path.read_text(encoding="utf-8"))
            if not data:
                return {}
            result = {}
            for name, cfg in data.items():
                if isinstance(cfg, dict):
                    result[name] = MCPServerConfig(**cfg)
            return result
        except Exception as e:
            logger.warning("Failed to load MCP config: {}", e)
            return {}

    def _save_mcp_config(self, servers: dict[str, MCPServerConfig]) -> None:
        data = {}
        for name, cfg in servers.items():
            data[name] = {
                "command": cfg.command,
                "args": cfg.args,
                "env": cfg.env,
                "url": cfg.url,
                "headers": cfg.headers,
                "tool_timeout": cfg.tool_timeout,
            }
        self.mcp_config_path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def _mcp_list(self) -> str:
        servers = self._load_mcp_config()
        if not servers:
            return "No MCP servers configured. Use mcp_add to add one."
        lines = ["MCP servers:"]
        for name, cfg in servers.items():
            if cfg.url:
                lines.append(f"- {name}: url={cfg.url}")
            else:
                lines.append(f"- {name}: {cfg.command} {' '.join(cfg.args)}")
        return "\n".join(lines)

    def _mcp_add(self, name: str | None, content: str | None) -> str:
        if not name:
            return "Error: name is required for mcp_add"
        servers = self._load_mcp_config()
        if name in servers:
            return f"Error: MCP server '{name}' already exists. Use mcp_remove first."
        if not content:
            content = self._mcp_template(name)
        try:
            cfg_data = yaml.safe_load(content)
            if not isinstance(cfg_data, dict):
                return "Error: invalid MCP config format, expected YAML dict"
            servers[name] = MCPServerConfig(**cfg_data)
            self._save_mcp_config(servers)
            result = self._mcp_reload()
            return f"Added MCP server: {name}\n{result}"
        except Exception as e:
            return f"Error: failed to add MCP server: {e}"

    def _mcp_remove(self, name: str | None) -> str:
        if not name:
            return "Error: name is required for mcp_remove"
        servers = self._load_mcp_config()
        if name not in servers:
            return f"Error: MCP server '{name}' not found"
        del servers[name]
        self._save_mcp_config(servers)
        result = self._mcp_reload()
        return f"Removed MCP server: {name}\n{result}"

    def _mcp_reload(self) -> str:
        if not self._reload_mcp:
            return "Error: MCP reload not available"
        try:
            report = self._reload_mcp()
            loaded = report.get("loaded") or []
            errors = report.get("errors") or []
            lines = [f"MCP reloaded. servers={len(loaded)}, errors={len(errors)}"]
            if loaded:
                lines.append("Connected: " + ", ".join(loaded))
            if errors:
                lines.append("Errors: " + " | ".join(errors))
            return "\n".join(lines)
        except Exception as e:
            return f"Error: MCP reload failed: {e}"

    @staticmethod
    def _mcp_template(name: str) -> str:
        return f"""command: npx
args:
  - "-y"
  - "@modelcontextprotocol/server-filesystem"
  - "/path/to/directory"
tool_timeout: 30"""

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
