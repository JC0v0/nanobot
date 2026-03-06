"""Runtime loader for workspace Python tools."""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


_TOOL_FILE_RE = re.compile(r"^[a-z0-9_]{2,64}\.py$")


class WorkspaceToolRuntime:
    """Load/unload tool plugins from workspace/tools."""

    def __init__(self, workspace: Path, registry: ToolRegistry):
        self.workspace = workspace
        self.registry = registry
        self.tools_dir = workspace / "tools"
        self._loaded_names: set[str] = set()

    def reload(self) -> dict[str, list[str]]:
        """Reload all workspace tools and return a report."""
        for name in list(self._loaded_names):
            self.registry.unregister(name)
        self._loaded_names.clear()

        self.tools_dir.mkdir(parents=True, exist_ok=True)
        loaded: list[str] = []
        errors: list[str] = []

        for path in sorted(self.tools_dir.glob("*.py")):
            if path.name.endswith(".disabled.py"):
                continue
            if not _TOOL_FILE_RE.match(path.name):
                errors.append(f"{path.name}: invalid filename")
                continue
            try:
                count = self._load_file(path)
                if count == 0:
                    errors.append(f"{path.name}: no Tool subclass found")
                else:
                    loaded.append(path.name)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                logger.warning("Failed to load workspace tool {}: {}", path, exc)

        return {"loaded": loaded, "errors": errors}

    def _load_file(self, path: Path) -> int:
        module_name = f"nanobot_workspace_tools_{path.stem}_{path.stat().st_mtime_ns}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if not spec or not spec.loader:
            raise RuntimeError("cannot create import spec")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        loaded = 0
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Tool or not issubclass(obj, Tool):
                continue
            instance = self._instantiate_tool(obj)
            if self.registry.has(instance.name):
                raise RuntimeError(f"tool name conflict: {instance.name}")
            self.registry.register(instance)
            self._loaded_names.add(instance.name)
            loaded += 1
        return loaded

    def _instantiate_tool(self, cls: type[Tool]) -> Tool:
        try:
            return cls(workspace=self.workspace)
        except TypeError:
            return cls()
