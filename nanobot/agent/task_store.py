"""Task persistence for nanobot agent.

This module provides task persistence and recovery functionality.
Tasks are saved to disk and can be recovered after a restart.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


def _now_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


class TaskProgress(str, Enum):
    """Task execution progress stages."""
    PENDING = "pending"
    BUILDING_CONTEXT = "building_context"
    AGENT_LOOP = "agent_loop"
    WAITING_LLM = "waiting_llm"
    EXECUTING_TOOL = "executing_tool"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """Persistent task state."""
    session_key: str
    task_id: str
    created_at_ms: int
    updated_at_ms: int
    progress: TaskProgress

    # Message context - full list of messages
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Current execution position in agent loop
    current_iteration: int = 0
    tools_used: list[str] = field(default_factory=list)

    # The last inbound message (to reconstruct InboundMessage)
    last_inbound: dict[str, Any] | None = None

    # Flag indicating this task needs recovery
    needs_recovery: bool = False

    # Current tool being executed (if any)
    current_tool: str | None = None
    current_tool_args: dict[str, Any] | None = None


class TaskStore:
    """Manage persistent task storage.

    Similar to CronService, stores tasks in JSONL format.
    Each session can have at most one active task.
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._store: dict[str, TaskState] = {}  # session_key -> TaskState
        self._loaded = False

    def _load_store(self) -> None:
        """Load tasks from disk."""
        if self._loaded:
            return

        self._store = {}

        if self.store_path.exists():
            try:
                with open(self.store_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        task = TaskState(
                            session_key=data["session_key"],
                            task_id=data["task_id"],
                            created_at_ms=data["created_at_ms"],
                            updated_at_ms=data["updated_at_ms"],
                            progress=TaskProgress(data["progress"]),
                            messages=data.get("messages", []),
                            current_iteration=data.get("current_iteration", 0),
                            tools_used=data.get("tools_used", []),
                            last_inbound=data.get("last_inbound"),
                            needs_recovery=data.get("needs_recovery", False),
                            current_tool=data.get("current_tool"),
                            current_tool_args=data.get("current_tool_args"),
                        )
                        self._store[task.session_key] = task
                logger.info("Loaded {} tasks from {}", len(self._store), self.store_path)
            except Exception as e:
                logger.warning("Failed to load task store: {}", e)
                self._store = {}

        self._loaded = True

    def _save_store(self) -> None:
        """Save all tasks to disk."""
        if not self._loaded:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for task in self._store.values():
            data = {
                "session_key": task.session_key,
                "task_id": task.task_id,
                "created_at_ms": task.created_at_ms,
                "updated_at_ms": task.updated_at_ms,
                "progress": task.progress.value,
                "messages": task.messages,
                "current_iteration": task.current_iteration,
                "tools_used": task.tools_used,
                "last_inbound": task.last_inbound,
                "needs_recovery": task.needs_recovery,
                "current_tool": task.current_tool,
                "current_tool_args": task.current_tool_args,
            }
            lines.append(json.dumps(data, ensure_ascii=False))

        self.store_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def create_task(
        self,
        session_key: str,
        last_inbound: dict[str, Any],
    ) -> TaskState:
        """Create a new task for a session.

        If a task already exists for this session, it will be replaced.
        """
        self._load_store()

        task = TaskState(
            session_key=session_key,
            task_id=str(uuid.uuid4())[:8],
            created_at_ms=_now_ms(),
            updated_at_ms=_now_ms(),
            progress=TaskProgress.PENDING,
            last_inbound=last_inbound,
            needs_recovery=True,  # Mark as needing recovery until completed
        )

        self._store[session_key] = task
        self._save_store()
        logger.info("Created task {} for session {}", task.task_id, session_key)
        return task

    def update_task(
        self,
        session_key: str,
        progress: TaskProgress | None = None,
        messages: list[dict[str, Any]] | None = None,
        current_iteration: int | None = None,
        tools_used: list[str] | None = None,
        current_tool: str | None = None,
        current_tool_args: dict[str, Any] | None = None,
    ) -> TaskState | None:
        """Update a task's state."""
        self._load_store()

        task = self._store.get(session_key)
        if not task:
            return None

        task.updated_at_ms = _now_ms()

        if progress is not None:
            task.progress = progress
        if messages is not None:
            task.messages = messages
        if current_iteration is not None:
            task.current_iteration = current_iteration
        if tools_used is not None:
            task.tools_used = tools_used
        if current_tool is not None:
            task.current_tool = current_tool
        if current_tool_args is not None:
            task.current_tool_args = current_tool_args

        self._save_store()
        return task

    def complete_task(self, session_key: str) -> None:
        """Mark a task as completed and remove it from the store."""
        self._load_store()

        if session_key in self._store:
            task = self._store[session_key]
            logger.info("Completed task {} for session {}", task.task_id, session_key)
            del self._store[session_key]
            self._save_store()

    def fail_task(self, session_key: str) -> None:
        """Mark a task as failed but keep it for debugging."""
        self._load_store()

        task = self._store.get(session_key)
        if task:
            task.progress = TaskProgress.FAILED
            task.needs_recovery = False  # Don't auto-recover failed tasks
            task.updated_at_ms = _now_ms()
            self._save_store()
            logger.warning("Failed task {} for session {}", task.task_id, session_key)

    def get_task(self, session_key: str) -> TaskState | None:
        """Get a task by session key."""
        self._load_store()
        return self._store.get(session_key)

    def get_pending_tasks(self) -> list[TaskState]:
        """Get all tasks that need recovery."""
        self._load_store()
        return [
            task for task in self._store.values()
            if task.needs_recovery and task.progress != TaskProgress.COMPLETED
        ]

    def remove_task(self, session_key: str) -> bool:
        """Remove a task from the store."""
        self._load_store()

        if session_key in self._store:
            task = self._store[session_key]
            logger.info("Removed task {} for session {}", task.task_id, session_key)
            del self._store[session_key]
            self._save_store()
            return True
        return False

    def clear_all(self) -> None:
        """Clear all tasks."""
        self._load_store()
        count = len(self._store)
        self._store = {}
        self._save_store()
        logger.info("Cleared {} tasks", count)
