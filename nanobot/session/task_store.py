"""Task persistence with SQLite backend.

This module provides task persistence and recovery functionality.
Tasks are saved to SQLite database and can be recovered after a restart.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
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


@dataclass
class TaskStore:
    """Manage persistent task storage with SQLite backend.

    Each session can have at most one active task.
    Requires a PersistenceManager parent to provide database access.
    """

    # Note: _db and _store are set by PersistenceManager during initialization
    _db: Any = field(init=False, default=None)
    _store: dict[str, TaskState] = field(init=False, default_factory=dict)
    _loaded: bool = field(init=False, default=False)

    def _set_db(self, db: Any) -> None:
        """Set the database connection (called by PersistenceManager)."""
        self._db = db

    async def _load_store(self) -> None:
        """Load tasks from database (lazy load on first access)."""
        if self._loaded:
            return

        self._store = {}

        if self._db is None:
            self._loaded = True
            return

        try:
            async with self._db.execute("""
                SELECT session_key, task_id, created_at_ms, updated_at_ms, progress,
                       messages, current_iteration, tools_used, last_inbound,
                       needs_recovery, current_tool, current_tool_args
                FROM tasks
            """) as cursor:
                async for row in cursor:
                    task = TaskState(
                        session_key=row[0],
                        task_id=row[1],
                        created_at_ms=row[2],
                        updated_at_ms=row[3],
                        progress=TaskProgress(row[4]),
                        messages=json.loads(row[5]) if row[5] else [],
                        current_iteration=row[6] or 0,
                        tools_used=json.loads(row[7]) if row[7] else [],
                        last_inbound=json.loads(row[8]) if row[8] else None,
                        needs_recovery=bool(row[9]),
                        current_tool=row[10],
                        current_tool_args=json.loads(row[11]) if row[11] else None,
                    )
                    self._store[task.session_key] = task
            if self._store:
                logger.info("Loaded {} tasks from database", len(self._store))
        except Exception as e:
            # Don't warn if DB is not initialized yet - this is expected
            # PersistenceManager will initialize it when needed
            self._store = {}

        self._loaded = True

    async def create_task(
        self,
        session_key: str,
        last_inbound: dict[str, Any],
    ) -> TaskState:
        """Create a new task for a session.

        If a task already exists for this session, it will be replaced.
        """
        await self._load_store()

        if self._db is None:
            raise RuntimeError("Database not initialized")

        task = TaskState(
            session_key=session_key,
            task_id=str(uuid.uuid4())[:8],
            created_at_ms=_now_ms(),
            updated_at_ms=_now_ms(),
            progress=TaskProgress.PENDING,
            last_inbound=last_inbound,
            needs_recovery=True,  # Mark as needing recovery until completed
        )

        # Replace existing task if exists
        if session_key in self._store:
            await self._db.execute("DELETE FROM tasks WHERE session_key = ?", (session_key,))

        messages_json = json.dumps(task.messages, ensure_ascii=False) if task.messages else None
        tools_used_json = json.dumps(task.tools_used, ensure_ascii=False) if task.tools_used else None
        last_inbound_json = json.dumps(task.last_inbound, ensure_ascii=False) if task.last_inbound else None
        current_tool_args_json = json.dumps(task.current_tool_args, ensure_ascii=False) if task.current_tool_args else None

        await self._db.execute("""
            INSERT INTO tasks (
                session_key, task_id, created_at_ms, updated_at_ms, progress,
                messages, current_iteration, tools_used, last_inbound,
                needs_recovery, current_tool, current_tool_args
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.session_key,
            task.task_id,
            task.created_at_ms,
            task.updated_at_ms,
            task.progress.value,
            messages_json,
            task.current_iteration,
            tools_used_json,
            last_inbound_json,
            1 if task.needs_recovery else 0,
            task.current_tool,
            current_tool_args_json,
        ))
        await self._db.commit()

        self._store[session_key] = task
        logger.info("Created task {} for session {}", task.task_id, session_key)
        return task

    async def update_task(
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
        await self._load_store()

        task = self._store.get(session_key)
        if not task:
            return None

        if self._db is None:
            raise RuntimeError("Database not initialized")

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

        # Build update statement dynamically
        updates = []
        params = []

        updates.append("updated_at_ms = ?")
        params.append(task.updated_at_ms)

        if progress is not None:
            updates.append("progress = ?")
            params.append(progress.value)
        if messages is not None:
            updates.append("messages = ?")
            params.append(json.dumps(messages, ensure_ascii=False) if messages else None)
        if current_iteration is not None:
            updates.append("current_iteration = ?")
            params.append(current_iteration)
        if tools_used is not None:
            updates.append("tools_used = ?")
            params.append(json.dumps(tools_used, ensure_ascii=False) if tools_used else None)
        if current_tool is not None:
            updates.append("current_tool = ?")
            params.append(current_tool)
        if current_tool_args is not None:
            updates.append("current_tool_args = ?")
            params.append(json.dumps(current_tool_args, ensure_ascii=False) if current_tool_args else None)

        params.append(session_key)

        await self._db.execute(f"""
            UPDATE tasks SET {', '.join(updates)} WHERE session_key = ?
        """, params)
        await self._db.commit()

        return task

    async def complete_task(self, session_key: str) -> None:
        """Mark a task as completed and remove it from the store."""
        await self._load_store()

        if self._db is None:
            raise RuntimeError("Database not initialized")

        if session_key in self._store:
            task = self._store[session_key]
            logger.info("Completed task {} for session {}", task.task_id, session_key)
            del self._store[session_key]
            await self._db.execute("DELETE FROM tasks WHERE session_key = ?", (session_key,))
            await self._db.commit()

    async def fail_task(self, session_key: str) -> None:
        """Mark a task as failed but keep it for debugging."""
        await self._load_store()

        task = self._store.get(session_key)
        if task:
            task.progress = TaskProgress.FAILED
            task.needs_recovery = False  # Don't auto-recover failed tasks
            task.updated_at_ms = _now_ms()

            if self._db is None:
                raise RuntimeError("Database not initialized")

            await self._db.execute("""
                UPDATE tasks SET progress = ?, needs_recovery = 0, updated_at_ms = ?
                WHERE session_key = ?
            """, (TaskProgress.FAILED.value, task.updated_at_ms, session_key))
            await self._db.commit()

            logger.warning("Failed task {} for session {}", task.task_id, session_key)

    async def get_task(self, session_key: str) -> TaskState | None:
        """Get a task by session key."""
        await self._load_store()
        return self._store.get(session_key)

    async def get_pending_tasks(self) -> list[TaskState]:
        """Get all tasks that need recovery."""
        await self._load_store()
        return [
            task for task in self._store.values()
            if task.needs_recovery and task.progress != TaskProgress.COMPLETED
        ]

    async def remove_task(self, session_key: str) -> bool:
        """Remove a task from the store."""
        await self._load_store()

        if self._db is None:
            raise RuntimeError("Database not initialized")

        if session_key in self._store:
            task = self._store[session_key]
            logger.info("Removed task {} for session {}", task.task_id, session_key)
            del self._store[session_key]
            await self._db.execute("DELETE FROM tasks WHERE session_key = ?", (session_key,))
            await self._db.commit()
            return True
        return False

    async def clear_all(self) -> None:
        """Clear all tasks."""
        await self._load_store()

        if self._db is None:
            raise RuntimeError("Database not initialized")

        count = len(self._store)
        self._store = {}
        await self._db.execute("DELETE FROM tasks")
        await self._db.commit()
        logger.info("Cleared {} tasks", count)
