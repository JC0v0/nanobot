"""Unified persistence manager for sessions and tasks.

This module provides a single entry point for both session and task persistence,
using a shared SQLite database connection for data consistency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

from nanobot.session.task_store import TaskStore, TaskState, TaskProgress
from nanobot.utils.helpers import ensure_dir

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class Session:
    """
    A conversation session.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a user turn."""
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid orphaned tool_result blocks
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break

        out: list[dict[str, Any]] = []
        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name", "media"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)
        return out

    def _count_message_tokens(self, msg: dict[str, Any], encoding: Any) -> int:
        """Count tokens for a single message."""
        tokens = 4
        for key, value in msg.items():
            if value is None:
                continue
            if isinstance(value, str):
                tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        tokens += self._count_message_tokens(item, encoding)
        return tokens

    def get_history_by_tokens(
        self,
        max_tokens: int = 64000,
    ) -> list[dict[str, Any]]:
        """Return unconsolidated messages within token limit (latest-first packing)."""
        if tiktoken is None:
            logger.warning(
                "tiktoken not available, falling back to message count limit"
            )
            return self.get_history(max_messages=50)

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning(
                "Failed to load tiktoken encoding, falling back to message count"
            )
            return self.get_history(max_messages=50)

        unconsolidated = self.messages[self.last_consolidated :]

        if not unconsolidated:
            return []

        for i, m in enumerate(unconsolidated):
            if m.get("role") == "user":
                unconsolidated = unconsolidated[i:]
                break

        if not unconsolidated:
            return []

        total_tokens = 0
        result: list[dict[str, Any]] = []

        for m in reversed(unconsolidated):
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name", "media"):
                if k in m:
                    entry[k] = m[k]

            msg_tokens = self._count_message_tokens(entry, encoding)

            if total_tokens + msg_tokens <= max_tokens:
                result.insert(0, entry)
                total_tokens += msg_tokens
            else:
                break

        return result

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


@dataclass
class PersistenceManager:
    """
    Unified persistence manager for conversation sessions and tasks.

    Both sessions and tasks are stored in the same SQLite database with
    WAL mode enabled. In-memory caches reduce database reads.
    """

    workspace: Path
    _db_path: Path = field(init=False)
    _db: aiosqlite.Connection | None = field(init=False, default=None)
    _session_cache: dict[str, Session] = field(init=False, default_factory=dict)
    _initialized: bool = field(init=False, default=False)

    # Sub-stores
    task_store: TaskStore = field(init=False)

    def __post_init__(self) -> None:
        sessions_dir = ensure_dir(self.workspace / "sessions")
        self._db_path = sessions_dir / "sessions.db"
        self.task_store = TaskStore()

    async def _ensure_db(self) -> None:
        """Ensure database connection is established and schema is created."""
        if self._initialized and self._db is not None:
            return

        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA foreign_keys = ON")

        await self._create_schema()

        user_version = await self._get_user_version()
        if user_version == 0:
            await self._db.execute("PRAGMA user_version = 1")
            await self._db.commit()
        elif user_version == 1:
            # Migrate from version 1 to 2: add tasks table
            await self._migrate_v1_to_v2()
            await self._db.execute("PRAGMA user_version = 2")
            await self._db.commit()

        # Pass database connection to task_store
        self.task_store._set_db(self._db)

        self._initialized = True

    async def _get_user_version(self) -> int:
        """Get database user_version for migrations."""
        if self._db is None:
            return 0
        async with self._db.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def _create_schema(self) -> None:
        """Create database tables and indexes if they don't exist."""
        if self._db is None:
            return

        # Sessions table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                last_consolidated INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)

        # Messages table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp TIMESTAMP NOT NULL,
                tool_call_id TEXT,
                name TEXT,
                tool_calls TEXT,
                extra TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, position)
            )
        """)

        # Tasks table (new for v2)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                progress TEXT NOT NULL,
                messages TEXT,
                current_iteration INTEGER DEFAULT 0,
                tools_used TEXT,
                last_inbound TEXT,
                needs_recovery INTEGER DEFAULT 1,
                current_tool TEXT,
                current_tool_args TEXT,
                FOREIGN KEY (session_key) REFERENCES sessions(key) ON DELETE CASCADE,
                UNIQUE(session_key)
            )
        """)

        # Indexes
        await self._db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_key ON sessions(key)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)
        """)
        await self._db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_session_key ON tasks(session_key)
        """)

        await self._db.commit()

    async def _migrate_v1_to_v2(self) -> None:
        """Migrate database from version 1 to 2 (add tasks table)."""
        if self._db is None:
            return

        # Tasks table and index will be created by _create_schema
        # This is just a placeholder for any data migration needed
        logger.info("Migrating database from v1 to v2: adding tasks table")

    # ===== SessionStore methods =====

    async def get_or_create_session(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        await self._ensure_db()

        if key in self._session_cache:
            return self._session_cache[key]

        session = await self._load_session(key)
        if session is None:
            session = Session(key=key)

        self._session_cache[key] = session
        return session

    async def _load_session(self, key: str) -> Session | None:
        """Load a session from database."""
        if self._db is None:
            return None

        try:
            async with self._db.execute(
                "SELECT id, key, created_at, updated_at, last_consolidated, metadata "
                "FROM sessions WHERE key = ?",
                (key,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None

                session_id = row["id"]
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                messages = await self._load_messages(session_id)

                return Session(
                    key=row["key"],
                    messages=messages,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    metadata=metadata,
                    last_consolidated=row["last_consolidated"] or 0,
                )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    async def _load_messages(self, session_id: int) -> list[dict[str, Any]]:
        """Load all messages for a session."""
        if self._db is None:
            return []

        messages = []
        async with self._db.execute(
            "SELECT position, role, content, timestamp, tool_call_id, name, tool_calls, extra "
            "FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ) as cursor:
            async for row in cursor:
                msg = self._row_to_message(row)
                messages.append(msg)

        return messages

    def _row_to_message(self, row: aiosqlite.Row) -> dict[str, Any]:
        """Convert a database row to a message dict."""
        msg: dict[str, Any] = {
            "role": row["role"],
            "timestamp": row["timestamp"],
        }

        if row["content"] is not None:
            msg["content"] = row["content"]

        if row["tool_call_id"] is not None:
            msg["tool_call_id"] = row["tool_call_id"]

        if row["name"] is not None:
            msg["name"] = row["name"]

        if row["tool_calls"] is not None:
            msg["tool_calls"] = json.loads(row["tool_calls"])

        if row["extra"] is not None:
            extra = json.loads(row["extra"])
            msg.update(extra)

        return msg

    async def save_session(self, session: Session) -> None:
        """Save a session to database."""
        await self._ensure_db()

        if self._db is None:
            raise RuntimeError("Database not initialized")

        async with self._db.execute("BEGIN"):
            session_id = await self._upsert_session(session)
            await self._replace_messages(session_id, session.messages)

        await self._db.commit()

        self._session_cache[session.key] = session

    async def _upsert_session(self, session: Session) -> int:
        """Insert or update a session record. Returns session_id."""
        if self._db is None:
            raise RuntimeError("Database not initialized")

        metadata_json = (
            json.dumps(session.metadata, ensure_ascii=False)
            if session.metadata
            else None
        )

        async with self._db.execute(
            "SELECT id FROM sessions WHERE key = ?", (session.key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is not None:
                session_id = row["id"]
                await self._db.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?, last_consolidated = ?, metadata = ?
                    WHERE id = ?
                """,
                    (
                        session.updated_at.isoformat(),
                        session.last_consolidated,
                        metadata_json,
                        session_id,
                    ),
                )
                return session_id
            else:
                cursor = await self._db.execute(
                    """
                    INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        session.key,
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                        session.last_consolidated,
                        metadata_json,
                    ),
                )
                return cursor.lastrowid or 0

    async def _replace_messages(
        self, session_id: int, messages: list[dict[str, Any]]
    ) -> None:
        """Replace all messages for a session."""
        if self._db is None:
            return

        await self._db.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )

        for position, msg in enumerate(messages):
            await self._insert_message(session_id, position, msg)

    async def _insert_message(
        self, session_id: int, position: int, msg: dict[str, Any]
    ) -> None:
        """Insert a single message."""
        if self._db is None:
            return

        role = msg.get("role", "user")
        content = msg.get("content")
        timestamp = msg.get("timestamp", datetime.now().isoformat())
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")
        tool_calls = msg.get("tool_calls")

        core_keys = {
            "role",
            "content",
            "timestamp",
            "tool_call_id",
            "name",
            "tool_calls",
        }
        extra = {k: v for k, v in msg.items() if k not in core_keys}

        tool_calls_json = (
            json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        )
        extra_json = json.dumps(extra, ensure_ascii=False) if extra else None

        await self._db.execute(
            """
            INSERT INTO messages (session_id, position, role, content, timestamp,
                                   tool_call_id, name, tool_calls, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                position,
                role,
                content,
                timestamp,
                tool_call_id,
                name,
                tool_calls_json,
                extra_json,
            ),
        )

    def invalidate_session(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._session_cache.pop(key, None)

    async def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        await self._ensure_db()

        if self._db is None:
            return []

        sessions = []
        async with self._db.execute("""
            SELECT key, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
        """) as cursor:
            async for row in cursor:
                sessions.append(
                    {
                        "key": row["key"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "path": str(self._db_path),
                    }
                )

        return sessions

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def __aenter__(self) -> PersistenceManager:
        await self._ensure_db()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ===== Compatibility aliases =====

    async def get_or_create(self, key: str) -> Session:
        """Alias for get_or_create_session."""
        return await self.get_or_create_session(key)

    async def save(self, session: Session) -> None:
        """Alias for save_session."""
        await self.save_session(session)

    def invalidate(self, key: str) -> None:
        """Alias for invalidate_session."""
        self.invalidate_session(key)

    # ===== Task store wrapper methods (ensure DB initialized) =====

    async def create_task(
        self, session_key: str, last_inbound: dict[str, Any]
    ) -> TaskState:
        """Create a new task for a session with DB initialization."""
        await self._ensure_db()
        return await self.task_store.create_task(session_key, last_inbound)

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
        """Update a task's state with DB initialization."""
        await self._ensure_db()
        return await self.task_store.update_task(
            session_key,
            progress=progress,
            messages=messages,
            current_iteration=current_iteration,
            tools_used=tools_used,
            current_tool=current_tool,
            current_tool_args=current_tool_args,
        )

    async def complete_task(self, session_key: str) -> None:
        """Mark a task as completed with DB initialization."""
        await self._ensure_db()
        await self.task_store.complete_task(session_key)

    async def fail_task(self, session_key: str) -> None:
        """Mark a task as failed with DB initialization."""
        await self._ensure_db()
        await self.task_store.fail_task(session_key)

    async def get_task(self, session_key: str) -> TaskState | None:
        """Get a task by session key with DB initialization."""
        await self._ensure_db()
        return await self.task_store.get_task(session_key)

    async def get_pending_tasks(self) -> list[TaskState]:
        """Get all tasks that need recovery with DB initialization."""
        await self._ensure_db()
        return await self.task_store.get_pending_tasks()

    async def remove_task(self, session_key: str) -> bool:
        """Remove a task from the store with DB initialization."""
        await self._ensure_db()
        return await self.task_store.remove_task(session_key)

    async def clear_all_tasks(self) -> None:
        """Clear all tasks with DB initialization."""
        await self._ensure_db()
        await self.task_store.clear_all()
