"""Async SQLite session management for conversation history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

from nanobot.session.manager import Session
from nanobot.utils.helpers import ensure_dir


@dataclass
class AsyncSessionManager:
    """
    Manages conversation sessions using SQLite for async persistence.

    Sessions are stored in SQLite database with WAL mode enabled.
    An in-memory cache reduces database reads.
    """

    workspace: Path
    _db_path: Path = field(init=False)
    _db: aiosqlite.Connection | None = field(init=False, default=None)
    _cache: dict[str, Session] = field(init=False, default_factory=dict)
    _initialized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        sessions_dir = ensure_dir(self.workspace / "sessions")
        self._db_path = sessions_dir / "sessions.db"

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

        await self._db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_key ON sessions(key)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)
        """)

        await self._db.commit()

    async def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        await self._ensure_db()

        if key in self._cache:
            return self._cache[key]

        session = await self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    async def _load(self, key: str) -> Session | None:
        """Load a session from database."""
        if self._db is None:
            return None

        try:
            async with self._db.execute(
                "SELECT id, key, created_at, updated_at, last_consolidated, metadata "
                "FROM sessions WHERE key = ?",
                (key,)
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
                    last_consolidated=row["last_consolidated"] or 0
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
            (session_id,)
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

    async def save(self, session: Session) -> None:
        """Save a session to database."""
        await self._ensure_db()

        if self._db is None:
            raise RuntimeError("Database not initialized")

        async with self._db.execute("BEGIN"):
            session_id = await self._upsert_session(session)
            await self._replace_messages(session_id, session.messages)

        await self._db.commit()

        self._cache[session.key] = session

    async def _upsert_session(self, session: Session) -> int:
        """Insert or update a session record. Returns session_id."""
        if self._db is None:
            raise RuntimeError("Database not initialized")

        metadata_json = json.dumps(session.metadata, ensure_ascii=False) if session.metadata else None

        async with self._db.execute(
            "SELECT id FROM sessions WHERE key = ?",
            (session.key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is not None:
                session_id = row["id"]
                await self._db.execute("""
                    UPDATE sessions
                    SET updated_at = ?, last_consolidated = ?, metadata = ?
                    WHERE id = ?
                """, (
                    session.updated_at.isoformat(),
                    session.last_consolidated,
                    metadata_json,
                    session_id
                ))
                return session_id
            else:
                cursor = await self._db.execute("""
                    INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session.key,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.last_consolidated,
                    metadata_json
                ))
                return cursor.lastrowid or 0

    async def _replace_messages(self, session_id: int, messages: list[dict[str, Any]]) -> None:
        """Replace all messages for a session."""
        if self._db is None:
            return

        await self._db.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,)
        )

        for position, msg in enumerate(messages):
            await self._insert_message(session_id, position, msg)

    async def _insert_message(self, session_id: int, position: int, msg: dict[str, Any]) -> None:
        """Insert a single message."""
        if self._db is None:
            return

        role = msg.get("role", "user")
        content = msg.get("content")
        timestamp = msg.get("timestamp", datetime.now().isoformat())
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")
        tool_calls = msg.get("tool_calls")

        core_keys = {"role", "content", "timestamp", "tool_call_id", "name", "tool_calls"}
        extra = {k: v for k, v in msg.items() if k not in core_keys}

        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        extra_json = json.dumps(extra, ensure_ascii=False) if extra else None

        await self._db.execute("""
            INSERT INTO messages (session_id, position, role, content, timestamp,
                                   tool_call_id, name, tool_calls, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, position, role, content, timestamp,
            tool_call_id, name, tool_calls_json, extra_json
        ))

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

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
                sessions.append({
                    "key": row["key"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "path": str(self._db_path)
                })

        return sessions

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def __aenter__(self) -> AsyncSessionManager:
        await self._ensure_db()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
