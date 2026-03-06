"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.evolution import EvolutionEngine
from nanobot.agent.memory import GraphMemoryStore as MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.token_budget import TokenBudgetEstimator
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.skill_manager import SkillManagerTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.tool_manager import ToolManagerTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.tools.workspace_runtime import WorkspaceToolRuntime
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.store import PersistenceManager
from nanobot.session.store import Session
from nanobot.session.task_store import TaskProgress, TaskState

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        ExecToolConfig,
        SelfEvolutionConfig,
    )
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tokens_window: int = 64000,
        memory_window: int | None = None,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: PersistenceManager | None = None,
        persistence_manager: PersistenceManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        self_evolution_config: SelfEvolutionConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, SelfEvolutionConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokens_window = max(1, tokens_window)
        self.consolidation_window = (
            max(2, memory_window) if memory_window is not None else 10
        )
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.self_evolution_config = self_evolution_config or SelfEvolutionConfig()

        self.context = ContextBuilder(workspace)
        self.persistence = persistence_manager or PersistenceManager(workspace)
        # For backward compatibility, keep self.sessions pointing to persistence
        self.sessions = self.persistence
        self._using_async_sessions = True
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._token_budget = TokenBudgetEstimator()
        self._last_context_tokens: dict[str, int] = {}
        self._consolidating: set[str] = (
            set()
        )  # Session keys with consolidation in progress
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        # Session processing state for interruptible handling
        self._session_tasks: dict[str, asyncio.Task] = {}  # session_key -> Task
        self._session_cancel_events: dict[
            str, asyncio.Event
        ] = {}  # session_key -> Event
        self._evolution_tasks: set[asyncio.Task] = set()
        self.workspace_tools = WorkspaceToolRuntime(self.workspace, self.tools)
        self.evolution = (
            EvolutionEngine(
                workspace=self.workspace,
                provider=self.provider,
                model=self.model,
                min_confidence=self.self_evolution_config.min_confidence,
            )
            if self.self_evolution_config.enabled
            else None
        )
        # Task persistence is now part of persistence
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            )
        )
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(SkillManagerTool(workspace=self.workspace))
        self.tools.register(
            ToolManagerTool(
                workspace=self.workspace,
                reload_callback=self.workspace_tools.reload,
            )
        )
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        report = self.workspace_tools.reload()
        if report["errors"]:
            logger.warning("Workspace tools loaded with errors: {}", report["errors"])

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error(
                "Failed to connect MCP servers (will retry next message): {}", e
            )
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None
    ) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return (
                f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
            )

        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        cancel_event: asyncio.Event | None = None,
        session_key: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            # Check for cancellation before each iteration
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError()

            iteration += 1

            # Update task progress
            if session_key:
                await self._update_task_progress(
                    session_key,
                    TaskProgress.WAITING_LLM,
                    messages=messages,
                    current_iteration=iteration,
                    tools_used=tools_used,
                )

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                cancel_event=cancel_event,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(
                        self._tool_hint(response.tool_calls), tool_hint=True
                    )

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    # Check for cancellation before each tool call
                    if cancel_event and cancel_event.is_set():
                        raise asyncio.CancelledError()

                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])

                    # Update task progress for tool execution
                    if session_key:
                        await self._update_task_progress(
                            session_key,
                            TaskProgress.EXECUTING_TOOL,
                            current_tool=tool_call.name,
                            current_tool_args=tool_call.arguments,
                        )

                    result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = self._strip_think(response.content)
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        # Recover pending tasks on startup
        await self._recover_pending_tasks()

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                # Handle message in background task to allow interruptions
                asyncio.create_task(self._handle_incoming_message(msg))
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    async def _prepare_messages_with_preflight(
        self,
        session: Session,
        msg: InboundMessage,
        cancel_event: asyncio.Event | None = None,
        max_rebuilds: int = 2,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build context messages and compress memory if context tokens exceed window."""
        rebuild_count = 0

        while True:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError()

            history = session.get_history_by_tokens(
                max_tokens=self.tokens_window,
            )
            initial_messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )

            context_tokens = self._token_budget.count_messages(
                initial_messages, self.model
            )
            self._last_context_tokens[session.key] = context_tokens

            if context_tokens <= self.tokens_window:
                return history, initial_messages

            if rebuild_count >= max_rebuilds:
                logger.warning(
                    "Context still over tokensWindow after {} rebuilds: {} > {}",
                    rebuild_count,
                    context_tokens,
                    self.tokens_window,
                )
                return history, initial_messages

            logger.info(
                "Context over tokensWindow ({} > {}), consolidating memory (attempt {}/{})",
                context_tokens,
                self.tokens_window,
                rebuild_count + 1,
                max_rebuilds,
            )

            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    ok = await self._consolidate_memory(session)
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            if not ok:
                logger.warning(
                    "Memory consolidation failed during token preflight, continuing with current context"
                )
                return history, initial_messages

            rebuild_count += 1

    async def _get_or_create_session(self, key: str) -> Session:
        """Get or create a session, handling both sync and async managers."""
        if self._using_async_sessions:
            return await self.sessions.get_or_create(key)  # type: ignore
        else:
            return self.sessions.get_or_create(key)  # type: ignore

    def _evolution_log_path(self) -> Path:
        if self.evolution:
            return self.evolution.log_file
        return self.workspace / "evolution" / "proposals.jsonl"

    def _read_evolution_records(self) -> list[dict[str, Any]]:
        path = self._evolution_log_path()
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    def _append_evolution_record(self, record: dict[str, Any]) -> None:
        path = self._evolution_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _collect_proposals(self) -> list[dict[str, Any]]:
        proposals: dict[str, dict[str, Any]] = {}
        for rec in self._read_evolution_records():
            pid = rec.get("id")
            if not isinstance(pid, str) or not pid:
                continue
            rtype = rec.get("type")
            if rtype == "proposal":
                proposals[pid] = dict(rec)
                continue
            if pid not in proposals:
                continue
            if rtype == "apply":
                proposals[pid]["status"] = rec.get(
                    "status", proposals[pid].get("status")
                )
                proposals[pid]["auto_applied"] = rec.get("auto_applied", [])
            elif rtype == "decision":
                proposals[pid]["status"] = rec.get(
                    "status", proposals[pid].get("status")
                )
                proposals[pid]["decision_reason"] = rec.get("reason", "")
                proposals[pid]["decided_at"] = rec.get("created_at", "")

        items = list(proposals.values())
        items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return items

    async def _apply_evolution_actions(
        self, actions: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for action in actions:
            kind = str(action.get("kind", ""))
            name = str(action.get("name", ""))
            content = str(action.get("content", ""))
            description = str(action.get("description", ""))
            if not kind or not name:
                continue

            if kind == "skill_update":
                out = await self.tools.execute(
                    "skill_manager",
                    {
                        "action": "update",
                        "name": name,
                        "description": description,
                        "content": content,
                    },
                )
                if out.startswith("Error: skill") and "not found" in out:
                    out = await self.tools.execute(
                        "skill_manager",
                        {
                            "action": "create",
                            "name": name,
                            "description": description or name,
                            "content": content,
                        },
                    )
                    kind = "skill_create"
            elif kind == "skill_create":
                out = await self.tools.execute(
                    "skill_manager",
                    {
                        "action": "create",
                        "name": name,
                        "description": description or name,
                        "content": content,
                    },
                )
            elif kind == "skill_deprecate":
                out = await self.tools.execute(
                    "skill_manager", {"action": "deprecate", "name": name}
                )
            elif kind == "tool_update":
                out = await self.tools.execute(
                    "tool_manager",
                    {"action": "update", "name": name, "content": content},
                )
                if out.startswith("Error: tool") and "not found" in out:
                    out = await self.tools.execute(
                        "tool_manager",
                        {"action": "create", "name": name, "content": content},
                    )
                    kind = "tool_create"
            elif kind == "tool_create":
                out = await self.tools.execute(
                    "tool_manager",
                    {"action": "create", "name": name, "content": content},
                )
            elif kind == "tool_deprecate":
                out = await self.tools.execute(
                    "tool_manager", {"action": "deprecate", "name": name}
                )
            else:
                out = f"Skipped: unknown kind '{kind}'"

            results.append({"kind": kind, "name": name, "result": out})
        return results

    async def _handle_evolution_command(
        self,
        msg: InboundMessage,
    ) -> OutboundMessage:
        raw = msg.content.strip()
        parts = raw.split(maxsplit=3)

        if len(parts) == 1 or parts[1] in {"help", "-h", "--help"}:
            content = (
                "🧬 evolution commands:\n"
                "/evolution list [N] — 查看最近提案\n"
                "/evolution show <id> — 查看提案详情\n"
                "/evolution approve <id> — 批准并执行提案\n"
                "/evolution reject <id> [reason] — 拒绝提案"
            )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content
            )

        action = parts[1].lower()
        proposals = self._collect_proposals()

        if action == "list":
            limit = 10
            if len(parts) >= 3:
                try:
                    limit = max(1, int(parts[2]))
                except ValueError:
                    pass
            if not proposals:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="No evolution proposals found.",
                )
            rows = ["Evolution proposals:"]
            for p in proposals[:limit]:
                rows.append(
                    f"- {p.get('id', '-')} | {p.get('status', 'pending')} | conf={float(p.get('confidence', 0.0)):.2f} | actions={len(p.get('actions') or [])}"
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(rows)
            )

        if len(parts) < 3:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Usage: /evolution show|approve|reject <proposal_id>",
            )

        proposal_id = parts[2]
        proposal = next((p for p in proposals if p.get("id") == proposal_id), None)
        if not proposal:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Proposal not found: {proposal_id}",
            )

        if action == "show":
            lines = [
                f"Proposal: {proposal_id}",
                f"- status: {proposal.get('status', 'pending')}",
                f"- confidence: {float(proposal.get('confidence', 0.0)):.2f}",
                f"- created_at: {proposal.get('created_at', '-')}",
            ]
            summary = str(proposal.get("summary", "")).strip()
            if summary:
                lines.append(f"- summary: {summary}")
            for idx, a in enumerate(proposal.get("actions") or [], start=1):
                lines.append(
                    f"  {idx}. {a.get('kind')}:{a.get('name')} (risk={a.get('risk', '-')})"
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines)
            )

        if action == "approve":
            results = await self._apply_evolution_actions(proposal.get("actions") or [])
            has_errors = any(
                str(r.get("result", "")).startswith("Error") for r in results
            )
            status = "approved_with_errors" if has_errors else "approved"
            self._append_evolution_record(
                {
                    "type": "decision",
                    "id": proposal_id,
                    "status": status,
                    "created_at": datetime.now().isoformat(),
                    "results": results,
                }
            )
            lines = [f"Approved: {proposal_id} ({status})"]
            for r in results:
                head = str(r.get("result", "")).splitlines()[0]
                lines.append(f"- {r.get('kind')}:{r.get('name')} -> {head}")
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines)
            )

        if action == "reject":
            reason = parts[3] if len(parts) >= 4 else ""
            self._append_evolution_record(
                {
                    "type": "decision",
                    "id": proposal_id,
                    "status": "rejected",
                    "reason": reason,
                    "created_at": datetime.now().isoformat(),
                }
            )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Rejected: {proposal_id}",
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Unknown evolution action: {action}",
        )

    def _trigger_self_evolution(
        self,
        msg: InboundMessage,
        final_content: str,
        tools_used: list[str],
        all_msgs: list[dict[str, Any]],
    ) -> None:
        """Launch async self-evolution analysis without blocking user response."""
        if not self.evolution:
            return
        if msg.channel == "system":
            return
        if msg.content.strip().startswith("/"):
            return

        task = asyncio.create_task(
            self._run_self_evolution(msg, final_content, tools_used, all_msgs)
        )
        self._evolution_tasks.add(task)
        task.add_done_callback(self._evolution_tasks.discard)

    async def _run_self_evolution(
        self,
        msg: InboundMessage,
        final_content: str,
        tools_used: list[str],
        all_msgs: list[dict[str, Any]],
    ) -> None:
        """Generate and optionally apply safe evolution proposals."""
        if not self.evolution:
            return

        try:
            proposal = await self.evolution.propose(
                user_message=msg.content,
                final_response=final_content,
                tools_used=tools_used,
                messages=all_msgs,
                session_key=msg.session_key,
            )
            if not proposal:
                return

            if self.self_evolution_config.mode == "auto_safe":
                await self._apply_safe_evolution_actions(proposal)

            if self.self_evolution_config.notify:
                text = self._format_evolution_notice(proposal)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=text,
                        metadata={"_evolution": True},
                    )
                )
        except Exception:
            logger.exception("Self-evolution failed")

    async def _apply_safe_evolution_actions(self, proposal: dict[str, Any]) -> None:
        """Apply low-risk actions automatically via tool manager skills."""
        actions = proposal.get("actions") or []
        applied: list[dict[str, Any]] = []

        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("risk") != "low":
                continue

            kind = str(action.get("kind", ""))
            name = str(action.get("name", ""))
            content = str(action.get("content", ""))
            description = str(action.get("description", ""))
            if not kind or not name:
                continue

            if kind == "skill_update":
                result = await self.tools.execute(
                    "skill_manager",
                    {
                        "action": "update",
                        "name": name,
                        "description": description,
                        "content": content,
                    },
                )
                if result.startswith("Error: skill") and "not found" in result:
                    result = await self.tools.execute(
                        "skill_manager",
                        {
                            "action": "create",
                            "name": name,
                            "description": description or name,
                            "content": content,
                        },
                    )
                    kind = "skill_create"
                applied.append({"kind": kind, "name": name, "result": result})

        proposal["auto_applied"] = applied
        proposal["status"] = (
            "auto_applied" if applied else proposal.get("status", "pending")
        )
        evolution_engine = self.evolution
        if evolution_engine is not None:
            evolution_engine.append_log({"type": "apply", **proposal})

    @staticmethod
    def _format_evolution_notice(proposal: dict[str, Any]) -> str:
        actions = proposal.get("actions") or []
        auto_applied = proposal.get("auto_applied") or []
        lines = [
            "[Self-Evolution] 已生成改进提案",
            f"- id: {proposal.get('id', '-')}",
            f"- confidence: {proposal.get('confidence', 0):.2f}",
            f"- actions: {len(actions)}",
        ]
        if auto_applied:
            lines.append(f"- auto applied: {len(auto_applied)}")
        return "\n".join(lines)

    async def _save_session(self, session: Session) -> None:
        """Save a session, handling both sync and async managers."""
        if self._using_async_sessions:
            await self.sessions.save(session)  # type: ignore
        else:
            result = self.sessions.save(session)  # type: ignore
            if asyncio.iscoroutine(result):
                await result

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1)
                if ":" in msg.chat_id
                else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = await self._get_or_create_session(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history, messages = await self._prepare_messages_with_preflight(
                session, msg
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            await self._save_session(session)
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(
            "Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview
        )

        key = session_key or msg.session_key
        session = await self._get_or_create_session(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd.startswith("/evolution"):
            return await self._handle_evolution_command(msg)
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated :]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            await self._save_session(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="New session started."
            )
        if cmd == "/help":
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="🐈 nanobot commands:\n/new — Start a new conversation\n/status — Show current context info\n/evolution — Manage self-evolution proposals\n/help — Show available commands",
            )

        if cmd == "/status":
            return await self._handle_status_command(session, msg)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history, initial_messages = await self._prepare_messages_with_preflight(
            session,
            msg,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        final_content, tools_used, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        await self._save_session(session)

        self._trigger_self_evolution(msg, final_content, tools_used, all_msgs)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    _TOOL_RESULT_MAX_CHARS = 500

    def _truncate_tool_result(self, content: str) -> str:
        """Truncate tool result content if it's too long."""
        if len(content) > self._TOOL_RESULT_MAX_CHARS:
            return content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
        return content

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = (
                        content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                    )
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session,
            self.provider,
            self.model,
            archive_all=archive_all,
            memory_window=self.consolidation_window,
        )

    async def _handle_status_command(
        self, session: Session, msg: InboundMessage
    ) -> OutboundMessage:
        """Handle /status command to show context info."""
        memory_path = self.workspace / "memory" / "MEMORY.md"
        history_path = self.workspace / "memory" / "HISTORY.md"

        msg_count = len(session.messages)
        unconsolidated = msg_count - session.last_consolidated

        lines = [
            "📊 **Context Status**",
            "",
            "**Token Quota:**",
            f"- tokensWindow: {self.tokens_window:,}",
            "",
            "**Current Session:**",
            f"- Total messages: {msg_count}",
            f"- Unconsolidated: {unconsolidated}",
            f"- Last consolidated: {session.last_consolidated}",
            f"- Last context tokens: {self._last_context_tokens.get(session.key, 0):,}",
            "",
            "**Memory Files:**",
        ]

        if memory_path.exists():
            size = memory_path.stat().st_size
            lines.append(f"- MEMORY.md: ✓ ({size:,} bytes)")
        else:
            lines.append("- MEMORY.md: not exists")

        if history_path.exists():
            size = history_path.stat().st_size
            lines.append(f"- HISTORY.md: ✓ ({size:,} bytes)")
        else:
            lines.append("- HISTORY.md: not exists")

        content = "\n".join(lines)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id, content=content
        )
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress
        )
        return response.content if response else ""

    # ===== Interruptible message handling methods =====

    def _add_user_message_to_session(
        self, session: Session, msg: InboundMessage
    ) -> None:
        """Add a user message to the session without processing it."""
        from datetime import datetime

        entry: dict[str, Any] = {
            "role": "user",
            "content": msg.content,
            "timestamp": datetime.now().isoformat(),
        }
        if msg.media:
            entry["media"] = msg.media
        session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _send_interruption_notice(self, msg: InboundMessage) -> None:
        """Send an interruption notice to the user."""
        notice = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="收到新消息，重新思考...",
            metadata=dict(msg.metadata or {}),
        )
        await self.bus.publish_outbound(notice)

    async def _process_system_message(self, msg: InboundMessage) -> None:
        """Process a system message (keeps original logic)."""
        channel, chat_id = (
            msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
        )
        logger.info("Processing system message from {}", msg.sender_id)
        key = f"{channel}:{chat_id}"
        session = await self._get_or_create_session(key)
        self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
        history, messages = await self._prepare_messages_with_preflight(
            session,
            msg,
        )
        final_content, _, all_msgs = await self._run_agent_loop(messages)
        self._save_turn(session, all_msgs, 1 + len(history))
        await self._save_session(session)
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )
        )

    async def _handle_incoming_message(self, msg: InboundMessage) -> None:
        """
        Handle an incoming message: add to session immediately,
        interrupt ongoing processing, and start new processing.
        """
        session_key = msg.session_key

        # System messages get special handling
        if msg.channel == "system":
            await self._process_system_message(msg)
            return

        # 1. Get or create session, add user message immediately
        session = await self._get_or_create_session(session_key)
        self._add_user_message_to_session(session, msg)
        await self._save_session(session)

        # 2. Create persistent task
        await self.persistence.create_task(
            session_key,
            last_inbound=self._inbound_to_dict(msg),
        )
        await self._update_task_progress(session_key, TaskProgress.PENDING)

        # 3. Check if there's ongoing processing for this session
        if session_key in self._session_tasks:
            logger.info(f"Interrupting ongoing processing for session {session_key}")

            # Send interruption notice
            await self._send_interruption_notice(msg)

            # Cancel the old task
            old_task = self._session_tasks[session_key]
            cancel_event = self._session_cancel_events.get(session_key)

            if cancel_event:
                cancel_event.set()

            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass  # Expected cancellation
            except Exception as e:
                logger.error(f"Error in cancelled task: {e}")

        # 4. Create new cancel event
        cancel_event = asyncio.Event()
        self._session_cancel_events[session_key] = cancel_event

        # 5. Start new processing task
        task = asyncio.create_task(
            self._process_session_task(session_key, msg, cancel_event)
        )
        self._session_tasks[session_key] = task

    async def _process_session_task(
        self, session_key: str, msg: InboundMessage, cancel_event: asyncio.Event
    ) -> None:
        """Process a single session task with cancellation support."""
        try:
            # Check if already cancelled
            if cancel_event.is_set():
                return

            session = await self._get_or_create_session(session_key)

            # Update task progress
            await self._update_task_progress(session_key, TaskProgress.BUILDING_CONTEXT)

            # Process single turn with cancellation support
            response = await self._process_single_turn(
                session, msg, cancel_event=cancel_event
            )

            # Check cancellation again before sending response
            if cancel_event.is_set():
                return

            if response:
                await self.bus.publish_outbound(response)
            elif msg.channel == "cli":
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="",
                        metadata=msg.metadata or {},
                    )
                )

            # Task completed successfully
            await self.persistence.complete_task(session_key)

        except asyncio.CancelledError:
            logger.info(f"Session task cancelled for {session_key}")
            # Don't complete the task - leave it for recovery
            raise
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.persistence.fail_task(session_key)
            try:
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}",
                    )
                )
            except Exception:
                pass
        finally:
            # Clean up state only if we're still the current task
            current_task = self._session_tasks.get(session_key)
            if current_task == asyncio.current_task():
                self._session_tasks.pop(session_key, None)
                # Ensure cancel_event is set before removing to avoid "Task was destroyed" warning
                pending_cancel_event = self._session_cancel_events.pop(
                    session_key, None
                )
                if pending_cancel_event and not pending_cancel_event.is_set():
                    pending_cancel_event.set()

    async def _process_single_turn(
        self,
        session: Session,
        msg: InboundMessage,
        cancel_event: asyncio.Event | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single turn of conversation (without adding the user message).
        Supports cancellation via cancel_event.
        """
        session_key = session.key

        # Check for cancellation at start
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd.startswith("/evolution"):
            await self.persistence.complete_task(session_key)
            return await self._handle_evolution_command(msg)
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated :]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            await self._save_session(session)
            self.sessions.invalidate(session.key)
            await self.persistence.complete_task(session_key)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="New session started."
            )
        if cmd == "/help":
            await self.persistence.complete_task(session_key)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="🐈 nanobot commands:\n/new — Start a new conversation\n/status — Show current context info\n/evolution — Manage self-evolution proposals\n/help — Show available commands",
            )

        if cmd == "/status":
            await self.persistence.complete_task(session_key)
            return await self._handle_status_command(session, msg)

        # Check cancellation after command handling
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

        # Check cancellation before building context
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # Update task progress
        await self._update_task_progress(session_key, TaskProgress.BUILDING_CONTEXT)

        history, initial_messages = await self._prepare_messages_with_preflight(
            session,
            msg,
            cancel_event=cancel_event,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Check cancellation before agent loop
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

        # Update task progress
        await self._update_task_progress(session_key, TaskProgress.AGENT_LOOP)

        final_content, tools_used, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            cancel_event=cancel_event,
            session_key=session_key,
        )

        # Check cancellation after agent loop
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        await self._save_session(session)

        self._trigger_self_evolution(msg, final_content, tools_used, all_msgs)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    # ===== Task persistence and recovery methods =====

    async def _recover_pending_tasks(self) -> None:
        """Recover pending tasks on startup."""
        pending_tasks = await self.persistence.get_pending_tasks()
        if not pending_tasks:
            return

        logger.info("Recovering {} pending tasks", len(pending_tasks))

        for task in pending_tasks:
            asyncio.create_task(self._recover_task(task))

    async def _recover_task(self, task: TaskState) -> None:
        """Recover a single pending task."""
        if not task.last_inbound:
            logger.warning("Task {} has no last_inbound, removing", task.task_id)
            await self.persistence.remove_task(task.session_key)
            return

        # Reconstruct InboundMessage
        last_inbound = task.last_inbound
        msg = InboundMessage(
            channel=last_inbound.get("channel", "cli"),
            sender_id=last_inbound.get("sender_id", "user"),
            chat_id=last_inbound.get("chat_id", "direct"),
            content=last_inbound.get("content", ""),
            media=last_inbound.get("media") or [],
            metadata=last_inbound.get("metadata", {}),
        )

        # Send recovery notice
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="检测到未完成任务，正在恢复...",
                metadata=dict(msg.metadata or {}),
            )
        )

        # Process the task
        session_key = task.session_key
        session = await self._get_or_create_session(session_key)

        # Create cancel event
        cancel_event = asyncio.Event()
        self._session_cancel_events[session_key] = cancel_event

        # Start recovery task
        recovery_task = asyncio.create_task(
            self._process_session_task(session_key, msg, cancel_event)
        )
        self._session_tasks[session_key] = recovery_task

    def _inbound_to_dict(self, msg: InboundMessage) -> dict[str, Any]:
        """Convert InboundMessage to a serializable dict."""
        return {
            "channel": msg.channel,
            "sender_id": msg.sender_id,
            "chat_id": msg.chat_id,
            "content": msg.content,
            "media": msg.media,
            "metadata": msg.metadata,
        }

    async def _update_task_progress(
        self,
        session_key: str,
        progress: TaskProgress,
        messages: list[dict[str, Any]] | None = None,
        current_iteration: int | None = None,
        tools_used: list[str] | None = None,
        current_tool: str | None = None,
        current_tool_args: dict[str, Any] | None = None,
    ) -> None:
        """Update task progress in the store."""
        await self.persistence.update_task(
            session_key,
            progress=progress,
            messages=messages,
            current_iteration=current_iteration,
            tools_used=tools_used,
            current_tool=current_tool,
            current_tool_args=current_tool_args,
        )
