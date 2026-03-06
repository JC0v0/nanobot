"""Generate and persist self-evolution proposals - Learn from tasks."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider


class EvolutionEngine:
    """Learn from task execution and create skill summaries."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        min_confidence: float = 0.6,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.min_confidence = min_confidence
        self.store_dir = workspace / "evolution"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.store_dir / "proposals.jsonl"

    async def propose(
        self,
        user_message: str,
        final_response: str,
        tools_used: list[str],
        messages: list[dict[str, Any]],
        session_key: str,
    ) -> dict[str, Any] | None:
        """Analyze completed task and generate learning skill if valuable."""
        tool_results = self._extract_tool_results(messages)
        has_content = len(tool_results) > 0 or user_message.strip()

        if not has_content:
            return None

        prompt = self._build_prompt(
            user_message=user_message,
            final_response=final_response,
            tools_used=tools_used,
            tool_results=tool_results,
        )

        response = await self.provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a learning assistant for an AI agent. "
                        "Analyze the task execution and create skills that capture "
                        "valuable learnings, patterns, and solutions. "
                        "Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.2,
            max_tokens=1500,
        )

        payload = self._extract_json(response.content or "")
        if not payload:
            return None

        confidence = float(payload.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return None

        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            return None

        normalized_actions = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind", "")).strip()
            name = str(action.get("name", "")).strip()
            if not kind or not name:
                continue
            if not self._is_valid_name(kind, name):
                continue
            normalized_actions.append(
                {
                    "kind": kind,
                    "name": name,
                    "reason": str(action.get("reason", "")).strip(),
                    "content": str(action.get("content", "")).strip(),
                    "description": str(action.get("description", "")).strip(),
                    "risk": "low",
                }
            )

        if not normalized_actions:
            return None

        proposal = {
            "id": f"ev-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "created_at": datetime.now().isoformat(),
            "session_key": session_key,
            "summary": str(payload.get("summary", "")).strip(),
            "confidence": confidence,
            "status": "pending",
            "actions": normalized_actions,
        }
        self.append_log({"type": "proposal", **proposal})
        return proposal

    @staticmethod
    def _is_valid_name(kind: str, name: str) -> bool:
        """Validate name against kind constraints."""
        kind_lower = kind.strip().lower()
        name = name.strip()
        if not 2 <= len(name) <= 63:
            return False
        if kind_lower.startswith("skill_"):
            return bool(re.match(r"^[a-z0-9][a-z0-9-]{1,62}$", name))
        if kind_lower.startswith("tool_"):
            return bool(re.match(r"^[a-z0-9][a-z0-9_]{1,63}$", name))
        return False

    def append_log(self, record: dict[str, Any]) -> None:
        """Append a jsonl record to evolution log."""
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_prompt(
        self,
        user_message: str,
        final_response: str,
        tools_used: list[str],
        tool_results: list[dict[str, str]],
    ) -> str:
        tool_results_text = ""
        if tool_results:
            lines = []
            for r in tool_results:
                tool = r.get("tool", "unknown")
                content = r.get("content", "")[:300]
                status = r.get("status", "unknown")
                lines.append(f"- {tool} [{status}]: {content}")
            tool_results_text = "\n".join(lines)

        return f"""Analyze this completed task and extract valuable learnings.

Create a skill that captures:
1. **Problems encountered** - What issues or errors happened during execution?
2. **Solutions found** - How were problems solved? What worked?
3. **Patterns & insights** - Reusable patterns for similar future tasks
4. **Gotchas & pitfalls** - Things to avoid or be careful about

IMPORTANT:
- Name must be lowercase kebab-case (e.g., `web-scraping-tips`, `api-error-handling`)
- Content must be a complete SKILL.md in markdown format with YAML frontmatter
- Focus on actionable learnings, not generic advice

Output JSON:
{{
  "summary": "Brief description of what was learned",
  "confidence": 0.0-1.0,
  "actions": [{{
    "kind": "skill_create",
    "name": "learning-topic-name",
    "description": "1-2 sentence description",
    "reason": "Why this learning is valuable",
    "content": "Full SKILL.md markdown content"
  }}]
}}

User request:
{user_message}

Tools used:
{", ".join(tools_used) if tools_used else "none"}

Tool execution results:
{tool_results_text if tool_results_text else "none"}

Final response summary:
{final_response[:500] if final_response else "none"}"""

    @staticmethod
    def _extract_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        results = []
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("name", "unknown")
            content = msg.get("content", "")

            is_error = isinstance(content, str) and content.startswith("Error")
            status = "error" if is_error else "success"

            results.append(
                {
                    "tool": tool_name,
                    "content": content,
                    "status": status,
                }
            )

        return results[-10:]

    @staticmethod
    def _risk_for_kind(kind: str) -> str:
        return "low"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None

        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fenced:
            text = fenced.group(1).strip()

        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            logger.debug("Failed to parse evolution JSON payload")
            return None
