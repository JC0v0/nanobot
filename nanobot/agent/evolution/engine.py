"""Generate and persist self-evolution proposals."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider


class EvolutionEngine:
    """LLM-assisted proposal engine for skill/tool evolution."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        min_confidence: float = 0.7,
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
        """Generate a structured proposal for skill/tool evolution."""
        error_samples = self._extract_error_samples(messages)

        # Only trigger proposal if there are actual issues (errors or repeated failures)
        has_errors = len(error_samples) > 0
        if not has_errors:
            return None

        prompt = self._build_prompt(
            user_message=user_message,
            final_response=final_response,
            tools_used=tools_used,
            error_samples=error_samples,
        )

        response = await self.provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evolution reviewer for an AI agent. "
                        "Return strict JSON only. Keep changes minimal and reversible."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=1200,
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
                    "risk": self._risk_for_kind(kind),
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
            import re

            return bool(re.match(r"^[a-z0-9][a-z0-9-]{1,62}$", name))
        if kind_lower.startswith("tool_"):
            import re

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
        error_samples: list[str],
    ) -> str:
        return (
            "Analyze this completed task and propose minimal capability evolution.\n"
            "Prefer updating existing skills/tools over creating new ones.\n"
            "IMPORTANT name constraints:\n"
            "  - For skill_*: use lowercase kebab-case (2-63 chars, a-z 0-9 - only)\n"
            "  - For tool_*: use lowercase snake_case (2-64 chars, a-z 0-9 _ only)\n"
            "IMPORTANT content format:\n"
            "  - For tool_create/tool_update: content MUST be valid Python code that can be written to a .py file\n"
            "  - For skill_create/skills_update: content MUST be valid markdown (SKILL.md format)\n"
            "  - Do NOT write explanations or instructions in content, write the actual code/markdown\n"
            "Output JSON with schema:\n"
            "{\n"
            '  "summary": "string",\n'
            '  "confidence": 0.0,\n'
            '  "actions": [{\n'
            '    "kind": "skill_update|skill_create|skill_deprecate|tool_update|tool_create|tool_deprecate",\n'
            '    "name": "string",\n'
            '    "description": "string optional",\n'
            '    "reason": "string explaining why this change is needed",\n'
            '    "content": "full Python code for tools OR markdown for skills (NOT instructions)"\n'
            "  }]\n"
            "}\n\n"
            f"User request:\n{user_message}\n\n"
            f"Final response:\n{final_response}\n\n"
            f"Tools used:\n{', '.join(tools_used) if tools_used else 'none'}\n\n"
            "Tool errors:\n" + ("\n".join(error_samples) if error_samples else "none")
        )

    @staticmethod
    def _extract_error_samples(messages: list[dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.startswith("Error"):
                errors.append(content[:240])
        return errors[-5:]

    @staticmethod
    def _risk_for_kind(kind: str) -> str:
        kind = kind.strip().lower()
        mapping = {
            "skill_update": "low",
            "skill_create": "medium",
            "skill_deprecate": "medium",
            "tool_update": "medium",
            "tool_create": "high",
            "tool_deprecate": "medium",
        }
        return mapping.get(kind, "high")

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
