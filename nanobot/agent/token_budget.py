"""Token counting utilities for context budget checks."""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    import tiktoken
except ImportError:
    tiktoken = None


class TokenBudgetEstimator:
    """Estimate token usage for OpenAI-style chat messages."""

    def __init__(self) -> None:
        self._enc_cache: dict[str, Any] = {}

    def count_messages(self, messages: list[dict[str, Any]], model: str) -> int:
        """Estimate token usage of a message list."""
        enc = self._resolve_encoding(model)
        if enc is None:
            return self._fallback_count(messages)

        total = 0
        for msg in messages:
            total += 4
            for value in msg.values():
                total += self._count_value_tokens(value, enc)
        total += 2
        return total

    def _resolve_encoding(self, model: str) -> Any | None:
        """Resolve and cache encoding for a model string."""
        if tiktoken is None:
            return None

        if model in self._enc_cache:
            return self._enc_cache[model]

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.debug("Failed to load tiktoken encoding for model {}", model)
                enc = None

        self._enc_cache[model] = enc
        return enc

    def _count_value_tokens(self, value: Any, enc: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(enc.encode(value))
        if isinstance(value, (int, float, bool)):
            return len(enc.encode(str(value)))
        if isinstance(value, list):
            return sum(self._count_value_tokens(item, enc) for item in value)
        if isinstance(value, dict):
            total = 0
            for k, v in value.items():
                total += self._count_value_tokens(k, enc)
                total += self._count_value_tokens(v, enc)
            return total
        return len(enc.encode(str(value)))

    def _fallback_count(self, messages: list[dict[str, Any]]) -> int:
        """Fallback approximation when tiktoken is unavailable."""
        chars = 0
        for msg in messages:
            chars += len(str(msg))
        return max(1, chars // 4)
