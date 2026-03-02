"""VolcEngine (火山引擎) provider — direct OpenAI-compatible API, bypasses LiteLLM."""

from __future__ import annotations

import asyncio
from typing import Any

import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class VolcEngineProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "https://ark.cn-beijing.volces.com/api/v3", default_model: str = "doubao-pro"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   cancel_event: Any | None = None) -> LLMResponse:
        # Convert messages to Responses API input format
        input_data = self._convert_messages_to_input(messages)

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "input": input_data,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        if tools:
            # Convert OpenAI format tools to Responses API format
            responses_tools = self._convert_tools(tools)
            kwargs["tools"] = responses_tools
            kwargs["tool_choice"] = "auto"

        try:
            # Check for cancellation before starting
            if cancel_event and hasattr(cancel_event, "is_set") and cancel_event.is_set():
                raise asyncio.CancelledError()

            # Create the responses task
            completion_task = asyncio.create_task(self._client.responses.create(**kwargs))

            if cancel_event and hasattr(cancel_event, "wait"):
                # Wait for either completion or cancellation
                done, pending = await asyncio.wait(
                    [completion_task, asyncio.create_task(cancel_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                # Clean up pending tasks
                for task in pending:
                    task.cancel()

                if hasattr(cancel_event, "is_set") and cancel_event.is_set():
                    completion_task.cancel()
                    # Try to wait for cancellation
                    try:
                        await completion_task
                    except asyncio.CancelledError:
                        pass
                    raise asyncio.CancelledError()

                # Get response from the completed task
                if completion_task in done:
                    response = completion_task.result()
                else:
                    # Should not reach here if cancel_event check works correctly
                    response = await completion_task
            else:
                response = await completion_task

            return self._parse_response(response)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _convert_messages_to_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert standard chat messages to Responses API input format."""
        sanitized = self._sanitize_empty_content(messages)
        result = []
        for msg in sanitized:
            # Handle tool responses first
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id and msg.get("role") == "tool":
                content = msg.get("content")
                result.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": content if isinstance(content, str) else str(content),
                })
                continue

            # Handle regular messages
            input_item = {
                "type": "message",
                "role": msg.get("role", "user"),
            }
            content = msg.get("content")
            if content is not None:
                input_item["content"] = self._convert_content(content)

            # Handle tool calls in assistant messages
            tool_calls = msg.get("tool_calls")
            if tool_calls and msg.get("role") == "assistant":
                # Add function_call items for each tool call
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        result.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        })
                # Also add the message if it has content
                if input_item.get("content"):
                    result.append(input_item)
            else:
                result.append(input_item)
        return result

    def _convert_content(self, content: Any) -> Any:
        """Convert OpenAI format content to Responses API format."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            result = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        # Convert to input_text
                        result.append({
                            "type": "input_text",
                            "text": item.get("text", ""),
                        })
                    elif item_type == "image_url":
                        # Convert to input_image
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                            result.append({
                                "type": "input_image",
                                "image_url": url,
                            })
                        else:
                            result.append({
                                "type": "input_image",
                                "image_url": image_url,
                            })
                    elif item_type in ("input_text", "input_image", "input_video", "input_file"):
                        # Already in Responses API format, pass through
                        result.append(item)
                    else:
                        # Keep unknown types as-is
                        result.append(item)
                else:
                    result.append(item)
            return result

        return content

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI format tools to Responses API format."""
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                result.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                    "strict": True,
                })
        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Responses API response to LLMResponse."""
        content = None
        tool_calls = []
        finish_reason = "stop"
        usage = {}
        reasoning_content = None

        # Extract content from output
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type"):
                    if item.type == "message":
                        # Extract content from message
                        if hasattr(item, "content"):
                            # Content could be a list or a string
                            if isinstance(item.content, list):
                                text_parts = []
                                for part in item.content:
                                    if hasattr(part, "type") and part.type == "output_text":
                                        text_parts.append(getattr(part, "text", ""))
                                if text_parts:
                                    content = "\n".join(text_parts)
                            else:
                                content = item.content

                        # Extract tool calls
                        if hasattr(item, "content") and isinstance(item.content, list):
                            for part in item.content:
                                if hasattr(part, "type") and part.type == "function_call":
                                    tool_calls.append(ToolCallRequest(
                                        id=getattr(part, "call_id", ""),
                                        name=getattr(part, "name", ""),
                                        arguments=json_repair.loads(getattr(part, "arguments", "{}"))
                                        if isinstance(getattr(part, "arguments", ""), str)
                                        else getattr(part, "arguments", {})
                                    ))

                    elif item.type == "function_call":
                        tool_calls.append(ToolCallRequest(
                            id=getattr(item, "call_id", ""),
                            name=getattr(item, "name", ""),
                            arguments=json_repair.loads(getattr(item, "arguments", "{}"))
                            if isinstance(getattr(item, "arguments", ""), str)
                            else getattr(item, "arguments", {})
                        ))

        # Extract reasoning content
        if hasattr(response, "reasoning"):
            reasoning = response.reasoning
            if hasattr(reasoning, "summary") and reasoning.summary:
                summary_parts = []
                for part in reasoning.summary:
                    if hasattr(part, "type") and part.type == "summary_text":
                        summary_parts.append(getattr(part, "text", ""))
                if summary_parts:
                    reasoning_content = "\n".join(summary_parts)

        # Extract usage
        if hasattr(response, "usage"):
            u = response.usage
            usage = {
                "prompt_tokens": getattr(u, "input_tokens", 0),
                "completion_tokens": getattr(u, "output_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            }

        # Extract finish reason
        if hasattr(response, "status"):
            status = response.status
            if status == "completed":
                finish_reason = "stop"
            elif status == "incomplete":
                finish_reason = "length"
            elif status == "failed":
                finish_reason = "error"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def get_default_model(self) -> str:
        return self.default_model
