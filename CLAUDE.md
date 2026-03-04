# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanobot** is an AI agent framework with LLM provider abstractions, multi-channel support (Telegram, Discord, etc.), and an extensible skill system.

## Architecture

### Core Modules

- **`nanobot/providers/`** - LLM provider implementations
  - `base.py` - Abstract `LLMProvider` base class with `LLMResponse` and `ToolCallRequest`
  - `volcengine_provider.py` - **VolcEngine (火山引擎) Responses API implementation** - primary focus of recent work
  - `custom_provider.py` - Generic OpenAI-compatible endpoint provider
  - `registry.py` - Provider specs with `is_direct` flag for bypassing LiteLLM
  - `litellm_provider.py` - LiteLLM-based provider

- **`nanobot/session/`** - Session persistence
  - `store.py` - **SQLite-based async session storage** with WAL mode and memory cache

- **`nanobot/agent/`** - Agent loop and task management
  - `loop.py` - Main agent processing loop with `cancel_event` support
  - `subagent.py` - Sub-agent spawning
  - `memory.py` - Conversation memory management
  - `task_store.py` - Task persistence for crash recovery

- **`nanobot/channels/`** - Channel integrations (Telegram, Discord, etc.)

- **`nanobot/skills/`** - Built-in skills (GitHub, weather, tmux, etc.)
  - Each skill has a `SKILL.md` file with YAML frontmatter

### Recent Major Work: VolcEngine Responses API Provider

The `VolcEngineProvider` (`volcengine_provider.py`) was recently implemented to use **VolcEngine's Responses API** instead of the standard Chat API. This is a sophisticated implementation with:

**Content Type Conversions (OpenAI → VolcEngine):**
| OpenAI Format | VolcEngine Format | Parameters |
|--------------|-------------------|------------|
| `text` | `input_text` | `text` |
| `image_url` | `input_image` | `image_url`, `detail` |
| `video_url` | `input_video` | `video_url`, `fps` |
| `file` | `input_file` | `file_url`/`file_data`, `filename` |
| `tool` messages | `function_call_output` | `call_id`, `output` |
| `tool_calls` | `function_call` | `call_id`, `name`, `arguments` |

**Response Parsing:**
- `output_text` → `content`
- `function_call` → `tool_calls`
- `reasoning.summary` → `reasoning_content`
- `input_tokens`/`output_tokens` → `usage`

**Key Implementation Details:**
1. Uses `AsyncOpenAI` client with custom `base_url`
2. Full `cancel_event` support for request cancellation
3. Properly handles `max_output_tokens` instead of `max_tokens`
4. Converts `tool_choice="auto"` for function calling

### SQLite Async Session Persistence

The session storage uses **SQLite-based async storage** with the `PersistenceManager` class.

**Database Schema:**
| Table | Purpose |
|-------|---------|
| `sessions` | Session metadata (key, created_at, updated_at, last_consolidated, metadata) |
| `messages` | Individual messages with structured core fields + JSON for extensions |

**Key Implementation Details:**
1. Uses `aiosqlite` library for true async operations
2. WAL (Write-Ahead Logging) mode enabled for better concurrency
3. Memory cache `dict[str, Session]` for fast access
4. Full transaction support for atomicity

**Message Mapping:**
- Core fields: `role`, `content`, `timestamp`, `tool_call_id`, `name` → structured columns
- `tool_calls` → JSON column
- Extra fields → `extra` JSON column

### Design Patterns

**Provider Pattern:**
All providers inherit from `LLMProvider` and implement:
- `chat()` - Main method with `cancel_event` support
- `get_default_model()` - Returns default model name

**Registry Pattern:**
Providers are registered in `registry.py` with `ProviderSpec`:
- `is_direct=True` - Bypasses LiteLLM entirely
- `is_gateway=True` - Can route any model
- Keywords for model name matching

**Cancel Event Pattern:**
All providers support `cancel_event` (asyncio.Event) for request cancellation:
```python
if cancel_event and cancel_event.is_set():
    raise asyncio.CancelledError()
```

## Git Commit Conventions

Recent commits follow this pattern:
- `feat(provider):` - New features (VolcEngine Responses API)
- `fix(provider):` - Bug fixes (content type conversions)
- `chore:` - Maintenance tasks

## Important Implementation Notes

1. **VolcEngine uses Responses API, not Chat API** - This is a newer API format that differs from standard OpenAI chat completions.

2. **Content type conversion is critical** - VolcEngine strictly validates `input.content.type` and rejects unknown types like `video_url` (must be `input_video`).

3. **Tool calling uses different format** - Responses API uses `function_call` and `function_call_output` types, not `tool_calls`.

4. **Response structure differs** - Output is in `response.output` list with items having `type` fields like `output_text`, not direct `choices[0].message`.

5. **Always use `input_` prefix types** - VolcEngine Responses API expects `input_text`, `input_image`, `input_video`, `input_file` for input content.

6. **Database location** - SQLite database is at `workspace/sessions/sessions.db`.
