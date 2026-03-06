# nanobot

A lightweight AI agent framework with multi-channel support (Telegram, Discord, Slack, Feishu).

## Features

- **Multi-channel**: Connect via Telegram, Discord, Slack, Feishu, or CLI
- **Extensible skills**: GitHub, weather, web search, file operations, and more
- **Memory**: SQLite-based session persistence with optional graph memory
- **MCP support**: Connect to Model Context Protocol servers
- **Multiple LLM providers**: OpenAI, Anthropic, VolcEngine, LiteLLM, and more

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Interactive chat
nanobot agent

# Send a single message
nanobot agent -m "Hello"

# Start gateway (for Telegram/Discord channels)
nanobot run
```

## Configuration

See `CLAUDE.md` for detailed configuration options.