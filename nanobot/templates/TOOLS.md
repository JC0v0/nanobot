# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## cron — Scheduled Reminders

- Please refer to cron skill for usage.

## skill_manager — Workspace Skill Evolution

- Scope is restricted to `workspace/skills/`.
- Supported actions: `list`, `read`, `create`, `update`, `deprecate`.
- Do not edit builtin skills in `nanobot/skills/`.
- Prefer `update` over creating duplicates.
- Use `deprecate` instead of deleting skills to keep rollback options.

## tool_manager — Workspace Tool Evolution

- Scope is restricted to `workspace/tools/`.
- Supported actions: `list`, `read`, `create`, `update`, `deprecate`, `reload`.
- New tools must subclass `Tool` and expose valid name/description/parameters/execute.
- Use `deprecate` to disable a tool file (renamed to `.disabled.py`) instead of deleting it.

### MCP Server Management (mcp_* actions)

- MCP servers are configured in `workspace/mcp_servers.yaml` (YAML format).
- Supported actions:
  - `mcp_list`: List configured MCP servers
  - `mcp_add`: Add new MCP server (requires name and YAML config)
  - `mcp_remove`: Remove MCP server
  - `mcp_reload`: Reload MCP connections without restarting
- Example YAML config:
  ```yaml
  my_server:
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - "/path/to/directory"
    tool_timeout: 30
  ```
