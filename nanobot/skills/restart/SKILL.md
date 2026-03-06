---
name: restart
description: Restart nanobot Gateway service safely
metadata: {"nanobot":{"emoji":"🔄","os":["darwin","linux"],"requires":{"bins":["lsof","bash"]}}}
---

# restart Skill

Use this skill when you need to restart the nanobot Gateway service.

## When to Use This Skill

- User asks to "restart nanobot" or "重启 nanobot"
- User says "reboot the project" or "重启项目"
- Gateway is unresponsive and needs a restart
- After certain configuration changes that require restart
- User explicitly requests a restart

## IMPORTANT: Do NOT Use These Commands

- ❌ `pkill -f nanobot` — Too aggressive, may kill unintended processes
- ❌ `killall python` — Will kill all Python processes, including unrelated ones
- ❌ Manual process hunting — Error-prone and unsafe

## Correct Restart Method

The restart script is located in this skill's scripts directory. Use the absolute path to run it:

```bash
# Find the nanobot installation path first
NANOBOT_PATH=$(python3 -c "import nanobot; print(nanobot.__file__)" 2>/dev/null | xargs dirname | xargs dirname)
SCRIPT_DIR="${NANOBOT_PATH}/skills/restart/scripts/restart.sh"

# Run the restart script
bash "$SCRIPT_DIR"

# Or with explicit workdir (if you know the project path)
bash "$SCRIPT_DIR" --workdir ~/Documents/nanobot

# With custom port
bash "$SCRIPT_DIR" --port 18790
```

## Alternative: Direct Path (if you know the project location)

If you know the nanobot project is at `~/Documents/nanobot`:

```bash
~/Documents/nanobot/nanobot/skills/restart/scripts/restart.sh
```

## What the Script Does

1. **Locates** the running gateway process (by port 18790)
2. **Sends SIGTERM** for graceful shutdown (waits up to 10 seconds)
3. **Falls back to SIGKILL** if process doesn't exit gracefully
4. **Starts** a new gateway process
5. **Verifies** the new process is running (port is listening)

## Verification

After restart, you can verify the gateway is running:

```bash
# Check if port is listening
lsof -i :18790 -sTCP:LISTEN

# Or check process
ps aux | grep "nanobot gateway" | grep -v grep
```

## Success Response

When restart is complete, inform the user:

- Gateway has been restarted successfully
- The new PID (if they want to check)
- Any relevant log information if there were issues

## Error Handling

If restart fails:

1. Check the log file: `~/Documents/nanobot/nanobot.log`
2. Try to identify the error from the log
3. Inform the user about the failure and suggest checking the log
4. Do NOT repeatedly attempt restart — report the issue instead