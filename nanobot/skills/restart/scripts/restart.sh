#!/bin/bash
# Restart nanobot Gateway service
# Usage: ./restart.sh [--workdir DIR]

set -e

WORKDIR="${NANOBOT_WORKDIR:-$(dirname "$(dirname "$(dirname "$0")")")}"
PORT="${NANOBOT_PORT:-18790}"
TIMEOUT=10

# Find the gateway process by port
find_process() {
    lsof -i :"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1
}

# Get PID from pid file (if exists)
get_pid_from_file() {
    local pid_file="$WORKDIR/.nanobot.pid"
    if [[ -f "$pid_file" ]]; then
        cat "$pid_file"
    fi
}

# Save PID to file
save_pid() {
    local pid_file="$WORKDIR/.nanobot.pid"
    echo "$1" > "$pid_file"
}

# Kill the old process gracefully
kill_gateway() {
    local pid
    pid=$(find_process)
    
    if [[ -z "$pid" ]]; then
        # Try pid file
        pid=$(get_pid_from_file)
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            echo "⚠️  No running gateway found"
            return 0
        fi
    fi

    echo "📍 Found gateway process: $pid"
    
    # Try graceful shutdown first (SIGTERM)
    if kill -0 "$pid" 2>/dev/null; then
        echo "⏳ Sending SIGTERM (graceful shutdown)..."
        kill -TERM "$pid" 2>/dev/null || true
        
        # Wait for process to exit
        local count=0
        while kill -0 "$pid" 2>/dev/null && ((count < TIMEOUT)); do
            sleep 1
            ((count++))
            echo "  Waiting... ($count/$TIMEOUT)"
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚡ Sending SIGKILL (force kill)..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi
    
    # Clean up pid file
    rm -f "$WORKDIR/.nanobot.pid"
    
    echo "✓ Gateway stopped"
}

# Start the gateway
start_gateway() {
    echo "🚀 Starting nanobot Gateway..."
    
    cd "$WORKDIR"
    
    # Check if virtualenv exists
    if [[ -f "$WORKDIR/.venv/bin/nanobot" ]]; then
        VENV_PY="$WORKDIR/.venv/bin/python3"
    else
        VENV_PY="python3"
    fi
    
    # Start in background and save PID
    nohup "$VENV_PY" -m nanobot gateway --port "$PORT" > "$WORKDIR/nanobot.log" 2>&1 &
    local new_pid=$!
    
    # Save PID
    save_pid "$new_pid"
    
    echo "📍 Started with PID: $new_pid"
    
    # Wait for port to be ready
    local count=0
    while ! lsof -i :"$PORT" -sTCP:LISTEN -q >/dev/null 2>&1 && ((count < TIMEOUT)); do
        sleep 1
        ((count++))
    done
    
    if lsof -i :"$PORT" -sTCP:LISTEN -q >/dev/null 2>&1; then
        echo "✓ Gateway is running on port $PORT"
        return 0
    else
        echo "❌ Gateway failed to start"
        echo "📄 Check log: $WORKDIR/nanobot.log"
        return 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--workdir DIR] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --workdir DIR   Working directory (default: project root)"
            echo "  --port PORT     Gateway port (default: 18790)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main
main() {
    echo "🔄 Restarting nanobot Gateway (port $PORT)..."
    
    # Validate workdir
    if [[ ! -d "$WORKDIR" ]]; then
        echo "❌ Error: Directory not found: $WORKDIR"
        exit 1
    fi
    
    # Check if nanobot is available
    if [[ ! -f "$WORKDIR/.venv/bin/nanobot" ]] && ! command -v nanobot &> /dev/null; then
        echo "❌ Error: nanobot not found in $WORKDIR/.venv/bin/ or PATH"
        exit 1
    fi
    
    # Stop existing gateway
    kill_gateway || true
    
    # Start new gateway
    if start_gateway; then
        echo "✅ Restart complete!"
        exit 0
    else
        echo "❌ Restart failed"
        exit 1
    fi
}

main "$@"