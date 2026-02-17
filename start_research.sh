#!/bin/bash
# Research Engine Starter Script

cd "$(dirname "$0")"

# Remove stale lock file
LOCK_FILE="state/locks/research_engine.lock"
if [ -f "$LOCK_FILE" ]; then
    PID=$(jq -r '.pid' "$LOCK_FILE" 2>/dev/null)
    if [ -n "$PID" ] && ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Removing stale lock file (PID $PID not running)"
        rm -f "$LOCK_FILE"
    fi
fi

# Create logs directory
mkdir -p logs

# Start research engine
echo "Starting research engine..."
nohup python -m research > logs/research_engine.log 2>&1 &
RESEARCH_PID=$!

sleep 2

# Check if process is running
if ps -p $RESEARCH_PID > /dev/null 2>&1; then
    echo "✓ Research engine started successfully (PID: $RESEARCH_PID)"
    echo "  Dashboard: http://localhost:9998"
    echo "  Log file: logs/research_engine.log"
else
    echo "✗ Research engine failed to start"
    echo "  Check logs/research_engine.log for details"
    exit 1
fi
