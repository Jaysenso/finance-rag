#!/usr/bin/env bash
set -e

# Kill both servers on Ctrl+C
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting backend..."
uv run uvicorn api.main:app --reload &
BACKEND_PID=$!

echo "Starting frontend..."
cd "$ROOT/frontend" && npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend  → http://localhost:8000"
echo "Frontend → http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both."

wait
