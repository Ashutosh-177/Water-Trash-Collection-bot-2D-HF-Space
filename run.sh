#!/bin/bash
source /home/ashu_17/Documents/RL\ HACKATHON/.venv/bin/activate

# Use the workspace Python environment
export ENABLE_WEB_INTERFACE=true
export GEMINI_API_KEY="AIzaSyB6T98qoTPyODAolWfKq4npvN1KGyn0IFM"
export TASK_LEVEL="easy"

echo "Starting Uvicorn server..."
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server to start..."
sleep 3

echo "Starting inference agent..."
python3 inference.py

echo "Done! Shutting down server..."
kill $SERVER_PID
