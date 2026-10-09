#!/bin/bash
set -e

# Navigate to project directory
cd "$(dirname "$0")"

echo "Setting up environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment."
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Manufacturing QA Control Plane..."
echo "Access the UI at http://localhost:7865"
echo "Press Ctrl+C to stop."

python app.py
