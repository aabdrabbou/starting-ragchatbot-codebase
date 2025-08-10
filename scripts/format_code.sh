#!/bin/bash

# Code Formatting Script for RAG Chatbot
# This script automatically formats all Python code using Black and Ruff

set -e

echo "🎨 Formatting code..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Format with Black
echo "📋 Formatting with Black..."
uv run black backend/ main.py

# Format with Ruff
echo "🔧 Formatting with Ruff..."
uv run ruff format backend/ main.py

# Auto-fix Ruff issues
echo "🔨 Auto-fixing Ruff issues..."
uv run ruff check --fix backend/ main.py

echo "✅ Code formatting complete!"