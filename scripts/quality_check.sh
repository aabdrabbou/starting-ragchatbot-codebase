#!/bin/bash

# Quality Check Script for RAG Chatbot
# This script runs code quality checks including formatting and linting

set -e

echo "🔍 Running code quality checks..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Run Black formatter check
echo "📋 Checking code formatting with Black..."
uv run black --check --diff backend/ main.py

# Run Ruff linter
echo "🔧 Running Ruff linter..."
uv run ruff check backend/ main.py

# Run Ruff formatter check
echo "📝 Checking code format with Ruff..."
uv run ruff format --check backend/ main.py

echo "✅ All quality checks passed!"