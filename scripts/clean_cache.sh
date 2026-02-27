#!/usr/bin/env bash

set -e

echo "[CLEAN] Cleaning up local caches and temporary files..."

# Find and remove __pycache__ directories
echo "> Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# Find and remove compiled python files
echo "> Removing .pyc files..."
find . -type f -name "*.pyc" -delete

# Remove pytest cache
if [ -d ".pytest_cache" ]; then
    echo "> Removing .pytest_cache..."
    rm -rf .pytest_cache
fi

# Remove coverage data
if [ -f ".coverage" ]; then
    echo "> Removing .coverage..."
    rm -f .coverage
fi

# Remove mypy cache if present
if [ -d ".mypy_cache" ]; then
    echo "> Removing .mypy_cache..."
    rm -rf .mypy_cache
fi

# Remove ruff cache if present
if [ -d ".ruff_cache" ]; then
    echo "> Removing .ruff_cache..."
    rm -rf .ruff_cache
fi

echo "[OK] Cache cleanup complete!"
