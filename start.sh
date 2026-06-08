#!/bin/bash
set -e
echo "Starting Anthology API..."

echo "Running database migrations..."
PYTHONPATH=. alembic upgrade head

echo "Starting API server..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
