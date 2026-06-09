#!/bin/bash
set -e
echo "Starting Anthology API..."
echo "PORT: ${PORT:-8000}"

echo "Running database migrations..."
PYTHONPATH=. alembic upgrade head

echo "Starting API server on port ${PORT:-8000}..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
