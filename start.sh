#!/bin/bash
echo "Starting Anthology API..."

# Run alembic migrations
PYTHONPATH=. alembic upgrade head

# Start the API
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
