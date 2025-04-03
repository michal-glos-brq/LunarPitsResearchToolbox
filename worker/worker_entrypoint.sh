#!/bin/bash

# see .env for the environment variables

# Run the Celery app
echo "🚀 Launching Celery worker with ID: $WORKER_ID"

. venv/bin/activate
celery -A src.celery.app:celery worker --hostname="$WORKER_ID" --loglevel=info --concurrency="$CONCURRENCY"
bash
