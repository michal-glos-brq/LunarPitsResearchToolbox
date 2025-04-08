#!/bin/bash

# see .env for the environment variables

WORKER_ID=${HOSTNAME}
SUPRESS_TQDM=1

LOG_FILE="${UTILITY_VOLUME}/logs/worker-${WORKER_ID}.log"

# Run the Celery app
echo "🚀 Launching Celery worker with ID: $WORKER_ID"

. venv/bin/activate

celery -A src.pipeline.app:app worker \
    --hostname="$WORKER_ID" \
    --loglevel=info \
    --concurrency="$CONCURRENCY" \
    -E   \
    2>&1 | tee -a "$LOG_FILE"

