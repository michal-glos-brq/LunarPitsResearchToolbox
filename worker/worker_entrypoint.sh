#!/bin/bash

# Ensure ZeroTier daemon is running
service zerotier-one start

# # Start cron to auto-update the code
# service cron start

# Wait briefly for ZeroTier to initialize
sleep 15

# Join the ZeroTier network (only needed once per session)
zerotier-cli join "$NETWORK_ID"

# Run the Celery app
echo "🚀 Launching Celery worker with ID: $WORKER_ID"

. venv/bin/activate
celery -A src.celery.app:celery worker --hostname="$WORKER_ID" --loglevel=info --concurrency=1
bash
