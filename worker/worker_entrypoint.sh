#!/bin/bash

# Ensure ZeroTier daemon is running
service zerotier-one start

# Start cron to auto-update the code
service cron start

# Wait briefly for ZeroTier to initialize
sleep 5

# Join the ZeroTier network (only needed once per session)
zerotier-cli join "$NETWORK_ID"

# Run the Celery app
bash /app/run_celery.sh
