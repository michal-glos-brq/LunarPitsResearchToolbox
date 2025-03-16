# if [[ -z "$WORKER_ID" ]]; then
#     echo "❌ ERROR: WORKER_ID is not set."
#     exit 1
# fi

echo "🚀 Launching Celery worker with ID: $WORKER_ID"
# celery -A celery_app worker --hostname="$WORKER_ID" --loglevel=info --concurrency=1
bash
