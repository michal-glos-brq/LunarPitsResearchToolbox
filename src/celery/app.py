"""
Here we are defining the celery app, completely
"""

from celery import Celery

from src.celery.config import REDIS_CONNECTION_STRING

app = Celery("worker", broker=REDIS_CONNECTION_STRING, backend=REDIS_CONNECTION_STRING)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],  # Accept only JSON tasks.
    result_serializer="json",  # Store results as JSON.
    timezone="Europe/Prague",
    enable_utc=True,
    # Additional robust options (optional):
    worker_max_tasks_per_child=100,
    task_acks_late=True,
)
