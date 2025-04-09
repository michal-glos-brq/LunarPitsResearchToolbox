"""
Here we are defining the celery app, completely
"""
from requests.exceptions import ConnectionError

from celery import Celery

from src.pipeline.config import REDIS_CONNECTION_STRING
from src.pipeline.tasks.simulator import run_remote_sensing_simulation


app = Celery("worker", broker=REDIS_CONNECTION_STRING, backend=REDIS_CONNECTION_STRING)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],  # Accept only JSON tasks.
    result_serializer="json",  # Store results as JSON.
    timezone="Europe/Prague",
    enable_utc=True,
    # Additional robust options (optional):
    worker_max_tasks_per_child=8,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_default_queue="default",
    task_reject_on_worker_lost=False,
    task_track_started=True,
)

run_remote_sensing_simulation_task = app.task(
    name="src.pipeline.tasks.simulator.run_remote_sensing_simulation",
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={'max_retries': 5}
)(run_remote_sensing_simulation)
