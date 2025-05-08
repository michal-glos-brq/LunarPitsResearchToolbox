"""
Here we are defining the celery app, completely
"""
from celery import Celery
from celery.signals import worker_process_init

from src.pipeline.config import REDIS_CONNECTION_STRING
from src.pipeline.tasks.simulator import run_remote_sensing_simulation
from src.pipeline.tasks.extractor import run_data_extraction

app = Celery("worker", broker=REDIS_CONNECTION_STRING, backend=REDIS_CONNECTION_STRING)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Prague",
    enable_utc=True,

    # *** disable Celery’s logging hijack ***
    worker_hijack_root_logger=False,
    worker_redirect_stdouts=False,

    # your other robustness flags…
    worker_max_tasks_per_child=4,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_default_queue="default",
    task_reject_on_worker_lost=False,
    task_track_started=True,
)


# 2) On each worker process start, install your handler once:
@worker_process_init.connect
def _init_worker_logging(**kwargs):
    from src.global_config import setup_logging
    setup_logging()

run_remote_sensing_simulation_task = app.task(
    name="src.pipeline.tasks.simulator.run_remote_sensing_simulation",
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={'max_retries': 5}
)(run_remote_sensing_simulation)


run_data_extraction_task = app.task(
    name="src.pipeline.tasks.extractor.run_data_extraction",
    bind=True,
)(run_data_extraction)

