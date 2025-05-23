"""
============================================================
Pipeline description
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

from celery import Celery

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
    # Do not duplicate tasks
    broker_transport_options={"visibility_timeout": 172800 * 2},
    result_backend_transport_options={"visibility_timeout": 172800 * 2},
)


run_remote_sensing_simulation_task = app.task(
    name="src.pipeline.tasks.simulator.run_remote_sensing_simulation",
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)(run_remote_sensing_simulation)


run_data_extraction_task = app.task(
    name="src.pipeline.tasks.extractor.run_data_extraction",
    bind=True,
    autoretry_for=(),
    retry_kwargs={"max_retries": 0},
)(run_data_extraction)
