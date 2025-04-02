"""
Here we are defining the celery app, completely

Celery tasks will be defined inthe tasks subfolder


There are broad tasks:
 - Spacecraft and instrument simulation (Simulate instrument of spacecraft FOVs and obtain time intervals when
                the instrument was pointing on our points of interest)
 - Data fetcher - Based on time intervals obtained from the spacecraft and instrument simulation,
                fetch the dataset and filter only the data within this specific interval
 - Data Processor - Process the data in sensible temporal and spatial data
 - Analysis will be made locally
"""
import os
from celery import Celery

master_ip = os.getenv("MASTER_IP____", "localhost")  # fallback for local testing

celery = Celery(
    "worker",
    broker=f"redis://{master_ip}:6379/0",
    backend=f"redis://{master_ip}:6379/0"
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],  # Accept only JSON tasks
    result_serializer="json",  # Store results as JSON
    timezone="Europe/Prague",
    enable_utc=True,
)
