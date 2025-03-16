from dataclasses import dataclass, asdict
from src.celery.app import celery

@dataclass
class SimulatorTask:

    def to_dict(self):
        return asdict(self)  # Convert dataclass to a dictionary

@celery.task(bind=True)
def simulator_task(self, task_data: dict) -> None:
    task = SimulatorTask(**task_data)  # Recreate the dataclass from dict
    print(f"Running SimulatorTask with param1={task.param1}, param2={task.param2}")
