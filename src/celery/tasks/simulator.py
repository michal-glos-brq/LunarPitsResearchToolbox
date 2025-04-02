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





# @celery_app.task(bind=True)
# def run_remote_simulation(self, instrument_names: List[str], filter_config: Dict, time_range: Tuple[str, str]):
#     instruments = [load_instrument(name) for name in instrument_names]
#     filter_obj = create_filter_from_config(filter_config)
#     kernel_manager = preconfigured_kernel_manager()

#     simulator = RemoteSensingSimulator(instruments, filter_obj, kernel_manager)
#     simulator.start_simulation(
#         start_time=Time(time_range[0]),
#         end_time=Time(time_range[1]),
#         interactive_progress=False,
#         current_task=self
#     )
#     return {"status": "done"}



