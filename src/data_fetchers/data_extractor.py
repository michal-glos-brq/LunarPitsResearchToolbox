import logging
from typing import Dict, List

from astropy.time import Time, TimeDelta
import spiceypy as spice


logger = logging.getLogger(__name__)

class SimulationEngine:
    """
    Orchestrates the simulation using KernelManager, DataProductManagers, and IntervalManager.
    Iterates through time, fetches data for each instrument, reprojects,
    and processes points (e.g., using SPICE sincpt or custom logic).
    """
    def __init__(
        self,
        global_start: Time,
        global_end: Time,
        step: TimeDelta,
        kernel_manager: BaseKernelManager,
        data_managers: Dict[str, BaseDataProductManager],  # key: instrument name
    ):
        self.global_start = global_start
        self.global_end = global_end
        self.step = step
        self.interval_manager = IntervalManager(global_start, global_end, step)
        self.kernel_manager = kernel_manager
        self.data_managers = data_managers
        # This will hold final processed simulation results
        self.results: List[SimulationResult] = []

    def initialize(self):
        # Load kernels for global interval.
        self.kernel_manager.load_static_kernels()
        self.kernel_manager.load_dynamic_kernels(self.global_start, self.global_end)
        # Load data for each instrument.
        for dm in self.data_managers.values():
            dm.load_data(self.global_start, self.global_end)
        # Setup simulation time in the interval manager.
        self.current_time = self.global_start

    def process_data_point(self, data_point: Dict, et: float, instrument: str):
        """
        Process a single data point for a given instrument.
        You can add reprojection here using SPICE (e.g. sincpt),
        additional spatial filtering, etc.
        """
        # Example: get spacecraft state at time 'et'
        sc_state = self.kernel_manager.get_spacecraft_state(et)
        # Example: Use a projection helper:
        projected = self.project_data_point(data_point, sc_state)
        # Apply filters, validations, etc.
        if self.validate_projected(projected):
            self.results.append(SimulationResult(instrument, et, projected))
        else:
            logger.debug(f"[{instrument}] Data point at ET {et} filtered out.")

    def project_data_point(self, data_point: Dict, spacecraft_state: Dict) -> Dict:
        """
        Reproject raw data point using SPICE.
        This could call spice.sincpt() or a custom reprojection method.
        Return the projected point as a dictionary.
        """
        # Placeholder for your actual projection logic.
        # For example, use the data point's range and the instrument's boresight vector.
        projected = data_point.copy()  # Dummy: simply pass it through.
        # Real code will compute intersect, incidence angles, etc.
        return projected

    def validate_projected(self, projected: Dict) -> bool:
        """
        Return True if the projected point meets your spatial criteria,
        e.g., if it lies within a 5 km circle of a target.
        """
        # Implement your spatial criteria.
        return True  # Placeholder

    def run(self):
        self.initialize()
        et_step = self.step.sec
        # Main simulation loop
        while self.current_time < self.global_end:
            current_et = spice.str2et(self.current_time.utc.iso)
            # Step kernel manager to update SPICE state
            self.kernel_manager.step(self.current_time)
            # For each instrument, fetch and process data points in this time interval.
            for instrument, dm in self.data_managers.items():
                data_points = dm.get_data_in_interval(current_et, current_et + et_step)
                for dp in data_points:
                    self.process_data_point(dp, current_et, instrument)
            # Advance simulation time
            self.current_time += self.step
        self.cleanup()

    def cleanup(self):
        # Cleanup all resources
        for dm in self.data_managers.values():
            dm.cleanup()
        self.kernel_manager.unload_all()