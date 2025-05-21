"""
====================================================
SPICE Kernel Management Module – Base Kernel Manager
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

Defines the BaseKernelManager class, which handles loading and orchestration of
both static and dynamic SPICE kernels using mixin-driven configuration.
"""

from abc import ABC
from collections import OrderedDict
import inspect
from typing import List

import spiceypy as spice
from astropy.time import Time


from src.global_config import SPICE_DECIMAL_PRECISION
from src.SPICE.config import (
    SPICE_PRELOAD,
    root_path,
    generic_url,
)
from src.SPICE.kernel_utils.spice_kernels import (
    BaseKernel,
    StaticKernelLoader,
)
from src.SPICE.kernel_utils.kernel_manager_mixins import BaseKernelManagerMixin


class BaseKernelManager(ABC):
    """
    Abstract base class for managing SPICE kernel sets across time.

    Responsibilities:
        - Manages static kernel loading during initialization (if enabled).
        - Supports dynamic kernel loading based on observation time.
        - Delegates dataset-specific setup to mixins via `setup_kernels`.
        - Offers interface for stepping through time.

    Dynamic kernels are expected to implement `reload_kernels(Time)` and maintain
    `min_loaded_time` / `max_loaded_time` properties.
    """

    keys = ["lsk", "sclk", "pck", "fk", "bpck", "ik", "ck", "spk", "dsk"]

    def __init__(
        self,
        min_required_time: Time = None,
        max_required_time: Time = None,
        pre_load_static_kernels: bool = SPICE_PRELOAD,
        **kwargs,
    ):
        """
        Initialize the kernel manager and optionally preload static kernels.

        Parameters:
            min_required_time (Time): Optional lower bound on required time coverage.
            max_required_time (Time): Optional upper bound on required time coverage.
            pre_load_static_kernels (bool): Whether to load static kernels during initialization.
            **kwargs: Forwarded to mixin setup logic.
        """
        # In case we want only partial coverage, have it universally accesible
        self.static_kernel_manager = None
        self.min_required_time = min_required_time
        self.max_required_time = max_required_time

        self.static_kernels: OrderedDict[str, List[BaseKernel]] = OrderedDict(
            [
                (
                    "lsk",
                    [
                        BaseKernel(
                            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
                            root_path("lsk/naif0012.tls"),
                        ),
                    ],
                ),
                ("sclk", []),
                ("pck", [BaseKernel(generic_url("pck/pck00010.tpc"), root_path("pck/pck00010.tpc"))]),
                ("fk", []),
                # Binary pck files ...
                ("bpck", []),
                ("ik", []),
                ("ck", []),
                ("spk", []),
                ("dsk", []),
            ]
        )
        self.dynamic_kernels = []
        self.add_mixin_kernels(min_required_time=min_required_time, max_required_time=max_required_time, **kwargs)
        if pre_load_static_kernels:
            self.load_static_kernels()

    def setup_kernels(self, **kwargs):
        """
        To be implemented by subclasses or mixins to register dynamic kernels.

        This method is not called directly. Instead, `add_mixin_kernels()` invokes it
        on all applicable mixins.
        """
        pass

    @property
    def dsk(self, index=0):
        """
        Get a specific DSK (Digital Shape Kernel) from the static kernel pool.

        Parameters:
            index (int): Index of the desired DSK. Defaults to 0.

        Returns:
            BaseKernel or None: The requested DSK kernel, or None if unavailable.
        """
        if self.static_kernels["dsk"] and index < len(self.static_kernels["dsk"]):
            return self.static_kernels["dsk"][index]
        return None

    def unload_all(self):
        """
        Unload all static and dynamic kernels currently held by the manager.
        """
        for static_kernel in self.static_kernels.values():
            for kernel in static_kernel:
                kernel.unload()
        for dynamic_kernel in self.dynamic_kernels:
            dynamic_kernel.unload()

    def load_static_kernels(self) -> None:
        """
        Load all static kernels into SPICE.
        """
        if self.static_kernel_manager is None:
            self.static_kernel_manager = StaticKernelLoader(self.static_kernels)
        self.static_kernel_manager.load()

    @property
    def min_loaded_time(self) -> Time:
        """
        Get the minimum time currently covered by any dynamic kernel.

        Returns:
            Time or None: Earliest loaded time across all dynamic kernels.
        """
        return (
            max([kernel.min_loaded_time for kernel in self.dynamic_kernels if kernel.min_loaded_time])
            if self.dynamic_kernels
            else None
        )

    @property
    def max_loaded_time(self) -> Time:
        """
        Get the maximum time currently covered by any dynamic kernel.

        Returns:
            Time or None: Latest loaded time across all dynamic kernels.
        """
        return (
            min([kernel.max_loaded_time for kernel in self.dynamic_kernels if kernel.max_loaded_time])
            if self.dynamic_kernels
            else None
        )

    def step(self, time: Time):
        """
        Ensure all dynamic kernel managers have data loaded for the specified time.

        Parameters:
            time (Time): Observation or simulation time to step to.

        Returns:
            bool: True if all managers successfully loaded their data, False otherwise.
        """
        try:
            return all([kernel.reload_kernels(time) for kernel in self.dynamic_kernels])
        except Exception as e:
            print(f"Error while loading dynamic kernels: {e}")
            return False

    def step_et(self, et: float):
        """
        Step using SPICE ephemeris time (ET).

        Converts the ET to astropy Time object and calls `step()`.

        Parameters:
            et (float): SPICE ephemeris time (seconds past J2000).

        Returns:
            bool: True if stepping was successful, False otherwise.
        """
        return self.step(Time(spice.et2utc(et, "ISOC", SPICE_DECIMAL_PRECISION), format="isot", scale="utc"))

    def activate(self, activation_time: Time = None) -> None:
        """
        Activate the kernel manager at a given time.

        Loads static kernels (if not already loaded) and ensures dynamic
        kernels are available for the given time. If no time is provided,
        uses the manager's min_loaded_time.

        Parameters:
            activation_time (Time): Optional activation time. If None, min_loaded_time is used.
        """
        self.load_static_kernels()
        min_time = self.min_loaded_time if activation_time is None else activation_time
        self.step(min_time)

    def add_mixin_kernels(self, **kwargs):
        """
        Traverse the class hierarchy to call `setup_kernels(**kwargs)` on all mixins.

        This enables composable configuration of kernel managers using
        BaseKernelManagerMixin subclasses. Mixins must implement `setup_kernels(self, **kwargs)`.
        """
        for cls in inspect.getmro(self.__class__):
            if not issubclass(cls, BaseKernelManagerMixin):
                continue
            fn = cls.__dict__.get("setup_kernels")
            if not fn:
                continue
            fn(self, **kwargs)
