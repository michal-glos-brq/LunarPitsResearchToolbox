"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

This module defines high-level kernel managers for different lunar datasets
(GRAIL, LRO, etc.), built via mixins on top of BaseKernelManager.

Each manager encapsulates static and dynamic kernel orchestration, time-based
loading, and dataset-specific configuration.
"""

from typing import Literal

from astropy.time import Time


from src.SPICE.config import (
    SPICE_PERSIST,
    SPICE_PRELOAD,
)
from src.global_config import LUNAR_FRAME
from src.SPICE.kernel_utils.kernel_manager_mixins import (
    LunarKernelManagerMixin,
    GRAILKernelManagerMixin,
    LROKernelManagerMixin,
)
from src.SPICE.kernel_utils.base_kernel_manager import BaseKernelManager


class LeapsecondKernelManager(BaseKernelManager):
    """
    Lightweight kernel manager used solely for leapsecond conversions.

    Loads only the basic static kernels required to translate between
    UTC and ephemeris time (ET). No dynamic datasets or sensors involved.
    """

    def __init__(self, min_required_time: Time = None, max_required_time: Time = None):
        super().__init__(min_required_time=min_required_time, max_required_time=max_required_time)
        self.load_static_kernels()


class LunarKernelManager(BaseKernelManager, LunarKernelManagerMixin):
    """
    Kernel manager for generic lunar surface simulations.

    Loads static lunar kernels and optionally detailed DSKs.
    Provides core geometry and timing support for lunar simulations.
    """

    def __init__(
        self,
        frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME,
        detailed: bool = False,
        min_required_time: Time = None,
        max_required_time: Time = None,
        pre_load_static_kernels: bool = SPICE_PRELOAD,
        **kwargs,
    ):
        """
        Initialize the lunar kernel manager.

        Parameters:
            frame (Literal): Lunar reference frame ("MOON_ME" or "MOON_PA_DE440").
            detailed (bool): Whether to use detailed surface models (DSK).
            min_required_time (Time): Optional minimum required time.
            max_required_time (Time): Optional maximum required time.
            pre_load_static_kernels (bool): If True, loads static kernels on init.
            **kwargs: Passed to `setup_kernels()` in mixins.
        """
        super().__init__(
            min_required_time=min_required_time,
            max_required_time=max_required_time,
            pre_load_static_kernels=pre_load_static_kernels,
            frame=frame,
            detailed=detailed,
            **kwargs,
        )


class LROKernelManager(BaseKernelManager, LunarKernelManagerMixin, LROKernelManagerMixin):
    """
    Kernel manager for Lunar Reconnaissance Orbiter (LRO) datasets.

    Handles both static lunar kernels and dynamic sensor kernels such as
    LROC, DIVINER, and others. Can selectively enable CK kernel support
    for sensor pointing geometry.
    """

    def __init__(
        self,
        frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME,
        detailed: bool = False,
        pre_download_kernels: bool = SPICE_PRELOAD,
        pre_load_static_kernels: bool = SPICE_PRELOAD,
        diviner_ck: bool = False,
        lroc_ck: bool = False,
        keep_dynamic_kernels: bool = SPICE_PERSIST,
        min_required_time: Time = None,
        max_required_time: Time = None,
        **kwargs,
    ):
        """
        Initialize the LRO kernel manager.

        Parameters:
            frame (Literal): Lunar frame ("MOON_ME" or "MOON_PA_DE440").
            detailed (bool): Enable detailed DSK support.
            pre_download_kernels (bool): Download dynamic kernels at init.
            pre_load_static_kernels (bool): Load static kernels at init.
            diviner_ck (bool): Enable dynamic Diviner CK kernels.
            lroc_ck (bool): Enable dynamic LROC CK kernels.
            keep_dynamic_kernels (bool): Keep dynamic kernel files after use.
            min_required_time (Time): Optional time filtering lower bound.
            max_required_time (Time): Optional time filtering upper bound.
            **kwargs: Forwarded to mixin logic.
        """
        super().__init__(
            min_required_time=min_required_time,
            max_required_time=max_required_time,
            pre_download_kernels=pre_download_kernels,
            pre_load_static_kernels=pre_load_static_kernels,
            diviner_ck=diviner_ck,
            lroc_ck=lroc_ck,
            keep_dynamic_kernels=keep_dynamic_kernels,
            frame=frame,
            detailed=detailed,
            **kwargs,
        )


class GRAILKernelManager(BaseKernelManager, LunarKernelManagerMixin, GRAILKernelManagerMixin):
    """
    Kernel manager for GRAIL mission support.

    Extends the lunar kernel manager to include GRAIL-specific dynamic kernels
    for gravity data analysis and spacecraft tracking.
    """

    def __init__(
        self,
        frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME,
        detailed: bool = False,
        pre_download_kernels: bool = True,
        keep_dynamic_kernels: bool = SPICE_PERSIST,
        pre_load_static_kernels: bool = SPICE_PRELOAD,
        min_required_time: Time = None,
        max_required_time: Time = None,
        **kwargs,
    ):
        """
        Initialize the GRAIL kernel manager.

        Parameters:
            frame (Literal): Lunar frame ("MOON_ME" or "MOON_PA_DE440").
            detailed (bool): Enable detailed DSK support.
            pre_download_kernels (bool): Download dynamic kernels at init.
            keep_dynamic_kernels (bool): Keep dynamic kernel files after use.
            pre_load_static_kernels (bool): Load static kernels at init.
            min_required_time (Time): Optional time filtering lower bound.
            max_required_time (Time): Optional time filtering upper bound.
            **kwargs: Forwarded to GRAIL mixin logic.
        """
        super().__init__(
            min_required_time=min_required_time,
            max_required_time=max_required_time,
            pre_load_static_kernels=pre_load_static_kernels,
            frame=frame,
            detailed=detailed,
            pre_download_kernels=pre_download_kernels,
            keep_dynamic_kernels=keep_dynamic_kernels,
            **kwargs,
        )
