"""
====================================================
Lunar SPICE Kernel Management System
====================================================

Author: Michal Glos
Institution: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Project: Diploma Thesis – Space Applications

Overview:
---------
This module implements a modular and extensible kernel management system
for the SPICE toolkit, focusing on lunar missions such as LRO and GRAIL.
It provides structured handling of both static and dynamic kernels,
with support for:

  • Lunar reference frames (MOON_ME, MOON_PA_DE440)
  • Detailed vs. low-resolution DSK surface models
  • Time-windowed dynamic kernel loading
  • Automatic metadata management and kernel reuse
  • Kernel preloading, on-demand loading, and persistence control

Architecture:
-------------
The core design follows a mixin-based composition, enabling flexible
reusability across different mission configurations. Each mission or
dataset (e.g., LRO, GRAIL) provides its own mixin that augments the
base `BaseKernelManager` with appropriate static and dynamic kernels.

Classes:
--------
• BaseKernelManager:
    Core SPICE loader/unloader with time-bound control.

• LunarKernelManagerMixin:
    Adds universal lunar kernels and configurable DSK model support.

• LROKernelManagerMixin:
    Adds LRO spacecraft-specific instrument, frame, and CK/SPK kernels.

• GRAILKernelManagerMixin:
    Adds GRAIL mission-specific dual-satellite dynamic trajectory and CK data.

Usage:
------
Each final kernel manager class (e.g., `LROKernelManager`) inherits from
the base and appropriate mixins, and can be instantiated with various
options for dynamic kernel filtering, detailed surface models, and
activation time windows.

SPICE Toolkit:
--------------
This module is designed to be compatible with `spiceypy` and uses
NAIF-standard kernels fetched from public and local repositories.

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
    Just a simple helper for time translation from astropy.time.Time to ephemeris time
    Kernel is loaded upon instantialization of this object.
    """

    def __init__(self, min_required_time: Time = None, max_required_time: Time = None):
        super().__init__(min_required_time=min_required_time, max_required_time=max_required_time)
        self.load_static_kernels()


class LunarKernelManager(BaseKernelManager, LunarKernelManagerMixin):
    def __init__(
        self,
        frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME,
        detailed: bool = False,
        min_required_time: Time = None,
        max_required_time: Time = None,
        pre_load_static_kernels: bool = SPICE_PRELOAD,
        **kwargs,
    ):
        super().__init__(
            min_required_time=min_required_time,
            max_required_time=max_required_time,
            pre_load_static_kernels=pre_load_static_kernels,
            frame=frame,
            detailed=detailed,
            **kwargs,
        )


class LROKernelManager(BaseKernelManager, LunarKernelManagerMixin, LROKernelManagerMixin):
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


class GRAILKernelManager(LunarKernelManager, GRAILKernelManagerMixin):

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
