"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import os

from astropy.time import Time

from src.SPICE.config import (
    SPICE_PERSIST,
    grail_path,
    grail_url,
)
from src.global_config import LUNAR_FRAME
from src.SPICE.kernel_utils.spice_kernels import (
    BaseKernel,
    LBLDynamicKernelLoader,
    AutoUpdateKernel,
    PriorityKernelLoader,
)
from src.SPICE.kernel_utils.kernel_manager_mixins.base_mixin import BaseKernelManagerMixin

class GRAILKernelManagerMixin(BaseKernelManagerMixin):

    def setup_kernels(
        self,
        pre_download_kernels: bool = True,
        keep_dynamic_kernels: bool = SPICE_PERSIST,
        min_required_time: Time = None,
        max_required_time: Time = None,
        **kwargs,
    ):
        if keep_dynamic_kernels and os.getenv("PURGE_DYNAMIC_KERNELS"):
            keep_dynamic_kernels = False
        # Spacecraft clock
        self.static_kernels["sclk"].append(
            AutoUpdateKernel(grail_url("sclk/"), grail_path("sclk"), r"grb_sclkscet.*.tsc")
        )
        # Grail frames
        self.static_kernels["fk"].append(BaseKernel(grail_url("fk/grail_v07.tf"), grail_path("fk/grail_v07.tf")))
        # Instrument kernels for both satellites
        self.static_kernels["ik"] += [
            BaseKernel(grail_url("ik/gra_sta_v01.ti"), grail_path("ik/gra_sta_v01.ti")),
            BaseKernel(grail_url("ik/grb_sta_v01.ti"), grail_path("ik/grb_sta_v01.ti")),
        ]

        # Dynamic kernels - for both GRAIL satellites - A & B
        self.dynamic_kernels = [
            LBLDynamicKernelLoader(
                grail_path("ck"),
                grail_url("ck/"),
                r"gra_rec.*.bc",
                r"gra_rec.*.lbl",
                pre_download_kernels=pre_download_kernels,
                min_time_to_load=min_required_time,
                max_time_to_load=max_required_time,
                keep_kernels=keep_dynamic_kernels,
            ),
            LBLDynamicKernelLoader(
                grail_path("ck"),
                grail_url("ck/"),
                r"grb_rec.*.bc",
                r"grb_rec.*.lbl",
                pre_download_kernels=pre_download_kernels,
                min_time_to_load=min_required_time,
                max_time_to_load=max_required_time,
                keep_kernels=keep_dynamic_kernels,
            ),
            PriorityKernelLoader(
                [
                    LBLDynamicKernelLoader(
                        grail_path("spk"),
                        grail_url("spk/"),
                        r"^(?!grail_120301_120529_sci_v01\.bsp$).*grail.*sci.*\.bsp$",
                        r"^(?!grail_120301_120529_sci_v01\.lbl$).*grail.*sci.*\.lbl$",
                        pre_download_kernels=pre_download_kernels,
                        min_time_to_load=min_required_time,
                        max_time_to_load=max_required_time,
                        keep_kernels=keep_dynamic_kernels,
                    ),
                    LBLDynamicKernelLoader(
                        grail_path("spk"),
                        grail_url("spk/"),
                        r"grail.*nav.*bsp",
                        r"grail.*nav.*lbl",
                        pre_download_kernels=pre_download_kernels,
                        min_time_to_load=min_required_time,
                        max_time_to_load=max_required_time,
                        keep_kernels=keep_dynamic_kernels,
                    ),
                ]
            ),
        ]
