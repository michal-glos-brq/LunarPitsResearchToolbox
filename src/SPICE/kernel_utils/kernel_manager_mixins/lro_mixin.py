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
from typing import Literal

from astropy.time import Time

from src.SPICE.config import (
    SPICE_PERSIST,
    lro_path,
    lro_url,
)
from src.global_config import LUNAR_FRAME
from src.SPICE.kernel_utils.spice_kernels import (
    BaseKernel,
    LROCDynamicKernelLoader,
    LBLDynamicKernelLoader,
    AutoUpdateKernel,
)
from src.SPICE.kernel_utils.kernel_manager_mixins.base_mixin import BaseKernelManagerMixin


class LROKernelManagerMixin(BaseKernelManagerMixin):
    def setup_kernels(
        self,
        frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME,
        pre_download_kernels: bool = True,
        diviner_ck: bool = False,
        lroc_ck: bool = False,
        keep_dynamic_kernels: bool = SPICE_PERSIST,
        min_required_time: Time = None,
        max_required_time: Time = None,
        **kwargs,
    ):
        """
        Above the Lunar kernel manager, add LRO specific SPICE kernels too

        Args:
            frame (Literal["MOON_ME", "MOON_PA_DE440"], optional): Lunar frame. Defaults to LUNAR_FRAME.
            detailed (bool, optional): Use detailed DSK model. Defaults to False to use MOON_LOWRES.dsk
            pre_download_kernels (bool, optional): Pre-download kernels, otherwise download just before loading. Defaults
            diviner_ck (bool, optional): Use DIVINER CK kernels. Defaults to False.
            lroc_ck (bool, optional): Use LROC CK kernels. Defaults to False.
            lola_ck (bool, optional): Use LOLA CK kernels. Defaults to False.
            keep_dynamic_kernels (bool, optional): Do not delete dynamic kernels once unloaded. Defaults to True. Can be overriden
                by ENV var PURGE_DYNAMIC_KERNELS set to 1
        """
        if keep_dynamic_kernels and os.getenv("PURGE_DYNAMIC_KERNELS"):
            keep_dynamic_kernels = False

        # Spacecraft clock
        self.static_kernels["sclk"].append(AutoUpdateKernel(lro_url("sclk/"), lro_path("sclk"), r"lro_clkcor.*.tsc"))
        self.static_kernels["fk"] += [
            # Frame kernels
            BaseKernel(lro_url("fk/lro_dlre_frames_2010132_v04.tf"), lro_path("fk/lro_dlre_frames_2010132_v04.tf")),
            # BaseKernel(grail_url("fk/lro_frames_2010214_v01.tf"), lro_path("fk/lro_frames_2010214_v01.tf")),
            # BaseKernel(lro_url("fk/lro_frames_2012255_v02.tf"), lro_path("fk/lro_frames_2012255_v02.tf")),
            # With the CK temp.corrected frames
            BaseKernel(
                "https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/fk/lro_frames_2014049_v01.tf",
                lro_path("fk/lro_frames_2014049_v01.tf"),
            ),
        ]
        self.static_kernels["ik"] += [
            # Instrument kernels
            BaseKernel(lro_url("ik/lro_crater_v03.ti"), lro_path("ik/lro_crater_v03.ti")),
            BaseKernel(lro_url("ik/lro_dlre_v05.ti"), lro_path("ik/lro_dlre_v05.ti")),
            BaseKernel(lro_url("ik/lro_lamp_v03.ti"), lro_path("ik/lro_lamp_v03.ti")),
            BaseKernel(lro_url("ik/lro_lend_v00.ti"), lro_path("ik/lro_lend_v00.ti")),
            BaseKernel(
                "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/geometry/lro_lola_v01.ti",
                lro_path("ik/lro_lola_v00.ti"),
            ),
            BaseKernel(
                "https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/ik/lro_lroc_v19.ti", lro_path("ik/lro_lroc_v19.ti")
            ),
        ]
        # Define dynamic kernels
        self.dynamic_kernels = [
            # CK kernels of LRO
            LBLDynamicKernelLoader(
                lro_path("ck"),
                lro_url("ck/"),
                r"lrosc.*.bc",
                r"lrosc.*.lbl",
                pre_download_kernels=pre_download_kernels,
                min_time_to_load=min_required_time,
                max_time_to_load=max_required_time,
                keep_kernels=keep_dynamic_kernels,
            ),
        ]
        if frame == "MOON_ME":
            # Not so precise data
            self.dynamic_kernels.append(
                LBLDynamicKernelLoader(
                    lro_path("spk"),
                    lro_url("spk/"),
                    r"lrorg.*.bsp",
                    r"lrorg.*.lbl",
                    pre_download_kernels=pre_download_kernels,
                    min_time_to_load=min_required_time,
                    max_time_to_load=max_required_time,
                    keep_kernels=keep_dynamic_kernels,
                ),
            )
        else:
            # Much more precise data
            self.dynamic_kernels.append(
                LBLDynamicKernelLoader(
                    lro_path("spk"),
                    "https://pds-geosciences.wustl.edu/lro/lro-l-rss-1-tracking-v1/lrors_0001/data/spk/",
                    r"/lro/lro-l-rss-1-tracking-v1/lrors_0001/data/spk/lro_.*grgm900c_l600.*bsp",
                    r"/lro/lro-l-rss-1-tracking-v1/lrors_0001/data/spk/lro_.*grgm900c_l600.*lbl",
                    pre_download_kernels=pre_download_kernels,
                    min_time_to_load=min_required_time,
                    max_time_to_load=max_required_time,
                    keep_kernels=keep_dynamic_kernels,
                )
            )
        if diviner_ck:
            self.dynamic_kernels.append(
                # CK kernels of DIVINER
                LBLDynamicKernelLoader(
                    lro_path("ck"),
                    lro_url("ck/"),
                    r"lrodv.*.bc",
                    r"lrodv.*.lbl",
                    pre_download_kernels=pre_download_kernels,
                    min_time_to_load=min_required_time,
                    max_time_to_load=max_required_time,
                    keep_kernels=keep_dynamic_kernels,
                ),
            )
        if lroc_ck:
            self.dynamic_kernels.append(
                # LROC temperature corrected kernels
                LROCDynamicKernelLoader(
                    lro_path("ck"),
                    "https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/ck/",
                    r"lrolc.*.bc",
                    pre_download_kernels=pre_download_kernels,
                    min_time_to_load=min_required_time,
                    max_time_to_load=max_required_time,
                    keep_kernels=keep_dynamic_kernels,
                )
            )
