"""
====================================================
Lunar SPICE Kernel Manager
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

Description:
------------
This module provides a structured management system for SPICE kernels 
related to lunar exploration. It facilitates the loading, unloading, 
and organization of static and dynamic kernels for precise orbital 
calculations and planetary data processing.
"""

from abc import ABC
from typing import List, Literal, Optional
from collections import OrderedDict

from astropy.time import Time
import spiceypy as spice

from src.SPICE.config import (
    TIF_SAMPLE_RATE,
    SPICE_PERSIST,
    root_path,
    grail_path,
    lro_path,
    generic_url,
    grail_url,
    lro_url,
)
from src.global_config import LUNAR_FRAME
from src.SPICE.kernel_utils.spice_utils import (
    BaseKernel,
    LROCDynamicKernelLoader,
    LBLDynamicKernelLoader,
    StaticKernelManager,
    AutoUpdateKernel,
    PriorityKernelManagement,
)
from src.SPICE.kernel_utils.detailed_model import DetailedModelDSKKernel




class BaseKernelManager(ABC):
    """
    This SweepIterator ensures we have correct SPICE files loaded for given datetime

    Static kernels are loaded upon init
    Dynamic kernels are loaded upon request
    """

    keys = ["lsk", "sclk", "pck", "fk", "bpck", "ik", "ck", "spk", "dsk"]

    def __init__(self):
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

    def unload_all(self):
        for static_kernel in self.static_kernels.values():
            for kernel in static_kernel:
                kernel.unload()
        for dynamic_kernel in self.dynamic_kernels:
            dynamic_kernel.unload()

    def load_static_kernels(self) -> None:
        self.static_kernel_manager = StaticKernelManager(self.static_kernels)
        self.static_kernel_manager.load()

    @property
    def min_loaded_time(self) -> Time:
        return max([kernel.min_loaded_time for kernel in self.dynamic_kernels]) if self.dynamic_kernels else None

    @property
    def max_loaded_time(self) -> Time:
        return min([kernel.max_loaded_time for kernel in self.dynamic_kernels]) if self.dynamic_kernels else None

    def step(self, time: Time):
        return all([kernel.reload_kernels(time) for kernel in self.dynamic_kernels])

    def activate(self, activation_time: Time = None) -> None:
        """
        Activate the kernel manager for given datetime
        """
        self.load_static_kernels()
        if not self.step(self.min_loaded_time if activation_time is None else activation_time):
            raise ValueError("Some of dynamic SPICE kernels were not loaded for required time")


class LunarKernelManager(BaseKernelManager):
    # spice.furnsh(LUNAR_MODEL["dsk_path"])
    def __init__(self, frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME, detailed: bool = False):
        """
        You can choose the lunar frame with frame and DSK model - more detailed have to be compiled locally
        """
        super().__init__()
        # Following two kernels are universal and always needed
        self.static_kernels["bpck"].append(
            BaseKernel(generic_url("pck/moon_pa_de421_1900-2050.bpc"), root_path("pck/moon_pa_de421_1900-2050.bpc"))
        )
        self.static_kernels["fk"].append(
            BaseKernel(generic_url("fk/satellites/moon_080317.tf"), root_path("fk/moon_080317.tf"))
        )

        if frame == "MOON_ME":
            self.static_kernels["fk"] += [
                BaseKernel(generic_url("fk/satellites/moon_assoc_me.tf"), root_path("fk/moon_assoc_me.tf")),
                BaseKernel(generic_url("fk/satellites/moon_assoc_pa.tf"), root_path("fk/moon_assoc_pa.tf")),
            ]
        elif frame == "MOON_PA_DE440":
            self.static_kernels["fk"].append(
                BaseKernel(generic_url("fk/satellites/moon_de440_220930.tf"), root_path("fk/moon_de440_220930.tf"))
            )
            self.static_kernels["bpck"].append(
                BaseKernel(generic_url("pck/moon_pa_de440_200625.bpc"), root_path("pck/moon_pa_de440_200625.bpc"))
            )
            self.static_kernels["spk"].append(
                BaseKernel(generic_url("spk/planets/de440.bsp"), root_path("spk/de440.bsp"))
            )
        else:
            raise NotImplementedError("Only MOON_ME and MOON_PA_DE440 frames are supported")

        if detailed:
            dsk_filename = "custom_lunar_model_" + f"{TIF_SAMPLE_RATE:.4}".replace(".", "_") + ".dsk"
            self.static_kernels["dsk"].append(
                DetailedModelDSKKernel(root_path(dsk_filename), tif_sample_rate=TIF_SAMPLE_RATE)
            )
        else:
            self.static_kernels["dsk"].append(
                BaseKernel(
                    "https://naif.jpl.nasa.gov/pub/naif/pds/wgc/lessons/event_finding_kplo_old/kernels/dsk/moon_lowres.bds",
                    root_path("moon_lowres.bds"),
                )
            )


class LROKernelManager(LunarKernelManager):
    def __init__(self, frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME, detailed: bool = False, pre_download_kernels: bool = True, diviner_ck: bool = False, lroc_ck: bool = False):
        """
        Above the Lunar kernel manager, add LRO specific SPICE kernels too
        
        Args:
            frame (Literal["MOON_ME", "MOON_PA_DE440"], optional): Lunar frame. Defaults to LUNAR_FRAME.
            detailed (bool, optional): Use detailed DSK model. Defaults to False to use MOON_LOWRES.dsk
            pre_download_kernels (bool, optional): Pre-download kernels, otherwise download just before loading. Defaults
            diviner_ck (bool, optional): Use DIVINER CK kernels. Defaults to False.
        """
        super().__init__(frame=frame, detailed=detailed)
        # Spacecraft clock
        self.static_kernels["sclk"].append(AutoUpdateKernel(lro_url("sclk/"), lro_path("sclk"), r"lro_clkcor.*.tsc"))
        self.static_kernels["fk"] += [
            # Frame kernels
            BaseKernel(lro_url("fk/lro_dlre_frames_2010132_v04.tf"), lro_path("fk/lro_dlre_frames_2010132_v04.tf")),
            # BaseKernel(grail_url("fk/lro_frames_2010214_v01.tf"), lro_path("fk/lro_frames_2010214_v01.tf")),
            # BaseKernel(lro_url("fk/lro_frames_2012255_v02.tf"), lro_path("fk/lro_frames_2012255_v02.tf")),
            # With the CK temp.corrected frames
            BaseKernel("https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/fk/lro_frames_2014049_v01.tf", lro_path("fk/lro_frames_2014049_v01.tf")),
        ]
        self.static_kernels["ik"] += [
            # Instrument kernels
            BaseKernel(lro_url("ik/lro_crater_v03.ti"), lro_path("ik/lro_crater_v03.ti")),
            BaseKernel(lro_url("ik/lro_dlre_v05.ti"), lro_path("ik/lro_dlre_v05.ti")),
            BaseKernel(lro_url("ik/lro_lamp_v03.ti"), lro_path("ik/lro_lamp_v03.ti")),
            BaseKernel(lro_url("ik/lro_lend_v00.ti"), lro_path("ik/lro_lend_v00.ti")),
            BaseKernel(lro_url("ik/lro_lola_v00.ti"), lro_path("ik/lro_lola_v00.ti")),
            BaseKernel(
                "https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/ik/lro_lroc_v19.ti", lro_path("ik/lro_lroc_v19.ti")
            ),
        ]
        # Define dynamic kernels
        self.dynamic_kernels = [
            # CK kernels of LRO
            LBLDynamicKernelLoader(lro_path("ck"), lro_url("ck/"), r"lrosc.*.bc", r"lrosc.*.lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST),
            # LRO trajectory
            LBLDynamicKernelLoader(lro_path("spk"), lro_url("spk/"), r"lrorg.*.bsp", r"lrorg.*.lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST),
        ]
        if diviner_ck:
            # For now, callbacks serve no actual purpose. Left here in case it would be needed
            # diviner_callback = [ (lambda et: spice.pxform("LRO_DLRE", "LRO_DLRE", et), [0], {}) ]
            self.dynamic_kernels.append(
                # CK kernels of DIVINER
                # LBLDynamicKernelLoader(lro_path("ck"), lro_url("ck/"), r"lrodv.*.bc", r"lrodv.*.lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST, load_callbacks=diviner_callback),
                LBLDynamicKernelLoader(lro_path("ck"), lro_url("ck/"), r"lrodv.*.bc", r"lrodv.*.lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST),
            )
        if lroc_ck:
            self.dynamic_kernels.append(
                # LROC temperature corrected kernels
                LROCDynamicKernelLoader(lro_path("ck"), "https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/ck/", r"lrolc.*.bc", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST)
            )

class GRAILKernelManager(LunarKernelManager):

    def __init__(self, frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME, detailed: bool = False, pre_download_kernels: bool = True):
        super().__init__(frame=frame, detailed=detailed)
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
            LBLDynamicKernelLoader(grail_path("ck"), grail_url("ck/"), r"gra_rec.*.bc", r"gra_rec.*.lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST),
            LBLDynamicKernelLoader(grail_path("ck"), grail_url("ck/"), r"grb_rec.*.bc", r"grb_rec.*.lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST),
            PriorityKernelManagement(
                [
                    LBLDynamicKernelLoader(
                        grail_path("spk"),
                        grail_url("spk/"),
                        r"^(?!grail_120301_120529_sci_v01\.bsp$).*grail.*sci.*\.bsp$",
                        r"^(?!grail_120301_120529_sci_v01\.lbl$).*grail.*sci.*\.lbl$",
                        pre_download_kernels=pre_download_kernels,
                        files_persist=SPICE_PERSIST,
                    ),
                    LBLDynamicKernelLoader(grail_path("spk"), grail_url("spk/"), r"grail.*nav.*bsp", r"grail.*nav.*lbl", pre_download_kernels=pre_download_kernels, files_persist=SPICE_PERSIST),
                ]
            ),
        ]
