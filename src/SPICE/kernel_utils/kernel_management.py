"""Sweeper class groups Dynamic and static kernels and manages them as a whole throughout the simulation"""

import os
import urllib
from functools import partial
from typing import List, Literal
from abc import ABC
from collections import OrderedDict

from astropy import Time

from src.global_config import TQDM_NCOLS
from src.config import (
    LUNAR_MODEL,
    SPICE_FOLDER,
    LRO_SPICE_KERNEL_BASE_URL,
    GRAIL_SPICE_KERNEL_BASE_URL,
    GENERIC_SPICE_KERNEL_BASE_URL,
)
from src.kernel_utils.spice_utils import (
    BaseKernel,
    LBLDynamicKernelLoader,
    StaticKernelManager,
    AutoUpdateKernel,
    PriorityKernelManagement,
)
from src.kernel_utils.detailed_model import DetailedModelDSKKernel

grail_path = partial(os.path.join, SPICE_FOLDER, "grail")
lro_path = partial(os.path.join, SPICE_FOLDER, "lro")
selene_path = partial(os.path.join, SPICE_FOLDER, "selene")
chandrayaan_path = partial(os.path.join, SPICE_FOLDER, "chandrayaan")
root_path = partial(os.path.join, SPICE_FOLDER)

grail_url = partial(urllib.parse.urljoin, GRAIL_SPICE_KERNEL_BASE_URL)
lro_url = partial(urllib.parse.urljoin, LRO_SPICE_KERNEL_BASE_URL)
#selene_url = partial(urllib.parse.urljoin, SELENE_SPICE_KERNEL_BASE_URL)
#chandrayaan_url = partial(urllib.parse.urljoin, CHANDRAYAAN_SPICE_KERNEL_BASE_URL)
generic_url = partial(urllib.parse.urljoin, GENERIC_SPICE_KERNEL_BASE_URL)


class BaseKernelManager(ABC):
    """
    This SweepIterator ensures we have correct SPICE files loaded for given datetime

    Static kernels are loaded upon init
    Dynamic kernels are loaded upon request
    """

    static_kernels: OrderedDict[str, List[BaseKernel]] = OrderedDict(
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
    dynamic_kernels = []

    def __init__(self):
        self.activate(self.min_loaded_time)

    def unload_all(self):
        for static_kernel in self.static_kernels.values():
            for kernel in static_kernel:
                kernel.unload()
        for dynamic_kernel in self.dynamic_kernels:
            dynamic_kernel.unload()

    def load_static_kernels(self) -> None:
        self.static_kernel_manager = StaticKernelManager(self.static_kernels)
        self.static_kernel_manager.furnsh()

    @property
    def min_loaded_time(self) -> Time:
        return max([kernel.min_loaded_time for kernel in self.dynamic_kernels])

    @property
    def max_loaded_time(self) -> Time:
        return min([kernel.max_loaded_time for kernel in self.dynamic_kernels])

    def step(self, time: Time):
        return all([kernel.reload_kernels(time) for kernel in self.dynamic_kernels])

    def activate(self, starting_datetime: Time) -> None:
        self.load_static_kernels()
        if not self.step(starting_datetime):
            raise ValueError("Some of dynamic SPICE kernels were not loaded for required time")


class LunarKernelManager(BaseKernelManager):
    # spice.furnsh(LUNAR_MODEL["dsk_path"])
    def __init__(self, frame: Literal["MOON_ME", "MOON_PA_DE440"], detailed: bool = False):
        """
        You can choose the lunar frame with frame and DSK model - more detailed have to be compiled locally
        """
        # Following two kernels are universal and always needed
        self.static_kernels["bpck"].append(
            BaseKernel(generic_url("pck/moon_pa_de421_1900-2050.bpc"), root_path("pck/moon_pa_de421_1900-2050.bpc"))
        )
        self.static_kernels["fk"].append(
            BaseKernel(generic_url("fk/satellites/moon_080317.tf"), root_path("fk/satellites/moon_080317.tf"))
        )

        if frame == "MOON_ME":
            self.static_kernels["fk"] += [
                BaseKernel(generic_url("fk/satellites/moon_assoc_me.tf"), root_path("fk/moon_assoc_me.tf")),
                BaseKernel(generic_url("fk/satellites/moon_assoc_pa.tf"), root_path("fk/moon_assoc_pa.tf")),
            ]
        elif frame == "MOON_PA_DE440":
            self.static_kernels["fk"].append(
                generic_url("fk/satellites/moon_de440_220930.tf"), root_path("fk/moon_de440_220930.tf")
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
            self.static_kernels["dsk"].append(DetailedModelDSKKernel(root_path("custom_lunar_model_0_075.dsk"), tif_sample_rate=0.075))
        else:
            self.static_kernels["dsk"].append(
                BaseKernel(
                    "https://naif.jpl.nasa.gov/pub/naif/pds/wgc/lessons/event_finding_kplo_old/kernels/dsk/moon_lowres.bds",
                    root_path("moon_lowres.bds"),
                )
            )
        super().__init__()


class LROKernelManager(LunarKernelManager):
    def __init__(self, frame: Literal["MOON_ME", "MOON_PA_DE440"], detailed: bool = False):
        """Above the Lunar kernel manager, add LRO specific SPICE kernels too"""
        # Spacecraft clock
        self.static_kernels["sclk"].append(AutoUpdateKernel(lro_url("sclk"), lro_path("sclk"), r"lro_clkcor.*.tsc"))
        self.static_kernels["fk"] += [
            # Frame kernels
            BaseKernel(lro_url("fk/lro_dlre_frames_2010132_v04.tf"), lro_path("fk/lro_dlre_frames_2010132_v04.tf")),
            # BaseKernel(grail_url("fk/lro_frames_2010214_v01.tf"), lro_path("fk/lro_frames_2010214_v01.tf")),
            BaseKernel(lro_url("fk/lro_frames_2012255_v02.tf"), lro_path("fk/lro_frames_2012255_v02.tf")),
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
            # CK kernels of DIVINER
            LBLDynamicKernelLoader(lro_path("ck"), lro_url("ck"), r"lrodv.*.bc", r"lrodv.*.lbl"),
            # CK kernels of LRO
            LBLDynamicKernelLoader(lro_path("ck"), lro_url("ck"), r"lrosc.*.bc", r"lrosc.*.lbl"),
            # LRO trajectory
            LBLDynamicKernelLoader(lro_path("spk"), lro_url("spk"), r"lrorg.*.bsp", r"lrorg.*.lbl"),
        ]
        super().__init__(frame=frame, detailed=detailed)


class GRAILKernelManager(LunarKernelManager):

    def __init__(self, frame: Literal["MOON_ME", "MOON_PA_DE440"], detailed: bool = False):
        # Spacecraft clock
        self.static_kernels["sclk"].append(
            AutoUpdateKernel(grail_url("sclk"), grail_path("sclk"), r"grb_sclkscet.*.tsc")
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
            LBLDynamicKernelLoader(grail_path("ck"), grail_url("ck"), r"gra_rec.*.bc", r"gra_rec.*.lbl"),
            LBLDynamicKernelLoader(grail_path("ck"), grail_url("ck"), r"grb_rec.*.bc", r"grb_rec.*.lbl"),
            PriorityKernelManagement(
                [
                    LBLDynamicKernelLoader(
                        grail_path("spk"),
                        grail_url("spk"),
                        r"^(?!grail_120301_120529_sci_v01\.bsp$).*grail.*sci.*\.bsp$",
                        r"^(?!grail_120301_120529_sci_v01\.lbl$).*grail.*sci.*\.lbl$",
                    ),
                    LBLDynamicKernelLoader(grail_path("spk"), grail_url("spk"), r"grail.*nav.*bsp", r"grail.*nav.*lbl"),
                ]
            ),
        ]
        super().__init__(frame=frame, detailed=detailed)
