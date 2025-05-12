from typing import Literal

from src.SPICE.config import (
    TIF_SAMPLE_RATE,
    root_path,
    generic_url,
)
from src.global_config import LUNAR_FRAME
from src.SPICE.kernel_utils.spice_kernels import (
    BaseKernel,
)
from src.SPICE.kernel_utils.detailed_model import DetailedModelDSKKernel


from src.SPICE.kernel_utils.kernel_manager_mixins.base_mixin import BaseKernelManagerMixin


class LunarKernelManagerMixin(BaseKernelManagerMixin):
    # spice.furnsh(LUNAR_MODEL["dsk_path"])
    def setup_kernels(self, frame: Literal["MOON_ME", "MOON_PA_DE440"] = LUNAR_FRAME, detailed: bool = False, **kwargs):
        """
        You can choose the lunar frame with frame and DSK model - more detailed have to be compiled locally
        """
        self.main_reference_frame = frame

        self.static_kernels["fk"].append(
            BaseKernel(generic_url("fk/satellites/moon_080317.tf"), root_path("fk/moon_080317.tf"))
        )

        if frame == "MOON_ME":
            self.static_kernels["bpck"].append(
                BaseKernel(generic_url("pck/moon_pa_de421_1900-2050.bpc"), root_path("pck/moon_pa_de421_1900-2050.bpc"))
            )
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
