import numpy as np
import spiceypy as spice
from typing import Optional, Union


IMPLICIT_BORESIGHT = np.array([0, 0, 1])


class SubInstrument:

    _id: Union[int, str]
    shape: str
    sub_instrument_frame: str
    bounds: Optional[np.array]
    boresight: np.array

    def __init__(self, naif_id: int):
        self._id = naif_id
        self.shape, self.sub_instrument_frame, self.boresight, _, self.bounds = spice.getfov(naif_id, room=1000)

    def transformed_boresight(self, frame: str):
        return spice.mxv(spice.pxform(self.sub_instrument_frame, frame, 0), self.boresight)

    def transformed_bounds(self, frame: str):
        transformation_matrix = spice.pxform(self.sub_instrument_frame, frame, 0)
        return np.array([spice.mxv(transformation_matrix, point) for point in self.bounds])

    def transform_frame(self, frame: str):
        self.boresight = self.transformed_boresight(frame)
        self.bounds = self.transformed_bounds(frame)
        self.sub_instrument_frame = frame


class FrameCorrectedSubInstrument(SubInstrument):

    def __init__(self, naif_id: int, frame: str):
        super().__init__(naif_id)
        self.transform_frame(frame)


class ImplicitSubInstrument(SubInstrument):
    """When we do not have instrument kernels for instruments, it has to be configured manually, usually sourcing information from mission SIS files"""

    def __init__(self, _id: str, frame: str, boresight=IMPLICIT_BORESIGHT, bounds=None, transform_frame=None):
        self._id = _id
        self.boresight = (
            boresight if transform_frame is None else spice.mxv(spice.pxform(transform_frame, frame, 0), boresight)
        )
        self.bounds = (
            bounds
            if transform_frame is None or bounds is None
            else np.array([spice.mxv(spice.pxform(transform_frame, frame, 0), point) for point in bounds])
        )
        self.sub_instrument_frame = frame if transform_frame is None else transform_frame
        self.shape = "POINT"  # Maybe rect?


class DivinerSubInstrument(FrameCorrectedSubInstrument):

    A_SENSORS_KEY = "INS-85205_NUM_DETECTORS_FP_A"
    # TODO: Doublecheck the B orientation - direction of pixels change with different data sources
    B_SENSORS_KEY = "INS-85205_NUM_DETECTORS_FP_B"

    def __init__(self, naif_id: int, frame: str, _id: int):
        """_id is for the index of subinstrument in list of subsintruments. THis way we can cut off pixel vectors from merged pixel list"""
        super().__init__(naif_id=naif_id, frame=frame)

        # There are 6 A and 3 B subsensors. Ids start at 0
        key = self.A_SENSORS_KEY if _id < 6 else self.B_SENSORS_KEY
        num_detectors = int(spice.gdpool(key, 0, 1)[0])
        pixel_directions = np.array(spice.gdpool("INS-85205_DETECTOR_DIRS_FP_A", 0, num_detectors * 3)).reshape(
            num_detectors, 3
        )
        pixel_boresights = pixel_directions[(_id * 21) : ((_id + 1) * 21), :]
        # TODO: Add bounds, could be obtained with pixel halfwidths
        # pixel_sizes = np.array(spice.gdpool("INS-85205_DETECTOR_HW_FP_A", 0, num_detectors * 2)).reshape(num_detectors, 2)
        self.pixels = [
            ImplicitSubInstrument(f"{_id}_{i}", frame, boresight=boresight, transform_frame=frame)
            for i, boresight in enumerate(pixel_boresights)
        ]
