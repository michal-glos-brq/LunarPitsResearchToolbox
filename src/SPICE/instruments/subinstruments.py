from typing import Optional, Union, Literal, Dict

import numpy as np
import spiceypy as spice


from src.SPICE.config import IMPLICIT_BORESIGHT, DIVINER_SUBINSTRUMENT_PIXEL_COUNT, LRO_MINIRF_FRAME_STR_ID


class SubInstrument:

    _id: Union[int, str]
    _matrices: Dict
    _ets: Dict
    shape: str
    sub_instrument_frame: str
    bounds: Optional[np.array]
    boresight: np.array

    def __init__(self, naif_id: int):
        self._id = naif_id
        self.shape, self.sub_instrument_frame, self.boresight, _, self.bounds = spice.getfov(naif_id, room=1000)
        self._base_setup()

    def _base_setup(self):
        self._matrices = {}
        self._ets = {}

    def transform_frame(self, frame: str, et: float = 0):
        """Transforms boresight and bounds to the given frame"""
        if self._ets.get(frame) != et:
            self._matrices[frame] = spice.pxform(self.sub_instrument_frame, frame, et)
            self._ets[frame] = et
        return self._matrices[frame]

    def transformed_boresight(self, frame: str, et: float = 0):
        """Returns boresight transformed to the given frame"""
        matrix = self.transform_frame(frame, et)
        return spice.mxv(matrix, self.boresight)

    def transformed_bounds(self, frame: str, et: float = 0):
        """Returns bounds transformed to the given frame"""
        matrix = self.transform_frame(frame, et)
        return np.array([spice.mxv(matrix, point) for point in self.bounds])

    def transform_vector(self, frame: str, vector: np.array, et: float = 0):
        """Transforms vector to the given frame"""
        matrix = self.transform_frame(frame, et)
        return spice.mxv(matrix, vector)

    def transform_to_frame(self, frame: str, et: float = 0):
        """Transforms boresight and bounds to the given frame"""
        self.boresight = self.transformed_boresight(frame, et=et)
        self.bounds = self.transformed_bounds(frame, et=et)
        self.sub_instrument_frame = frame


class ImplicitSubInstrument(SubInstrument):
    """When we do not have instrument kernels for instruments, it has to be configured manually, usually sourcing information from mission SIS files"""

    def __init__(
        self,
        _id: str,
        frame: str,
        boresight=IMPLICIT_BORESIGHT,
        bounds=None,
        transform_frame=None,
        shape: str = "POINT",
    ):
        """Non-SPICE KERNEL instruments"""
        self._id = _id
        self._base_setup()
        # Here we facilitate transformation only from and to static franmes or static trans. matrix
        self.boresight = (
            boresight if transform_frame is None else spice.mxv(spice.pxform(transform_frame, frame, 0), boresight)
        )
        self.bounds = (
            bounds
            if transform_frame is None or bounds is None
            else np.array([spice.mxv(spice.pxform(transform_frame, frame, 0), point) for point in bounds])
        )
        self.sub_instrument_frame = frame if transform_frame is None else transform_frame
        self.shape = shape  # Maybe rect?


class DivinerSubInstrument(SubInstrument):
    """
    Diviner has 6 A and 3 B subinstruments. Each subinstrument has 21 pixels

    In total 189 subinstruments (pixels) and 9 subinstruments - pixel lines
    Hiearchical structure is nusefull for approximate boresight projections and rough filtering
    """

    # TODO: Doublecheck the B orientation - direction of pixels change with different data sources
    # It is in RDR SIS document for DIVINER ### Now should be correct, but still doublecheck eventually

    def __init__(
        self, naif_id: int, _id: int, pixel_key: Literal["INS-85205_DETECTOR_DIRS_FP_A", "INS-85205_DETECTOR_DIRS_FP_B"]
    ):
        """
        Single channel of DIVINER instrument

        naif_if: int - SPICE index of the instrument channel (Divided into channels (rows of 21 pixels))
        frame: str -
        """
        super().__init__(naif_id=naif_id)

        index_start, get_numbers = _id * DIVINER_SUBINSTRUMENT_PIXEL_COUNT * 3, DIVINER_SUBINSTRUMENT_PIXEL_COUNT * 3
        pixel_boresights = np.array(spice.gdpool(pixel_key, index_start, get_numbers)).reshape((21, 3))
        if pixel_key[-1] == "B":
            pixel_boresights = pixel_boresights[::-1, :]
        self.pixels = [
            # Name is index + 1, because they are indexed from 1 in docs
            ImplicitSubInstrument(f"{_id+1}_{i+1}{pixel_key[-1]}", self.sub_instrument_frame, boresight=boresight)
            for i, boresight in enumerate(pixel_boresights)
        ]


class MiniRFSubInstrument(ImplicitSubInstrument):
    """
    MiniRF subinstrument implementation.

    Since MiniRF does not have a conventional SPICE instrument kernel, we use
    fixed (document-derived) parameters to define its boresight and FOV.

    The MRFLRO_DP_SIS document indicates that MiniRF has two modes:
      - S-band: Beam width ~3.6° (across) x 10.8° (along)
      - X-band: Beam width ~1.2° (across) x 3.6° (along)

    """

    def __init__(self, channel: str):
        # Use a fixed boresight – in practice this might be refined
        boresight = np.array([0.0, 0.0, 1.0])
        # Set the instrument frame; here we assume "MINIRF" as a placeholder.
        frame = LRO_MINIRF_FRAME_STR_ID

        # Select beam widths based on channel.
        # Beam widths are in degrees. For S-band, we have 3.6° (across) and 10.8° (along);
        # for X-band, 1.2° (across) and 3.6° (along).
        if channel.upper() == "S":
            across_deg, along_deg = 3.6, 10.8
        elif channel.upper() == "X":
            across_deg, along_deg = 1.2, 3.6
        else:
            raise ValueError("Channel must be 'S' or 'X'")

        # Compute half-angles in radians
        half_across = np.radians(across_deg / 2.0)
        half_along = np.radians(along_deg / 2.0)

        # rotation about X
        def rot_x(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([
                [1,  0,  0],
                [0,  c, -s],
                [0,  s,  c],
            ])

        # rotation about Y
        def rot_y(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([
                [ c, 0,  s],
                [ 0, 1,  0],
                [-s, 0,  c],
            ])

        # build the four corners: (+across,+along), (-across,+along), (-across,-along), (+across,-along)
        corners = []
        for sign_x, sign_y in [(+1,+1), (-1,+1), (-1,-1), (+1,-1)]:
            # first rotate boresight by sign_x * half_across about Y,
            # then by sign_y * half_along about X
            R = rot_x(sign_y * half_along) @ rot_y(sign_x * half_across)
            corners.append(R @ boresight)

        bounds = np.vstack(corners)  # shape (4,3), already unit-length

        left_bound  = rot_x(-half_along) @ boresight
        right_bound = rot_x(+half_along) @ boresight
        self.line_bounds = np.vstack([left_bound, right_bound])  # shape (2,3)


        # hand off to the base class
        super().__init__(f"minirf-{channel}", frame,
                         boresight=boresight,
                         bounds=bounds)

        self.channel = channel.upper()

    def look_vector_for_pixel(self, pixel_index: int, pixel_count: int) -> np.ndarray:
        """
        Returns the unit look-vector (instrument frame) for a given pixel
        by linearly interpolating between left and right line_bounds.
        """
        frac = (pixel_index + 0.5) / pixel_count
        vec  = (1.0 - frac) * self.line_bounds[0] + frac * self.line_bounds[1]
        return vec / np.linalg.norm(vec)

