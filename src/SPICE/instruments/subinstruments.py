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

    These are used here to define a rectangular FOV (via four corner vectors).
    For simplicity, we assume that the nominal boresight is [0, 0, 1] in the
    instrument frame, and that a small-angle linear approximation is acceptable.
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

        # We need two orthogonal directions in the instrument's tangent plane.
        # For simplicity, we assume a default "right" vector and "down" vector.
        # These choices are arbitrary for an implicit definition.
        right = np.array([1.0, 0.0, 0.0])
        down = np.array([0.0, -1.0, 0.0])

        # Compute approximate corner vectors in the instrument frame.
        # For small angles, the perturbed direction can be approximated as:
        #  boresight + (half_across)*right + (half_along)*down,
        # and similarly for the other three corners.
        corner1 = boresight + half_across * right + half_along * down
        corner2 = boresight - half_across * right + half_along * down
        corner3 = boresight - half_across * right - half_along * down
        corner4 = boresight + half_across * right - half_along * down
        corner1 /= np.linalg.norm(corner1)
        corner2 /= np.linalg.norm(corner2)
        corner3 /= np.linalg.norm(corner3)
        corner4 /= np.linalg.norm(corner4)
        bounds = np.array([corner1, corner2, corner3, corner4])

        # Now call the ImplicitSubInstrument initializer.
        # (ImplicitSubInstrument expects: _id, frame, boresight, bounds, transform_frame)
        # Here we assume that no additional transformation is required.
        super().__init__(f"minirf-{channel}", frame, boresight=boresight, bounds=bounds)

        # Store the channel information for later reference.
        self.channel = channel
