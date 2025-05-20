"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

This module defines subinstrument classes used to represent sensor units on board
spacecraft for geometric modeling. It includes:

    - SPICE-based instruments (SubInstrument)
    - Manually defined instruments (ImplicitSubInstrument)
    - Specialized implementations for Diviner and Mini-RF sensors

Each subinstrument provides coordinate transformation and projection support for
boresight and FOV bounds in the context of lunar surface modeling.
"""

from typing import Optional, Union, Literal, Dict

import numpy as np
import spiceypy as spice


from src.SPICE.config import IMPLICIT_BORESIGHT, DIVINER_SUBINSTRUMENT_PIXEL_COUNT, LRO_MINIRF_FRAME_STR_ID


class SubInstrument:
    """
    Represents a SPICE-defined subinstrument using `getfov`.

    Each instance retrieves its geometry (boresight, bounds, frame, shape) from SPICE
    and provides methods to transform vectors and coordinates between frames.

    Attributes:
        _id (int | str): SPICE ID or name of the instrument.
        sub_instrument_frame (str): Native SPICE frame of the instrument.
        boresight (np.array): Instrument boresight vector in native frame.
        bounds (Optional[np.array]): Boundary vectors (FOV corners).
        shape (str): Shape of the FOV (e.g., RECTANGLE, CIRCLE).
    """

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
        """
        Compute and cache the transformation matrix to the target frame.

        Parameters:
            frame (str): Destination SPICE frame.
            et (float): Ephemeris time.

        Returns:
            np.ndarray: 3x3 rotation matrix from native to `frame`.
        """
        if self._ets.get(frame) != et:
            self._matrices[frame] = spice.pxform(self.sub_instrument_frame, frame, et)
            self._ets[frame] = et
        return self._matrices[frame]

    def transformed_boresight(self, frame: str, et: float = 0):
        """
        Return boresight vector transformed into target frame.

        Parameters:
            frame (str): Destination SPICE frame.
            et (float): Ephemeris time.

        Returns:
            np.ndarray: Transformed boresight vector.
        """
        matrix = self.transform_frame(frame, et)
        return spice.mxv(matrix, self.boresight)

    def transformed_bounds(self, frame: str, et: float = 0):
        """
        Return all bound vectors transformed into the target frame.

        Parameters:
            frame (str): Destination SPICE frame.
            et (float): Ephemeris time.

        Returns:
            np.ndarray: Array of transformed bounds.
        """
        matrix = self.transform_frame(frame, et)
        return np.array([spice.mxv(matrix, point) for point in self.bounds])

    def transform_vector(self, frame: str, vector: np.array, et: float = 0):
        """
        Transform an arbitrary vector into the specified frame.

        Parameters:
            frame (str): Destination frame.
            vector (np.array): 3D vector to transform.
            et (float): Ephemeris time.

        Returns:
            np.array: Transformed vector.
        """
        matrix = self.transform_frame(frame, et)
        return spice.mxv(matrix, vector)

    def transform_to_frame(self, frame: str, et: float = 0):
        """
        Overwrite this instrument’s internal state with a new frame.

        Modifies the boresight, bounds, and frame in-place.

        Parameters:
            frame (str): Destination frame.
            et (float): Ephemeris time.
        """
        self.boresight = self.transformed_boresight(frame, et=et)
        self.bounds = self.transformed_bounds(frame, et=et)
        self.sub_instrument_frame = frame


class ImplicitSubInstrument(SubInstrument):
    """
    A manually defined subinstrument without SPICE kernel support.

    Used when sensor geometry is sourced from mission documents (SIS),
    not from loaded SPICE kernels.

    Supports static frame-to-frame transformation using fixed matrix.
    """

    def __init__(
        self,
        _id: str,
        frame: str,
        boresight=IMPLICIT_BORESIGHT,
        bounds=None,
        transform_frame=None,
        shape: str = "POINT",
    ):
        """
        Create a subinstrument with manually defined geometry.

        Parameters:
            _id (str): Unique string identifier.
            frame (str): Native reference frame.
            boresight (np.array): Unit vector defining center line of sight.
            bounds (Optional[np.array]): Field-of-view boundary vectors.
            transform_frame (Optional[str]): Frame to transform from, if needed.
            shape (str): Shape descriptor for the FOV (e.g. RECTANGLE, POINT).
        """
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
    Specialized Diviner subinstrument representing one of 9 linear detectors.

    Each Diviner unit (A1–A6, B1–B3) contains 21 adjacent detector pixels.
    Pixels are represented as ImplicitSubInstrument instances.

    Attributes:
        pixels (List[ImplicitSubInstrument]): 21 pixel-level subinstruments.
    """

    def __init__(
        self, naif_id: int, _id: int, pixel_key: Literal["INS-85205_DETECTOR_DIRS_FP_A", "INS-85205_DETECTOR_DIRS_FP_B"]
    ):
        """
        Initialize a Diviner channel and its 21 pixels.

        Parameters:
            naif_id (int): SPICE ID for the instrument channel.
            _id (int): Index of the Diviner channel (0–8).
            pixel_key (str): SPICE kernel variable to fetch pixel directions.
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
    Specialized Mini-RF subinstrument using fixed geometry parameters.

    MiniRF operates in either S-band or X-band, with beam width documented in
    the MRFLRO_DP_SIS. Since it lacks proper SPICE kernels, geometry is defined
    manually and interpolated to simulate pixel-level behavior.

    Attributes:
        bounds (np.ndarray): Four corner directions (3D unit vectors).
        line_bounds (np.ndarray): Two vectors defining scan line edge.
        channel (str): 'S' or 'X'.
    """

    def __init__(self, channel: str):
        """
        Construct a Mini-RF subinstrument with configured beam shape.

        Parameters:
            channel (str): 'S' or 'X' to choose beam width profile.

        Raises:
            ValueError: If invalid channel is provided.
        """
        # Use a fixed boresight
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
            return np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ]
            )

        # rotation about Y
        def rot_y(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ]
            )

        # build the four corners: (+across,+along), (-across,+along), (-across,-along), (+across,-along)
        corners = []
        for sign_x, sign_y in [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]:
            # first rotate boresight by sign_x * half_across about Y,
            # then by sign_y * half_along about X
            R = rot_x(sign_y * half_along) @ rot_y(sign_x * half_across)
            corners.append(R @ boresight)

        bounds = np.vstack(corners)  # shape (4,3), already unit-length

        left_bound = rot_x(-half_along) @ boresight
        right_bound = rot_x(+half_along) @ boresight
        self.line_bounds = np.vstack([left_bound, right_bound])  # shape (2,3)

        # hand off to the base class
        super().__init__(f"minirf-{channel}", frame, boresight=boresight, bounds=bounds)

        self.channel = channel.upper()

    def look_vector_for_pixel(self, pixel_index: int, pixel_count: int) -> np.ndarray:
        """
        Linearly interpolate the look vector for a specific pixel.

        Parameters:
            pixel_index (int): Index of the pixel (0-based).
            pixel_count (int): Total number of pixels.

        Returns:
            np.ndarray: Interpolated unit look vector in instrument frame.
        """
        frac = (pixel_index + 0.5) / pixel_count
        vec = (1.0 - frac) * self.line_bounds[0] + frac * self.line_bounds[1]
        return vec / np.linalg.norm(vec)
