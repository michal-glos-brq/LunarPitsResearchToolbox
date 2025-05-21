"""
====================================================
Lunar Surface DSK Kernel Generator for SPICE
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

This module implements the generation of a high-resolution Digital Shape Kernel (DSK)
for the Moon’s surface using SPICE and external elevation data.

Core functionality:
    - Downloads a TIF DEM from USGS if needed.
    - Converts it to XYZ using GDAL CLI.
    - Interpolates and triangulates the 3D mesh.
    - Writes the output as a SPICE-compatible .dsk file.
    - Optionally pulls pre-built DSKs from BunnyCDN cache.

Intended for planetary surface modeling, sensor simulation, and navigation system testing.
"""

import os
import subprocess
import logging
import requests
from tqdm import tqdm
from urllib.parse import urljoin
from typing import Union

import pandas as pd
import numpy as np
import spiceypy as spice
import astropy.units as u
from astropy.coordinates import SphericalRepresentation
from filelock import FileLock

from src.global_config import LUNAR_FRAME, TQDM_NCOLS
from src.SPICE.kernel_utils.spice_kernels import BaseKernel
from src.SPICE.config import (
    LUNAR_TIF_DATA_URL,
    DSK_FILE_CENTER_BODY_ID,
    DSK_FILE_SURFACE_ID,
    BUNNY_PASSWORD,
    BUNNY_BASE_URL,
    BUNNY_STORAGE,
    FINSCL,
    CORSCL,
    WORKSZ,
    VOXPSZ,
    VOXLSZ,
    MAKVTL,
    SPXISZ,
    DCLASS,
    CORSYS,
    CORPAR,
    DEFAULT_TIF_SAMPLE_RATE,
    TIF_TO_KM_SCALE,
    DSK_KERNEL_LOCK_TIMEOUT,
    KERNEL_LOCK_POLL_INTERVAL,
    root_path,
    generic_url,
)
from src.global_config import LUNAR_FRAME, HDD_BASE_PATH, SUPRESS_TQDM

logger = logging.getLogger(__name__)


# We have to have same basic kernels loaded in order to create the DSK model
KERNELS = [
    BaseKernel("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls", root_path("lsk/naif0012.tls")),
    BaseKernel(generic_url("pck/pck00010.tpc"), root_path("pck/pck00010.tpc")),
    BaseKernel(generic_url("pck/moon_pa_de421_1900-2050.bpc"), root_path("pck/moon_pa_de421_1900-2050.bpc")),
    BaseKernel(generic_url("fk/satellites/moon_080317.tf"), root_path("fk/satellites/moon_080317.tf")),
    # Do wee need less quality DSK lunar model?
    BaseKernel(
        "https://naif.jpl.nasa.gov/pub/naif/pds/wgc/lessons/event_finding_kplo_old/kernels/dsk/moon_lowres.bds",
        root_path("moon_lowres.bds"),
    ),
]


if LUNAR_FRAME == "MOON_ME":
    KERNELS += [
        BaseKernel(generic_url("fk/satellites/moon_assoc_me.tf"), root_path("fk/moon_assoc_me.tf")),
        BaseKernel(generic_url("fk/satellites/moon_assoc_pa.tf"), root_path("fk/moon_assoc_pa.tf")),
    ]

elif LUNAR_FRAME == "MOON_PA_DE440":
    KERNELS += [
        BaseKernel(generic_url("fk/satellites/moon_de440_220930.tf"), root_path("fk/moon_de440_220930.tf")),
        BaseKernel(generic_url("pck/moon_pa_de440_200625.bpc"), root_path("pck/moon_pa_de440_200625.bpc")),
        BaseKernel(generic_url("spk/planets/de440.bsp"), root_path("spk/de440.bsp")),
    ]

else:
    raise ValueError("Unknown lunar frame specified.")


class DetailedModelDSKKernel(BaseKernel):
    """
    Generates or retrieves a detailed SPICE DSK kernel for the Moon’s surface.

    On instantiation:
        - Checks for local .dsk file.
        - If missing, checks BunnyCDN for pre-generated DSK.
        - If still missing, downloads TIF, converts to XYZ, generates triangles, and writes a new .dsk.

    Optionally downsamples based on `tif_sample_rate`.

    Attributes:
        filename (str): Output path for generated or retrieved DSK.
        tif_scale_percents (float): GDAL CLI downsampling percentage.
        base_filename (str): Path without file extension, used for auxiliary files (e.g., .xyz).
    """

    def __init__(self, filename: str, tif_sample_rate: float = DEFAULT_TIF_SAMPLE_RATE):
        """
        Initialize and generate or retrieve the DSK kernel file.

        Parameters:
            filename (str): Output filename for the .dsk kernel.
            tif_sample_rate (float): Sampling rate for TIF-to-XYZ conversion (0–1 scale).
        """
        self.filename = filename  # The DSK Kernel name
        self.base_filename = ".".join(filename.split(".")[:-1])
        # Convert sample rate into percents for gdal CLI use
        self.tif_scale_percents = tif_sample_rate * 100
        super().__init__(LUNAR_TIF_DATA_URL, filename)

        with FileLock(
            self.filename + ".lock", timeout=DSK_KERNEL_LOCK_TIMEOUT, poll_interval=KERNEL_LOCK_POLL_INTERVAL
        ):
            # If the DSK model does not exist
            if not self.file_exists:
                # First look whether the model is existing remotely
                if self.remote_dsk_exists:
                    self.remote_dsk_download()
                else:
                    if not self.xyz_file_exists:
                        if not self.tif_file_exists:
                            self.download_tif_file()
                        self.create_xyz_from_tif()

                    for kernel in KERNELS:
                        kernel.load()
                    self.create_dsk_from_xyz()
                    for kernel in KERNELS:
                        kernel.unload()

    @property
    def remote_dsk_exists(self):
        """
        Check whether the desired DSK exists in BunnyCDN cloud storage.

        Returns:
            bool: True if the file is found remotely, False otherwise.
        """
        url = urljoin(BUNNY_BASE_URL, f"{BUNNY_STORAGE}")
        response = requests.get(url, headers={"AccessKey": BUNNY_PASSWORD})
        if response.status_code != 200:
            logger.error("Failed to access BunnyCDN storage.")
            return False

        filenames = [remote_file["ObjectName"] for remote_file in response.json()]
        return os.path.basename(self.filename) in filenames

    @property
    def tif_filename(self):
        """
        Get local path to the original TIF elevation data file.

        Returns:
            str: Full path to cached TIF.
        """
        return os.path.join(HDD_BASE_PATH, LUNAR_TIF_DATA_URL.split("/")[-1])

    @property
    def tif_file_exists(self):
        """Check whether the TIF file is present locally."""
        return os.path.exists(self.tif_filename)

    @property
    def xyz_filename(self):
        """Get local path for the intermediate XYZ file (used for DSK conversion)."""
        return self.base_filename + ".xyz"

    @property
    def xyz_file_exists(self):
        """Check whether the XYZ file is present locally."""
        return os.path.exists(self.xyz_filename)

    def latlon_to_cartesian(self, lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> np.ndarray:
        """
        Convert lat/lon to 3D Cartesian coordinates using DSK ray-surface intersection.

        Parameters:
            lat (float or np.ndarray): Latitude(s) in degrees.
            lon (float or np.ndarray): Longitude(s) in degrees.

        Returns:
            np.ndarray: Shape (3,) or (N, 3) array of Cartesian coordinates in km.
        """
        return self.dsk_latlon_to_cartesian(lat, lon, self.filename)

    @staticmethod
    def dsk_latlon_to_cartesian(
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        dsk_path: str,
        alt_km: float = 10000.0,  # Good enough for Moon
    ) -> np.ndarray:
        """
        Static version of lat/lon to Cartesian intersection using an arbitrary DSK path.

        Parameters:
            lat (float or np.ndarray): Latitude(s) in degrees.
            lon (float or np.ndarray): Longitude(s) in degrees.
            dsk_path (str): Path to the DSK file.
            alt_km (float): Altitude of ray origin in km (default 10,000 km), used as virtual spacecraft for projection.

        Returns:
            np.ndarray: Shape (3,) or (N, 3) array of Cartesian coordinates in km.
        """
        lat = np.radians(lat)
        lon = np.radians(lon)

        if np.isscalar(lat):
            lat = np.array([lat])
            lon = np.array([lon])
            scalar_input = True
        else:
            scalar_input = False

        dsk_handle = spice.dasopr(dsk_path)
        dladsc = spice.dlabfs(dsk_handle)

        results = []
        for la, lo in zip(lat, lon):
            ray_dir = np.array(spice.latrec(1.0, lo, la))
            ray_start = ray_dir * alt_km
            _, spoint, found = spice.dskx02(dsk_handle, dladsc, ray_start, -ray_dir)
            results.append(spoint if found else [np.nan, np.nan, np.nan])

        spice.dascls(dsk_handle)
        return results[0] if scalar_input else np.array(results)

    def remote_dsk_download(self):
        """
        Download pre-generated DSK file from BunnyCDN storage.

        Raises:
            HTTPError: If the remote request fails.
        """
        url = urljoin(BUNNY_BASE_URL, f"{BUNNY_STORAGE}")
        url = urljoin(url, f"{self.filename.split('/')[-1]}")
        with requests.get(url, headers={"AccessKey": BUNNY_PASSWORD}, stream=True) as r:
            r.raise_for_status()  # Ensure request is successful
            total_size = int(r.headers.get("content-length", 0))  # Get file size
            block_size = 1024 * 1024  # 1 MB chunks

            with (
                open(self.filename, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    ncols=TQDM_NCOLS,
                    desc=f"Downloading DSK from BunnyCDN",
                    miniters=1,  # Ensures frequent updates
                    disable=SUPRESS_TQDM,
                ) as t,
            ):
                for chunk in r.iter_content(block_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        t.update(len(chunk))  # Update progress bar

    def download_tif_file(self):
        """
        Download the high-resolution elevation TIF file from USGS to local cache.

        Raises:
            HTTPError: If the request fails.
        """
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024**2  # 1 MB
            with (
                open(self.tif_filename, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    ncols=TQDM_NCOLS,
                    desc="Downloading TIF file",
                    miniters=1,  # Ensure the progress bar updates frequently
                    disable=SUPRESS_TQDM,
                ) as t,
            ):
                for chunk in r.iter_content(block_size):
                    if chunk:  # Filter out keep-alive new chunks
                        f.write(chunk)
                        t.update(len(chunk))

    def create_xyz_from_tif(self):
        """
        Convert the downloaded TIF to XYZ format using GDAL CLI with specified downsampling.

        Raises:
            RuntimeError: If the GDAL CLI fails to produce output.
        """
        # This section here requires tweaking for bodies different then the Moon
        command = [
            "gdal_translate",
            "-of",
            "XYZ",
            "-scale",
            "-32768",
            "32767",
            "-1737.4",
            "1737.4",
            "-outsize",
            f"{self.tif_scale_percents:.2}%",
            f"{self.tif_scale_percents:.2}%",
            "-a_nodata",
            "-32768",
            str(self.tif_filename),
            str(self.xyz_filename),
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert TIF to XYZ: {result.stderr}")

    def create_dsk_from_xyz(self):
        """
        Process the XYZ file to generate a triangulated mesh and write a DSK segment.

        Steps:
            - Convert XYZ to spherical coordinates.
            - Apply cosine-weighted downsampling along latitudes.
            - Generate triangle mesh.
            - Compute voxel spatial index.
            - Write to DSK via SPICE's `dskw02`.

        Raises:
            AssertionError: If grid shape is inconsistent.
        """
        df = pd.read_csv(self.xyz_filename, sep=" ", names=["x", "y", "z"])
        # Extract coordinates
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = (df["z"] / TIF_TO_KM_SCALE).to_numpy().astype(np.float32)  # Convert elevation to kilometers

        lon = np.interp(x, (x.min(), x.max()), (-180, 180)) * u.deg
        lat = (90 - (y - y.min()) / (y.max() - y.min()) * 180) * u.deg
        lat = -lat  # Invert latitude

        # Compute the 3D radius and convert to Cartesian
        scalar_radius = spice.bodvrd("MOON", "RADII", 3)[1][0]
        radius = (scalar_radius + z) * u.m
        spherical_coords = SphericalRepresentation(lon, lat, radius)
        cartesian_coords = spherical_coords.to_cartesian()

        batch_rects = np.column_stack(
            [cartesian_coords.x.value, cartesian_coords.y.value, cartesian_coords.z.value]
        ).astype(np.float32)

        # --- Reshape grid and downsample based on latitude ---
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        num_cols = len(unique_x)
        num_rows = len(unique_y)
        assert num_rows * num_cols == len(x), "Data does not form a complete grid."

        # Reshape to 2D grid (rows = latitudes, cols = longitudes)
        vertices_grid = batch_rects.reshape(num_rows, num_cols, 3)
        lon_grid = lon.value.reshape(num_rows, num_cols)
        lat_grid = lat.value.reshape(num_rows, num_cols)

        global_vertices_list = []  # List of downsampled row vertex arrays.
        row_data = []  # List of dicts: {'indices': global indices, 'lons': sampled longitudes}
        global_index = 0

        for i in range(num_rows):
            row_lat = lat_grid[i, 0]
            factor = np.cos(np.deg2rad(abs(row_lat)))
            desired_count = max(1, int(np.round(num_cols * factor)))
            col_indices = np.linspace(0, num_cols - 1, desired_count, dtype=int)

            row_vertices = vertices_grid[i, col_indices, :]
            row_lons = lon_grid[i, col_indices]
            indices = np.arange(global_index, global_index + len(col_indices))
            global_index += len(col_indices)

            global_vertices_list.append(row_vertices)
            row_data.append({"indices": indices, "lons": row_lons})

        vertices_global = np.concatenate(global_vertices_list, axis=0)

        # --- Stitch adjacent rows into triangles ---
        def stitch_rows(idxA, idxB, lonA, lonB):
            """Merge two rows of vertices into triangles."""
            tris = []
            i, j = 0, 0
            while i < len(idxA) - 1 and j < len(idxB) - 1:
                diffA = abs(lonA[i + 1] - lonB[j])
                diffB = abs(lonB[j + 1] - lonA[i])
                if diffA < diffB:
                    tris.append([idxA[i], idxB[j], idxA[i + 1]])
                    i += 1
                else:
                    tris.append([idxA[i], idxB[j], idxB[j + 1]])
                    j += 1
            while i < len(idxA) - 1:
                tris.append([idxA[i], idxB[-1], idxA[i + 1]])
                i += 1
            while j < len(idxB) - 1:
                tris.append([idxA[-1], idxB[j], idxB[j + 1]])
                j += 1
            return tris

        triangles = []
        for r in range(len(row_data) - 1):
            idxA = row_data[r]["indices"]
            idxB = row_data[r + 1]["indices"]
            lonA = row_data[r]["lons"]
            lonB = row_data[r + 1]["lons"]

            if len(idxA) == 1:
                for j in range(len(idxB) - 1):
                    triangles.append([idxA[0], idxB[j], idxB[j + 1]])
            elif len(idxB) == 1:
                for i in range(len(idxA) - 1):
                    triangles.append([idxA[i], idxB[0], idxA[i + 1]])
            else:
                triangles.extend(stitch_rows(idxA, idxB, lonA, lonB))

        vertices = np.array(vertices_global)
        triangles = np.array([[x + 1, y + 1, z + 1] for x, y, z in triangles])

        # --- Open DSK File for Writing ---
        if os.path.exists(self.filename):
            os.remove(self.filename)
            logger.info("Existing DSK file removed to avoid conflicts.")

        mncor1, mxcor1 = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        mncor2, mxcor2 = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        mncor3, mxcor3 = np.min(vertices[:, 2]), np.max(vertices[:, 2])

        logger.info("Adjusted Bounds:")
        logger.info("  X: %s to %s", mncor1, mxcor1)
        logger.info("  Y: %s to %s", mncor2, mxcor2)
        logger.info("  Z: %s to %s", mncor3, mxcor3)

        NVXTOT = ((mxcor1 - mncor1) / FINSCL) * ((mxcor2 - mncor2) / FINSCL) * ((mxcor3 - mncor3) / FINSCL)
        logger.info("Estimated Fine Voxel Count (NVXTOT): %s", NVXTOT)

        handle = spice.dskopn(self.filename, "Generated DSK File", 0)
        logger.info("Opened DSK file: %s", self.filename)

        # --- Compute Spatial Index ---
        logger.info("Computing spatial index...")
        spaixd, spaixi = spice.dskmi2(vertices, triangles, FINSCL, CORSCL, WORKSZ, VOXPSZ, VOXLSZ, MAKVTL, SPXISZ)
        logger.info("Spatial index computed.")

        # --- Write DSK Segment ---
        logger.info("Writing DSK file...")
        time_start = -1e9  # Arbitrary large negative time (valid for all time)
        time_end = 1e9  # Arbitrary large positive time

        spice.dskw02(
            handle,
            DSK_FILE_CENTER_BODY_ID,
            DSK_FILE_SURFACE_ID,
            DCLASS,
            LUNAR_FRAME,
            CORSYS,
            CORPAR,
            mncor1,
            mxcor1,
            mncor2,
            mxcor2,
            mncor3,
            mxcor3,
            time_start,
            time_end,
            vertices,
            triangles,
            spaixd,
            spaixi,
        )
        logger.info("DSK segment written to %s", self.filename)

        # --- Close DSK File ---
        spice.dskcls(handle, True)
        logger.info("DSK file closed with compression enabled.")
