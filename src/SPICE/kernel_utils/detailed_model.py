"""
====================================================
Lunar Surface DSK Kernel Generator for SPICE
====================================================

Author: Michal Gloš
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

Description:
------------
This module implements the generation of a Digital Shape Kernel (DSK)
for the lunar surface using SPICE. The process includes:

1. Downloading high-resolution lunar elevation data from USGS.
2. Converting TIFF elevation data to XYZ format using GDAL.
3. Transforming the XYZ data into a triangulated mesh for DSK.
4. Writing the DSK file with SPICE.

The generated kernel allows precise modeling of the Moon's surface
for spacecraft navigation, simulations, and planetary science research.

Dependencies:
-------------
- SPICE (spiceypy)
- GDAL (for TIFF to XYZ conversion)
- Pandas, NumPy, Astropy

Usage:
------
This class should be instantiated with the desired output filename.
If the file does not exist, the required processing steps will be
executed automatically.

Example:
--------
    kernel = DetailedModelDSKKernel("moon_surface.dsk")
    # The file "moon_surface.dsk" will be created if not found.

"""

import os
import subprocess
import logging
import requests
from tqdm import tqdm

import pandas as pd
import numpy as np

import spiceypy as spice
import astropy.units as u
from astropy.coordinates import SphericalRepresentation

from src.global_config import LUNAR_FRAME, TQDM_NCOLS
from src.SPICE.kernel_utils.spice_utils import BaseKernel
from src.SPICE.kernel_utils.kernel_management import LunarKernelManager
from src.SPICE.config import LUNAR_TIF_DATA_URL, DSK_FILE_CENTER_BODY_ID, DSK_FILE_SURFACE_ID, FINSCL, CORSCL, WORKSZ, VOXPSZ, VOXLSZ, MAKVTL, SPXISZ, DCLASS, LUNAR_FRAME, CORSYS, CORPAR

logger = logging.getLogger(__name__)


class DetailedModelDSKKernel(BaseKernel):

    def __init__(self, filename: str, tif_sample_rate: float = 0.075):
        """
        This kernel is very specific - filename is the desired .dsk file, but will be caching files with different
        suffixes. Url will be location of the source tiff data

        If DSK file is not found locally, processing begins with kernel instantiation.
        """
        self.filename = filename
        self.base_filename = ".".join(filename.split(".")[:-1])
        self.tif_scale_percents = tif_sample_rate * 100
        super().__init__(filename, LUNAR_TIF_DATA_URL)

        if not self.file_exists:
            if not self.xyz_file_exists:
                if not self.tif_file_exists:
                    self.download_tif_file()
                self.create_xyz_from_tif()

            lunar_kernel_manager = LunarKernelManager(
                LUNAR_FRAME, detailed=False
            )  # We now work on the more detailed version
            lunar_kernel_manager.load_static_kernels()  # No dynamic kernels, no need to obtain arbitrary time to load them
            self.create_dsk_from_xyz()
            lunar_kernel_manager.unload_all()
            del lunar_kernel_manager

    @property
    def tif_filename(self):
        return self.base_filename + ".tif"

    @property
    def tif_file_exists(self):
        return os.path.exists(self.tif_filename)

    @property
    def xyz_filename(self):
        return self.base_filename + ".xyz"

    @property
    def xyz_file_exists(self):
        return os.path.exists(self.xyz_filename)

    def download_tif_file(self):
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            t = tqdm(total=total_size, unit="B", unit_scale=True, ncols=TQDM_NCOLS, desc="Downloading TIF file")
            with open(self.tif_filename, "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()

    def create_xyz_from_tif(self):
        command = [
            "gdal_translate", "-of", "XYZ",
            "-scale", "-32768", "32767", "-1737.4", "1737.4",
            "-outsize", f"{self.tif_scale_percents:.2}%", f"{self.tif_scale_percents:.2}%",
            "-a_nodata", "-32768", str(self.tif_filename), str(self.xyz_filename)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    
        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert TIF to XYZ: {result.stderr}")

    def create_dsk_from_xyz(self):
        df = pd.read_csv(self.xyz_filename, sep=" ", names=["x", "y", "z"])
        # Extract coordinates
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = (df["z"] / 100).to_numpy().astype(np.float32)  # Convert elevation to meters

        lon = np.interp(x, (x.min(), x.max()), (-180, 180)) * u.deg
        lat = (90 - (y - y.min()) / (y.max() - y.min()) * 180) * u.deg
        lat = -lat  # Invert latitude

        # Compute the 3D radius and convert to Cartesian
        scalar_radius = spice.bodvrd("MOON", "RADII", 3)[1][0] * 1000
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
