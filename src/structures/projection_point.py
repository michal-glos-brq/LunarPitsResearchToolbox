"""
============================================================
Dataclass for instrument projection onto a body surface
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectionPoint:
    et: float
    projection: np.array
    projection_trgepc: float
    spacecraft_relative: np.array

    def to_data(self):
        return {
            "cx_projected": self.projection[0],
            "cy_projected": self.projection[1],
            "cz_projected": self.projection[2],
            "trgepc": self.projection_trgepc,
            "sc_pos_x" : self.spacecraft_relative[0] + self.projection[0],
            "sc_pos_y" : self.spacecraft_relative[1] + self.projection[1],
            "sc_pos_z" : self.spacecraft_relative[2] + self.projection[2],
        }
