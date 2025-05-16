"""
============================================================
Pipeline Configuration
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""


from src.global_config import MASTER_IP

REDIS_CONNECTION_STRING = f"redis://{MASTER_IP}:6379/0"
