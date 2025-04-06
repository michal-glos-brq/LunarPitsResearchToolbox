import os

MASTER_IP = os.getenv("MASTER_IP", "host.docker.internal")
REDIS_CONNECTION_STRING = f"redis://{MASTER_IP}:6379/0"
