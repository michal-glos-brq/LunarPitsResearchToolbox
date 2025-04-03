

# This is DIVINER sampling rate and poses as a reasonable base timestep for the simulation
SIMULATION_STEP = 1.024

# To be really inclusive, treshold is multiplied by this values
TOLERANCE_MARGIN = 1.1


# Buffered max values for different parameters
DYNAMIC_MAX_BUFFER_HEIGHT_SIZE = 128
DYNAMIC_MAX_BUFFER_HEIGHT_UPDATE_RATE = 8

DYNAMIC_MAX_BUFFER_FOV_WIDTH_SIZE = 16
DYNAMIC_MAX_BUFFER_FOV_WIDTH_UPDATE_RATE = 128

DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_SIZE = 64
DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_UPDATE_RATE = 64

# Batching simulation points for efficient inserts
MONGO_PUSH_BATCH_SIZE = 256

# Printing the simulation progress in somehow human friendly way
SIM_STATE_DUMP_INTERVAL = 60 * 15 # Each 15 minutes

