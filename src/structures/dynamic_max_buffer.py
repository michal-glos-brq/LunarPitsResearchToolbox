import numpy as np

class DynamicMaxBuffer:
    """
    This class creates a dynamic, buffered maximum value.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size)
        self.index = 0
        self.maximum = float("-inf")
        self.maximum_ttl = buffer_size

    def add(self, value: float):
        self.buffer[self.index] = value
        if value >= self.maximum:
            # New max found – reset TTL
            self.maximum = value
            self.maximum_ttl = self.buffer_size
        else:
            # Not the new max – decrease TTL
            self.maximum_ttl -= 1
            if self.maximum_ttl == 0:
                # TTL expired – recompute max and reset TTL
                self.maximum = self.buffer.max()
                max_pos = np.argmax(self.buffer)
                distance = (max_pos - self.index) % self.buffer_size
                self.maximum_ttl = distance if distance != 0 else self.buffer_size
        self.index = (self.index + 1) % self.buffer_size

