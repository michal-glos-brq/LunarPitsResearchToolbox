#!/bin/bash

curl -s https://install.zerotier.com | bash

# Start the ZeroTier daemon in the background
zerotier-one -d &

# Wait briefly to ensure the daemon initializes
sleep 5

# Join the ZeroTier network using the provided network ID
zerotier-cli join "$NETWORK_ID"
