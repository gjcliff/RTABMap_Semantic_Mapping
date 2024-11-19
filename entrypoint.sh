#!/bin/bash

# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# Execute the passed command (arguments are passed via "$@")
exec python3 scripts/semantic_mapping.py "$@"
