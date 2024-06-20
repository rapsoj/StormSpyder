#!/bin/bash

# Path to your virtual environment
VENV_PATH="myenv"

# Path to your deployment script
DEPLOY_SCRIPT="__main__.py"

# Maximum execution time in seconds (1 hour)
MAX_EXECUTION_TIME=3600

# Activate the virtual environment
source $VENV_PATH/bin/activate

# Run your deployment script with a timeout
timeout $MAX_EXECUTION_TIME python $DEPLOY_SCRIPT

# Deactivate the virtual environment
deactivate