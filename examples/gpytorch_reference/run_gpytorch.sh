#!/bin/bash
# Input $1: Specify cpu/gpu

# Create & Activate python enviroment
if [ ! -d "gpytorch_env" ]; then
    python3 -m venv gpytorch_env
fi

# Activate enviroment
source gpytorch_env/bin/activate

# Install requirements
if ! python3 -c "import gpytorch"; then
    export TMPDIR="/scratch/$USER/tmp"
    pip3 install gpytorch
fi

# run on CPU or GPU
if [[ "$1" == "gpu" ]]
then
    # Execute the python script
    python3 execute.py --use-gpu
elif [[ "$1" == "cpu" ]]
then
    # Execute the python script
    python3 execute.py
else
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi
