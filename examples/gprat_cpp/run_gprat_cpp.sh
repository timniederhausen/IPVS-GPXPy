#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
spack load gcc@14.2.0
spack load cmake

# Activate environment
spack env activate gprat_cpu_gcc

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build
# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPRat_DIR=./lib/cmake/GPRat
 # Build the project
make -j

################################################################################
# Run code
################################################################################
./gprat_cpp
