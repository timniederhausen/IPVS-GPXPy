#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
module load gcc/13.2.0
module load cmake
# Activate spack environment
spack env activate gpxpy
# Set cmake command
export CMAKE_COMMAND=$(which cmake)
# Configure APEX
export APEX_SCREEN_OUTPUT=1
# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
rm -rf build_cpp && mkdir build_cpp && cd build_cpp
# Configure the project
$CMAKE_COMMAND ../core -DCMAKE_BUILD_TYPE=Release \
                  -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
                  -DCMAKE_C_COMPILER=$(which gcc) \
		  -DCMAKE_CXX_COMPILER=$(which g++) \
                  ${MKL_CONFIG}
 # Build the project
make -j all
make install
