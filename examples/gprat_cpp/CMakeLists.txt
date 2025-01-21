cmake_minimum_required(VERSION 3.16)

project(example_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HPX
find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

# Include directories
include_directories(${HPX_INCLUDE_DIR})
include_directories(${MKL_INCLUDE_DIRS})

# Determine the path to the include directory
get_filename_component(INCLUDE_DIR "install/include" ABSOLUTE)
# Add the include directory
include_directories(${INCLUDE_DIR})

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Determine the path to the library directory relative to the current
# CMakeLists.txt file
get_filename_component(LIB_DIR "install/lib" ABSOLUTE)
# Find the Gaussian Process library
find_library(GPRAT_LIB gprat_core HINTS ${LIB_DIR} REQUIRED)
message(STATUS "GPRat core library found: ${GPRAT_LIB}")

# Add the executable
add_executable(gprat_cpp src/execute.cpp)
# Link the libraries
target_link_libraries(
  gprat_cpp PUBLIC "${GPRAT_LIB}" HPX::hpx MKL::mkl_intel_lp64 MKL::mkl_core
                   MKL::MKL MKL::mkl_sequential)
