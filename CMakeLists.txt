message(STATUS "Building Python language bindings for GPXPy.")

cmake_minimum_required(VERSION 3.16)

project(gpxpy)

include(CMakeDependentOption)
include(cmake/CPM.cmake)

# What to build?
option(GPXPY_BUILD_CORE "Build the core library" ON)
cmake_dependent_option(GPXPY_BUILD_BINDINGS "Build the Python bindings" ON
                       "GPXPY_BUILD_CORE" OFF)

option(GPXPY_ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

if(GPXPY_ENABLE_FORMAT_TARGETS)
  cpmaddpackage("gh:TheLartians/Format.cmake@1.8.1")
endif()

if(GPXPY_BUILD_CORE)
  add_subdirectory(core)
  if(GPXPY_BUILD_BINDINGS)
    add_subdirectory(bindings)
  endif()
endif()
