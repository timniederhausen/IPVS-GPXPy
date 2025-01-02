# Same as in the root CMakeLists.txt
set(MKL_INTERFACE_FULL "intel_lp64")
set(MKL_THREADING "sequential")

find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/GPRatTargets.cmake")
