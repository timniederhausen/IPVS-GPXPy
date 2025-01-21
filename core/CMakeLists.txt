add_library(
  gprat_core STATIC
  src/gp_optimizer.cpp
  src/gp_functions.cpp
  src/gp_algorithms_mkl.cpp
  src/gprat_c.cpp
  src/adapter_mkl.cpp
  src/tiled_algorithms_cpu.cpp
  src/gp_uncertainty.cpp
  src/utils_c.cpp)

add_library(GPRat::core ALIAS gprat_core)

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp)
target_sources(gprat_core PRIVATE ${header_files})

# Link HPX libraries
target_link_libraries(
  gprat_core PUBLIC HPX::hpx MKL::mkl_intel_lp64 MKL::mkl_core MKL::MKL
                    MKL::mkl_sequential)

# Include directories
target_include_directories(
  gprat_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

target_compile_features(gprat_core PUBLIC cxx_std_17)

set_property(TARGET gprat_core PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS gprat_core DESTINATION "${CMAKE_INSTALL_PREFIX}/install/lib")

install(DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_PREFIX}/install/include")
