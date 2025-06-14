cmake_minimum_required(VERSION 3.21)

project(GPRatTests LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
  find_package(GPRat REQUIRED)
  enable_testing()
endif()

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0)

FetchContent_MakeAvailable(Catch2)

find_package(Boost REQUIRED)

# ---- Tests ----

add_executable(GPRat_test_output_correctness src/test_data.hpp
                                             src/output_correctness.cpp)
target_link_libraries(GPRat_test_output_correctness
                      PRIVATE GPRat::core Catch2::Catch2WithMain Boost::boost)
target_compile_features(GPRat_test_output_correctness PRIVATE cxx_std_17)

add_test(
  NAME GPRat_test_output_correctness
  COMMAND GPRat_test_output_correctness
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
