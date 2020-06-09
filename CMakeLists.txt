# Minimum CMake required
cmake_minimum_required(VERSION 3.8)

project(Euler CXX)

enable_testing()

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -O3 -fno-omit-frame-pointer)

option(USE_JEMALLOC "Option for using jemalloc" ON)
option(BUILD_COVERAGE "Build coverage system" OFF)
option(USE_OPENMP "Option for using open mp" OFF)

# Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
# for targets that link ${CMAKE_THREAD_LIBS_INIT}.
find_package (Threads)

if (USE_OPENMP)
  add_definitions(-DOPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif (USE_OPENMP)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add external module
include(grpc)
include(googletest)
include(zookeeper)
include(utils)
include(linenoise)

if (USE_JEMALLOC)
  include(jemalloc)
endif(USE_JEMALLOC)

# Add include directories

message(STATUS ${CUDA_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${linenoise_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CUDA_INCLUDE_DIRS}
                    )

include_directories(SYSTEM
                    ${PROTOBUF_INCLUDE_DIRS}
                    ${GRPC_INCLUDE_DIRS}
                    ${ZOOKEEPER_INCLUDE_DIRS})

add_subdirectory(tf_euler)
add_subdirectory(euler)
add_subdirectory(tools)

# Build as a whole shared object
include(euler_core)
