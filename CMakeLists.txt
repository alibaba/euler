# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

project(GraphServer CXX)

enable_testing()

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -O3 -fno-omit-frame-pointer)

option(USE_JEMALLOC "Option for using jemalloc" ON)
option(BUILD_COVERAGE "Build coverage system" OFF)
option(USE_HDFS "Option for using hdfs" OFF)

# Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
# for targets that link ${CMAKE_THREAD_LIBS_INIT}.
find_package (Threads)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add external module
include(grpc)
include(googletest)
include(glog)
include(zookeeper)
include(utils)
include(linenoise)

if (USE_JEMALLOC)
  include(jemalloc)
endif(USE_JEMALLOC)

# Add include directories
include_directories(${GTEST_INCLUDE_DIRS}
                    ${GLOG_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${linenoise_INCLUDE_DIRS}
                    .)
include_directories(SYSTEM
                    ${PROTOBUF_INCLUDE_DIRS}
                    ${GRPC_INCLUDE_DIRS}
                    ${ZOOKEEPER_INCLUDE_DIRS})

add_subdirectory(euler)
add_subdirectory(tf_euler)
add_subdirectory(tools)
