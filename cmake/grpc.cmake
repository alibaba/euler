set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable benchmark testing" FORCE)
set(GRPC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc)

add_subdirectory(${PROTOBUF_ROOT_DIR} third_party/grpc)

# Set grpc package
set(GRPC_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc/include)

# Set protobuf package
set(PROTOBUF_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc/third_party/protobuf/src/)
