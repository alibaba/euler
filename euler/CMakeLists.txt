add_definitions(-Wall -Wextra)

if (BUILD_COVERAGE)
  set(CMAKE_CXX_FLAGS "-g -O0 --coverage")
  set (CMAKE_EXE_LINKER_FLAGS "--coverage")
endif(BUILD_COVERAGE)

# Add subdirectories
add_subdirectory(client)
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(proto)
add_subdirectory(service)
