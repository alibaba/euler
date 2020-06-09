add_definitions(-Wall -Wextra)

if (BUILD_COVERAGE)
  set(CMAKE_CXX_FLAGS "-g -O0 --coverage")
  set (CMAKE_EXE_LINKER_FLAGS "--coverage")
endif(BUILD_COVERAGE)

add_subdirectory(framework)
add_subdirectory(kernels)
add_subdirectory(dag)
add_subdirectory(dag_def)
add_subdirectory(index)
add_subdirectory(graph)
add_subdirectory(api)
