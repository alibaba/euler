add_library(core SHARED node.cc edge.cc graph.cc graph_builder.cc graph_meta.cc)
target_link_libraries(core common)

add_executable(node_test node_test.cc)
target_link_libraries(node_test ${cmake_thread_libs_init} gtest gtest_main core)
add_test(NAME node_test COMMAND node_test)

add_executable(edge_test edge_test.cc)
target_link_libraries(edge_test ${cmake_thread_libs_init} gtest gtest_main core)
add_test(NAME edge_test COMMAND edge_test)

add_executable(graph_test graph_test.cc)
target_link_libraries(graph_test ${cmake_thread_libs_init} gtest gtest_main core)
add_test(NAME graph_test COMMAND graph_test)
