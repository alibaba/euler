add_library(core STATIC node.cc edge.cc compact_graph.cc compact_node.cc compact_edge.cc
            fast_node.cc fast_edge.cc fast_graph.cc fast_graph_factory.cc
            compact_graph_factory.cc graph_engine.cc graph_builder.cc)
target_link_libraries(core common)

add_executable(local_graph_test local_graph_test.cc)
add_custom_command(TARGET local_graph_test
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/testdata/* ${CMAKE_CURRENT_BINARY_DIR}/
                   COMMENT "Copying test data to build directory")

target_link_libraries(local_graph_test
                      ${cmake_thread_libs_init} gtest gtest_main glog core)

add_test(NAME local_graph_test COMMAND local_graph_test)
