add_subdirectory(testing)

add_library(client SHARED
            graph.cc
            graph_config.cc
            grpc_channel.cc
            grpc_manager.cc
            local_graph.cc
            remote_graph.cc
            remote_graph_shard.cc
            rpc_client.cc
            rpc_manager.cc
            status.cc)
target_link_libraries(client common core proto glog jemalloc_STATIC_PIC)

add_executable(graph_test graph_test.cc)
target_link_libraries(graph_test client
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main)

add_custom_command(TARGET graph_test
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/testdata
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/testdata ${CMAKE_CURRENT_BINARY_DIR}/testdata
                   COMMENT "Copying test data to build directory")

add_executable(grpc_channel_test grpc_channel_test.cc)
target_link_libraries(grpc_channel_test client client_testing_util
                      gtest gtest_main)

add_executable(rpc_manager_test rpc_manager_test.cc)
target_link_libraries(rpc_manager_test client client_testing_util
                      gmock gtest gtest_main)

add_executable(rpc_client_test rpc_client_test.cc)
target_link_libraries(rpc_client_test client client_testing_util
                      gmock gtest gtest_main)

add_executable(rpc_client_end2end_test rpc_client_end2end_test.cc)
target_link_libraries(rpc_client_end2end_test client client_testing_util
                      gtest gtest_main)

target_compile_options(grpc_channel_test PRIVATE -Wno-unused-parameter -Wno-inconsistent-missing-override)
target_compile_options(rpc_manager_test PRIVATE -Wno-unused-parameter -Wno-inconsistent-missing-override)
target_compile_options(rpc_client_test PRIVATE -Wno-unused-parameter -Wno-inconsistent-missing-override)
target_compile_options(rpc_client_end2end_test PRIVATE -Wno-unused-parameter -Wno-inconsistent-missing-override)

add_test(NAME graph_test COMMAND graph_test)
add_test(NAME grpc_channel_test COMMAND grpc_channel_test)
add_test(NAME rpc_manager_test COMMAND rpc_manager_test)
add_test(NAME rpc_client_test COMMAND rpc_client_test)
add_test(NAME rpc_client_end2end_test COMMAND rpc_client_end2end_test)
