add_subdirectory(testing)

add_library(client SHARED
            graph_config.cc
            grpc_channel.cc
            grpc_manager.cc
            rpc_client.cc
            rpc_manager.cc
            client_manager.cc
            query_proxy.cc
            query.cc)
target_link_libraries(client common framework core compiler dag index grpc++_unsecure)

add_executable(rpc_client_end2end_test rpc_client_end2end_test.cc)
target_link_libraries(rpc_client_end2end_test client client_testing_util gtest gtest_main)
add_test(NAME rpc_client_end2end_test COMMAND rpc_client_end2end_test)

add_executable(rpc_manager_test rpc_manager_test.cc)
target_link_libraries(rpc_manager_test client client_testing_util gmock gtest gtest_main)
add_test(NAME rpc_manager_test COMMAND rpc_manager_test)

add_executable(rpc_client_test rpc_client_test.cc)
target_link_libraries(rpc_client_test client client_testing_util gmock gtest gtest_main)
add_test(NAME rpc_client_test COMMAND rpc_client_test)

add_executable(grpc_channel_test grpc_channel_test.cc)
target_link_libraries(grpc_channel_test client client_testing_util gtest gtest_main)
add_test(NAME grpc_channel_test COMMAND grpc_channel_test)

add_executable(end2end_test end2end_test.cc)
target_link_libraries(end2end_test client ops compiler dag gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME end2end_test COMMAND end2end_test)

add_executable(end2end_gp_test end2end_gp_test.cc)
target_link_libraries(end2end_gp_test client ops compiler dag gtest gtest_main grpc++_unsecure dag_def service api)

add_executable(end2end_local_test end2end_local_test.cc)
target_link_libraries(end2end_local_test gtest_main euler_core)
add_test(NAME end2end_local_test COMMAND end2end_local_test)
