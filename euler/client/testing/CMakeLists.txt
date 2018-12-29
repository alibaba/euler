protobuf_generate_grpc_cpp(echo.proto)

add_library(client_testing_util echo.pb.cc echo.grpc.pb.cc mock_rpc_manager.cc)
target_link_libraries(client_testing_util libprotobuf grpc++_unsecure)
target_compile_options(client_testing_util PRIVATE -Wno-unused-parameter -Wno-inconsistent-missing-override)
