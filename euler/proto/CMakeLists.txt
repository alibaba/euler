protobuf_generate_grpc_cpp(graph_service.proto)

add_library(proto graph_service.pb.cc graph_service.grpc.pb.cc)
target_link_libraries(proto libprotobuf grpc++_unsecure)
target_compile_options(proto PRIVATE -Wno-unused-parameter)
