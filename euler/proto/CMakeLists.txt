protobuf_generate_cpp(worker.proto)
add_library(proto worker.pb.cc)
target_link_libraries(proto libprotobuf framework)
target_compile_options(proto PRIVATE -Wno-unused-parameter)
