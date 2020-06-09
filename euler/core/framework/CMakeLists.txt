protobuf_generate_cpp(dag_node.proto dag.proto tensor.proto tensor_shape.proto types.proto)

add_library(framework_proto
  dag_node.pb.cc
  dag.pb.cc
  tensor.pb.cc
  tensor_shape.pb.cc
  types.pb.cc)
target_link_libraries(framework_proto libprotobuf)
target_compile_options(framework_proto PRIVATE -Wno-unused-parameter)

add_library(framework
  tensor.cc
  tensor_util.cc
  allocator.cc
  executor.cc
  op_kernel.cc
  udf.cc)
target_link_libraries(framework framework_proto common)

add_executable(tensor_shape_test tensor_shape_test.cc)
target_link_libraries(tensor_shape_test ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main framework)
add_test(NAME tensor_shape_test COMMAND tensor_shape_test)

add_executable(tensor_test tensor_test.cc)
target_link_libraries(tensor_test ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main framework)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(tensor_util_test tensor_util_test.cc)
target_link_libraries(tensor_util_test ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main framework)
add_test(NAME tensor_util_test COMMAND tensor_util_test)
