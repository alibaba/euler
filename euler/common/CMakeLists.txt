add_definitions(-DTHREADED) # Enable Zookeeper Sync API.

protobuf_generate_cpp(server_meta.proto)
add_library(server_meta server_meta.pb.cc)
target_link_libraries(server_meta libprotobuf)
target_compile_options(server_meta PRIVATE -Wno-unused-parameter)

add_library(common SHARED
  alias_method.cc
  data_types.cc
  env.cc
  random.cc
  timmer.cc
  bytes_io.cc
  file_io.cc
  local_file_io.cc
  hdfs_file_io.cc
  logging.cc
  str_util.cc
  slice.cc
  status.cc
  env_posix.cc
  server_monitor.cc
  zk_server_monitor.cc
  zk_server_register.cc
  net_util.cc
  hash.cc)
target_link_libraries(common server_meta zookeeper)

add_executable(local_file_io_test local_file_io_test.cc)
target_link_libraries(local_file_io_test ${CMAKE_THREAD_LIBS_INIT} common gtest gtest_main)
add_test(NAME local_file_io_test COMMAND local_file_io_test)

add_executable(hdfs_file_io_test hdfs_file_io_test.cc)
target_link_libraries(hdfs_file_io_test ${CMAKE_THREAD_LIBS_INIT} common gtest gtest_main)

add_executable(bytes_io_test bytes_io_test.cc)
target_link_libraries(bytes_io_test ${CMAKE_THREAD_LIBS_INIT} common gtest gtest_main)
add_test(NAME bytes_io_test COMMAND bytes_io_test)

add_executable(compact_weighted_collection_test compact_weighted_collection_test.cc)
target_link_libraries(compact_weighted_collection_test ${CMAKE_THREAD_LIBS_INIT} common gtest gtest_main)
add_test(NAME compact_weighted_collection_test COMMAND compact_weighted_collection_test)

add_executable(fast_weighted_collection_test fast_weighted_collection_test.cc)
target_link_libraries(fast_weighted_collection_test ${CMAKE_THREAD_LIBS_INIT} common gtest gtest_main)
add_test(NAME fast_weighted_collection_test COMMAND fast_weighted_collection_test)

add_executable(env_test env_test.cc)
target_link_libraries(env_test common gtest gtest_main)
add_test(NAME env_test COMMAND env_test)

add_executable(str_util_test str_util_test.cc)
target_link_libraries(str_util_test common gtest gtest_main)
add_test(NAME str_util_test COMMAND str_util_test)

add_executable(data_types_test data_types_test.cc)
target_link_libraries(data_types_test common gtest gtest_main)
add_test(NAME data_types_test COMMAND data_types_test)
