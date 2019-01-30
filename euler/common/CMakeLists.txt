add_definitions(-DTHREADED) # Enable Zookeeper Sync API.
set (COMMON_FILE alias_method.cc random.cc timmer.cc bytes_reader.cc file_io_factory.cc local_file_io.cc
  net_util.cc server_monitor.cc string_util.cc zk_server_monitor.cc zk_server_register.cc zk_service.cc)
if (USE_HDFS)
  add_definitions(-DHDFS) # Enable hdfs
  add_library(common STATIC ${COMMON_FILE} hdfs_file_io.cc)
  target_link_libraries(common zookeeper hdfs glog)
else ()
  add_library(common STATIC ${COMMON_FILE})
  target_link_libraries(common zookeeper glog)
endif (USE_HDFS)

add_executable(local_file_io_test random.cc bytes_reader.cc local_file_io.cc local_file_io_test.cc)
target_link_libraries(local_file_io_test
                      ${CMAKE_THREAD_LIBS_INIT} common gtest gtest_main glog)

add_executable(compact_weighted_collection_test random.cc compact_weighted_collection_test.cc)
target_link_libraries(compact_weighted_collection_test
                      ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main glog)

add_executable(fast_weighted_collection_test random.cc fast_weighted_collection_test.cc alias_method.cc)
target_link_libraries(fast_weighted_collection_test
                      ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main glog)

add_executable(net_util_test net_util_test.cc net_util.cc)
target_link_libraries(net_util_test
                      ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main glog)

add_executable(server_monitor_test server_monitor_test.cc)
target_link_libraries(server_monitor_test common
                      glog gtest gtest_main zookeeper)

add_test(NAME local_file_io_test COMMAND local_file_io_test)
add_test(NAME compact_weighted_collection_test COMMAND compact_weighted_collection_test)
add_test(NAME fast_weighted_collection_test COMMAND fast_weighted_collection_test)
add_test(NAME net_util_test COMMAND net_util_test)
add_test(NAME server_monitor_test COMMAND server_monitor_test)
