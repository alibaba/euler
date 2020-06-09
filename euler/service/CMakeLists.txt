add_library(service SHARED
  server_interface.cc
  grpc_server.cc
  grpc_euler_service.cc
  grpc_worker_service.cc
  grpc_worker.cc
  python_api.cc)

target_link_libraries(service api grpc++_unsecure dag common proto jemalloc_STATIC_PIC ${CMAKE_THREAD_LIBS_INIT})

add_library(mock_api SHARED mock_api.cc)

add_executable(grpc_server_test grpc_server_test.cc)
target_link_libraries(grpc_server_test service ops mock_api gtest gtest_main grpc++_unsecure)
add_test(NAME grpc_server_test COMMAND grpc_server_test)
