add_library(dag dag.cc node.cc)
target_link_libraries(dag common framework)

add_executable(dag_test dag_test.cc)
target_link_libraries(dag_test ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main dag)
add_test(NAME dag_test COMMAND dag_test)
