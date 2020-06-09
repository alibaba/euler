add_library(dag_def dag_node_def.cc dag_def.cc sub_graph_iso.cc)
target_link_libraries(dag_def dag)

add_executable(dag_def_test dag_def_test.cc)
target_link_libraries(dag_def_test dag_def gtest gtest_main)
add_test(NAME dag_def_test COMMAND dag_def_test)
