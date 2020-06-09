add_library(ops SHARED
  common.cc
  sample_node_op.cc
  sample_n_with_types_op.cc
  sample_edge_op.cc
  get_node_op.cc
  get_edge_op.cc
  get_feature_op.cc
  get_node_type_op.cc
  get_neighbor_op.cc
  get_neighbor_edge_op.cc
  get_nb_filter_op.cc
  sample_neighbor_op.cc

  remote_op.cc
  as_op.cc
  post_process_op.cc
  sample_node_split_op.cc
  sample_n_with_types_split_op.cc
  sample_edge_split_op.cc
  append_merge_op.cc
  broad_cast_split_op.cc
  id_split_op.cc
  multi_type_idx_merge_op.cc
  multi_type_data_merge_op.cc
  idx_merge_op.cc
  idx_row_append_merge_op.cc
  data_merge_op.cc
  regular_data_merge_op.cc
  data_row_append_merge_op.cc
  get_edge_sum_weight_op.cc
  sample_root_op.cc
  local_sample_layer_op.cc
  reshape_op.cc
  sample_layer_op.cc
  sparse_gen_adj_op.cc
  sparse_get_adj_op.cc
  gather_result_op.cc
  id_unique_op.cc
  idx_gather_op.cc
  data_gather_op.cc
  sample_graph_label_op.cc
  get_graph_by_label_op.cc

  gp_unique_merge_op.cc

  min_udf.cc
  max_udf.cc
  mean_udf.cc
)

target_link_libraries(ops framework core index client)

add_executable(remote_op_test remote_op_test.cc)
target_link_libraries(remote_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME remote_op_test COMMAND remote_op_test)

add_executable(append_merge_op_test append_merge_op_test.cc)
target_link_libraries(append_merge_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME append_merge_op_test COMMAND append_merge_op_test)

add_executable(gp_unique_merge_op_test gp_unique_merge_op_test.cc)
target_link_libraries(gp_unique_merge_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME gp_unique_merge_op_test COMMAND gp_unique_merge_op_test)

add_executable(regular_data_merge_op_test regular_data_merge_op_test.cc)
target_link_libraries(regular_data_merge_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME regular_data_merge_op_test COMMAND regular_data_merge_op_test)

add_executable(pp_op_test post_process_op_test.cc)
target_link_libraries(pp_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME pp_op_test COMMAND pp_op_test)

add_executable(idx_merge_op_test idx_merge_op_test.cc)
target_link_libraries(idx_merge_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME idx_merge_op_test COMMAND idx_merge_op_test)

add_executable(data_merge_op_test data_merge_op_test.cc)
target_link_libraries(data_merge_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME data_merge_op_test COMMAND data_merge_op_test)

add_executable(layerwise_op_test layerwise_op_test.cc)
target_link_libraries(layerwise_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME layerwise_op_test COMMAND layerwise_op_test)

add_executable(get_nb_filter_op_test get_nb_filter_op_test.cc)
target_link_libraries(get_nb_filter_op_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME get_nb_filter_op_test COMMAND get_nb_filter_op_test)

add_executable(unique_gather_test unique_gather_test.cc)
target_link_libraries(unique_gather_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME unique_gather_test COMMAND unique_gather_test)

add_executable(multi_type_merge_test multi_type_merge_op_test.cc)
target_link_libraries(multi_type_merge_test ops gtest gtest_main grpc++_unsecure dag_def service api)
add_test(NAME multi_type_merge_test COMMAND multi_type_merge_test)

add_executable(ops_test ops_test.cc)
target_link_libraries(ops_test ops gtest gtest_main api proto)
add_test(NAME ops_test COMMAND ops_test)

add_executable(udf_test udf_test.cc)
target_link_libraries(udf_test gtest gtest_main framework)
add_test(NAME udf_test COMMAND udf_test)
