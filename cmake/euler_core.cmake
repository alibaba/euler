add_definitions(-DTHREADED) # Enable Zookeeper Sync API.

protobuf_generate_cpp(
  euler/common/server_meta.proto
  euler/core/framework/dag_node.proto
  euler/core/framework/dag.proto
  euler/core/framework/tensor.proto
  euler/core/framework/tensor_shape.proto
  euler/core/framework/types.proto
  euler/proto/worker.proto)

SET_SOURCE_FILES_PROPERTIES(euler/parser/lex.yy.c euler/parser/gremlin.tab.c PROPERTIES LANGUAGE CXX)

add_library(euler_core SHARED
  euler/common/server_meta.pb.cc
  euler/core/framework/dag_node.pb.cc
  euler/core/framework/dag.pb.cc
  euler/core/framework/tensor.pb.cc
  euler/core/framework/tensor_shape.pb.cc
  euler/core/framework/types.pb.cc
  euler/proto/worker.pb.cc

  euler/parser/lex.yy.c
  euler/parser/gremlin.tab.c

  euler/common/zk_server_register.cc
  euler/common/logging.cc
  euler/common/file_io.cc
  euler/common/timmer.cc
  euler/common/str_util.cc
  euler/common/slice.cc
  euler/common/net_util.cc
  euler/common/alias_method.cc
  euler/common/status.cc
  euler/common/random.cc
  euler/common/zk_server_monitor.cc
  euler/common/bytes_io.cc
  euler/common/local_file_io.cc
  euler/common/hash.cc
  euler/common/server_monitor.cc
  euler/common/env_posix.cc
  euler/common/hdfs_file_io.cc
  euler/common/env.cc
  euler/common/data_types.cc
  euler/parser/optimizer.cc
  euler/parser/translator.cc
  euler/parser/compiler.cc
  euler/parser/gen_node_def_input_output.cc
  euler/parser/attribute_calculator.cc
  euler/parser/optimize_type.cc
  euler/core/index/index_manager.cc
  euler/core/index/common_index_result.cc
  euler/core/index/index_meta.cc
  euler/core/graph/graph_builder.cc
  euler/core/graph/edge.cc
  euler/core/graph/graph.cc
  euler/core/graph/graph_meta.cc
  euler/core/graph/node.cc
  euler/core/api/api.cc
  euler/core/dag/dag.cc
  euler/core/dag/node.cc

  euler/core/kernels/remote_op.cc
  euler/core/kernels/append_merge_op.cc
  euler/core/kernels/idx_row_append_merge_op.cc
  euler/core/kernels/sample_node_split_op.cc
  euler/core/kernels/sample_n_with_types_split_op.cc
  euler/core/kernels/sample_edge_split_op.cc
  euler/core/kernels/id_split_op.cc
  euler/core/kernels/get_node_type_op.cc
  euler/core/kernels/as_op.cc
  euler/core/kernels/post_process_op.cc
  euler/core/kernels/get_feature_op.cc
  euler/core/kernels/sample_neighbor_op.cc
  euler/core/kernels/common.cc
  euler/core/kernels/multi_type_idx_merge_op.cc
  euler/core/kernels/multi_type_data_merge_op.cc
  euler/core/kernels/idx_merge_op.cc
  euler/core/kernels/sample_edge_op.cc
  euler/core/kernels/get_neighbor_op.cc
  euler/core/kernels/get_neighbor_edge_op.cc
  euler/core/kernels/get_nb_filter_op.cc
  euler/core/kernels/broad_cast_split_op.cc
  euler/core/kernels/sample_node_op.cc
  euler/core/kernels/sample_n_with_types_op.cc
  euler/core/kernels/get_node_op.cc
  euler/core/kernels/get_edge_op.cc
  euler/core/kernels/data_merge_op.cc
  euler/core/kernels/regular_data_merge_op.cc
  euler/core/kernels/data_row_append_merge_op.cc
  euler/core/kernels/get_edge_sum_weight_op.cc
  euler/core/kernels/sample_root_op.cc
  euler/core/kernels/local_sample_layer_op.cc
  euler/core/kernels/reshape_op.cc
  euler/core/kernels/sample_layer_op.cc
  euler/core/kernels/sparse_gen_adj_op.cc
  euler/core/kernels/sparse_get_adj_op.cc
  euler/core/kernels/gather_result_op.cc
  euler/core/kernels/id_unique_op.cc
  euler/core/kernels/idx_gather_op.cc
  euler/core/kernels/data_gather_op.cc
  euler/core/kernels/sample_graph_label_op.cc
  euler/core/kernels/get_graph_by_label_op.cc

  euler/core/kernels/min_udf.cc
  euler/core/kernels/max_udf.cc
  euler/core/kernels/mean_udf.cc

  euler/core/dag_def/sub_graph_iso.cc
  euler/core/dag_def/dag_node_def.cc
  euler/core/dag_def/dag_def.cc
  euler/core/framework/executor.cc
  euler/core/framework/op_kernel.cc
  euler/core/framework/udf.cc
  euler/core/framework/tensor_util.cc
  euler/core/framework/tensor.cc
  euler/core/framework/allocator.cc
  euler/client/grpc_manager.cc
  euler/client/rpc_client.cc
  euler/client/client_manager.cc
  euler/client/query.cc
  euler/client/graph_config.cc
  euler/client/grpc_channel.cc
  euler/client/rpc_manager.cc
  euler/client/query_proxy.cc
  euler/service/grpc_worker_service.cc
  euler/service/grpc_server.cc
  euler/service/grpc_euler_service.cc
  euler/service/grpc_worker.cc
  euler/service/server_interface.cc
  euler/service/python_api.cc
  euler/util/python_api.cc)

target_link_libraries(euler_core
  libprotobuf
  zookeeper
  grpc++_unsecure
  jemalloc_STATIC_PIC
  ${CMAKE_THREAD_LIBS_INIT})

set(LIB_DST ${PROJECT_SOURCE_DIR}/euler/python)
add_custom_command(TARGET euler_core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:euler_core> ${LIB_DST}
)
