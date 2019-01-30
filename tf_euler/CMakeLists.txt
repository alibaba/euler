if (BUILD_COVERAGE)
  set(CMAKE_CXX_FLAGS "-g -O0 --coverage")
  set (CMAKE_EXE_LINKER_FLAGS "--coverage")
ENDIF(BUILD_COVERAGE)

# Add tensorflow framework

execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE TF_INC_DIR)
string(REGEX REPLACE "\n$" "" TF_INC_DIR  "${TF_INC_DIR}")
string(REGEX REPLACE "\n$" "" TF_LIB_DIR  "${TF_LIB_DIR}")

include_directories(SYSTEM ${TF_INC_DIR})

find_library(
  TF_FRAMEWORK
  NAMES tensorflow_framework
  PATHS ${TF_LIB_DIR}
)

message(STATUS "tensorflow_framework library found: in ${TF_FRAMEWORK}")

add_library(tf_euler SHARED
            ops/sample_ops.cc
            ops/feature_ops.cc
            ops/neighbor_ops.cc
            ops/walk_ops.cc
            ops/type_ops.cc
            ops/util_ops.cc

            kernels/sample_node_op.cc
            kernels/sample_edge_op.cc

            kernels/get_node_type_op.cc

            kernels/get_binary_feature_op.cc
            kernels/get_dense_feature_op.cc
            kernels/get_sparse_feature_op.cc
            kernels/get_edge_binary_feature_op.cc
            kernels/get_edge_dense_feature_op.cc
            kernels/get_edge_sparse_feature_op.cc

            kernels/get_full_neighbor_op.cc
            kernels/get_sorted_full_neighbor_op.cc
            kernels/get_top_k_neighbor_op.cc
            kernels/sample_neighbor_op.cc
            kernels/random_walk_op.cc
            kernels/gen_pair_op.cc
            kernels/inflate_idx_op.cc

            utils/create_graph.cc)

target_link_libraries(tf_euler client common ${TF_FRAMEWORK})
target_include_directories(tf_euler BEFORE PRIVATE ${TF_INC_DIR}
                           "${TF_INC_DIR}/external/nsync/public/")

set(LIB_DST ${CMAKE_CURRENT_SOURCE_DIR}/python/euler_ops)
add_custom_command(TARGET tf_euler POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:client> ${LIB_DST}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tf_euler> ${LIB_DST}
)
