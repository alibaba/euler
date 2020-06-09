/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "euler/common/logging.h"

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"

namespace euler {

TEST(IdxMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("IDX_MERGE,0");
  node_proto.set_op("IDX_MERGE");
  node_proto.add_inputs("API_GET_P,0:0");  // idx
  node_proto.add_inputs("GET_P_SPLIT,0,0");  // merge idx
  node_proto.add_inputs("API_GET_P,1:0");  // idx
  node_proto.add_inputs("GET_P_SPLIT,0,2");  // merge idx

  // put input tensor into context
  TensorShape shard_idx_shape({2, 2});
  TensorShape merge_idx_shape({2});
  Tensor* idx0 = nullptr;
  Tensor* merge_idx0 = nullptr;
  Tensor* idx1 = nullptr;
  Tensor* merge_idx1 = nullptr;
  ctx.Allocate("API_GET_P,0:0", shard_idx_shape, DataType::kInt32, &idx0);
  ctx.Allocate("GET_P_SPLIT,0,0", merge_idx_shape,
               DataType::kInt32, &merge_idx0);
  ctx.Allocate("API_GET_P,1:0", shard_idx_shape, DataType::kInt32, &idx1);
  ctx.Allocate("GET_P_SPLIT,0,2", merge_idx_shape,
               DataType::kInt32, &merge_idx1);

  std::vector<int32_t> idx0_v = {0, 2, 2, 5};
  std::vector<int32_t> merge_idx0_v = {0, 2};
  std::vector<int32_t> idx1_v = {0, 3, 3, 5};
  std::vector<int32_t> merge_idx1_v = {1, 3};
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(merge_idx0_v.begin(), merge_idx0_v.end(),
            merge_idx0->Raw<int32_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());
  std::copy(merge_idx1_v.begin(), merge_idx1_v.end(),
            merge_idx1->Raw<int32_t>());

  // create op and run
  OpKernel* idx_merge = nullptr;
  CreateOpKernel("IDX_MERGE", &idx_merge);
  idx_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  std::vector<int32_t> idx_result = {0, 2, 2, 5, 5, 8, 8, 10};
  ctx.tensor("IDX_MERGE,0:0", &output);
  for (int i = 0; i < output->NumElements(); ++i) {
    ASSERT_EQ(idx_result[i], output->Raw<int32_t>()[i]);
  }
}

TEST(IdxRowAppendMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("IDX_ROW_APPEND_MERGE,0");
  node_proto.set_op("IDX_ROW_APPEND_MERGE");
  node_proto.add_inputs("graph_label_out,0:0");  // idx
  node_proto.add_inputs("");  // merge idx
  node_proto.add_inputs("graph_label_out,1:0");  // idx
  node_proto.add_inputs("");  // merge idx

  Tensor* idx0 = nullptr;
  Tensor* idx1 = nullptr;
  ctx.Allocate("graph_label_out,0:0", {2, 2}, DataType::kInt32, &idx0);
  ctx.Allocate("graph_label_out,1:0", {2, 2}, DataType::kInt32, &idx1);
  std::vector<int32_t> idx0_v = {0, 2, 2, 5};
  std::vector<int32_t> idx1_v = {0, 3, 3, 5};
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());

  // create op and run
  OpKernel* idx_row_app_merge = nullptr;
  CreateOpKernel("IDX_ROW_APPEND_MERGE", &idx_row_app_merge);
  idx_row_app_merge->Compute(node_proto, &ctx);

  Tensor* output = nullptr;
  std::vector<int32_t> idx_result = {0, 5, 5, 10};
  ctx.tensor("IDX_ROW_APPEND_MERGE,0:0", &output);
  ASSERT_EQ(idx_result.size(), output->NumElements());
  for (int i = 0; i < output->NumElements(); ++i) {
    ASSERT_EQ(idx_result[i], output->Raw<int32_t>()[i]);
  }
}

TEST(GPIdxMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("GP_IDX_MERGE,0");
  node_proto.set_op("GP_IDX_MERGE");
  node_proto.add_inputs("API_GET_P,0:0");  // idx
  node_proto.add_inputs("GET_P_SPLIT,0,0");  // merge idx
  node_proto.add_inputs("API_GET_P,1:0");  // idx
  node_proto.add_inputs("GET_P_SPLIT,0,2");  // merge idx

  // put input tensor into context
  TensorShape shard_idx_shape({3, 2});
  TensorShape merge_idx_shape({3});
  Tensor* idx0 = nullptr;
  Tensor* merge_idx0 = nullptr;
  Tensor* idx1 = nullptr;
  Tensor* merge_idx1 = nullptr;
  ctx.Allocate("API_GET_P,0:0", shard_idx_shape, DataType::kInt32, &idx0);
  ctx.Allocate("GET_P_SPLIT,0,0", merge_idx_shape,
               DataType::kInt32, &merge_idx0);
  ctx.Allocate("API_GET_P,1:0", shard_idx_shape, DataType::kInt32, &idx1);
  ctx.Allocate("GET_P_SPLIT,0,2", merge_idx_shape,
               DataType::kInt32, &merge_idx1);

  std::vector<int32_t> idx0_v = {0, 2, 2, 2, 2, 5};
  std::vector<int32_t> merge_idx0_v = {0, 1, 2};
  std::vector<int32_t> idx1_v = {0, 3, 3, 3, 3, 5};
  std::vector<int32_t> merge_idx1_v = {1, 2, 3};
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(merge_idx0_v.begin(), merge_idx0_v.end(),
            merge_idx0->Raw<int32_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());
  std::copy(merge_idx1_v.begin(), merge_idx1_v.end(),
            merge_idx1->Raw<int32_t>());

  // create op and run
  OpKernel* gp_idx_merge = nullptr;
  CreateOpKernel("GP_IDX_MERGE", &gp_idx_merge);
  gp_idx_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  std::vector<int32_t> idx_result = {0, 2, 2, 5, 5, 8, 8, 10};
  ctx.tensor("GP_IDX_MERGE,0:0", &output);
  for (int i = 0; i < output->NumElements(); ++i) {
    ASSERT_EQ(idx_result[i], output->Raw<int32_t>()[i]);
  }
}

}  // namespace euler
