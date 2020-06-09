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

TEST(GPUniqueMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("GP_UNIQUE_MERGE,0");
  node_proto.set_op("GP_UNIQUE_MERGE");
  node_proto.add_inputs("data,0:0");
  node_proto.add_inputs("");
  node_proto.add_inputs("data,2:0");
  node_proto.add_inputs("");

  // put intput tensor into context
  std::vector<uint64_t> v0 = {1, 2, 3, 4, 5};
  std::vector<uint64_t> v1 = {2, 3, 4, 5, 6};
  Tensor* data0 = nullptr;
  Tensor* data1 = nullptr;
  TensorShape data_shape({5});
  ctx.Allocate("data,0:0", data_shape, DataType::kUInt64, &data0);
  ctx.Allocate("data,2:0", data_shape, DataType::kUInt64, &data1);
  std::copy(v0.begin(), v0.end(), data0->Raw<uint64_t>());
  std::copy(v1.begin(), v1.end(), data1->Raw<uint64_t>());

  // create op and run
  OpKernel* gp_append_merge;
  CreateOpKernel("GP_UNIQUE_MERGE", &gp_append_merge);
  gp_append_merge->Compute(node_proto, &ctx);

  // check output
  Tensor* output = nullptr;
  Tensor* merge_idx1 = nullptr;
  Tensor* merge_idx2 = nullptr;
  ctx.tensor("GP_UNIQUE_MERGE,0:0", &output);
  ctx.tensor("GP_UNIQUE_MERGE,0:1", &merge_idx1);
  ctx.tensor("GP_UNIQUE_MERGE,0:2", &merge_idx2);
  float result[6] = {1, 2, 3, 4, 5, 6};
  for (int32_t i = 0; i < 6; ++i) {
    ASSERT_EQ(output->Raw<uint64_t>()[i], result[i]);
  }
  int32_t merge_i1[5] = {0, 1, 2, 3, 4};
  int32_t merge_i2[5] = {1, 2, 3, 4, 5};
  for (int32_t i = 0; i < 5; ++i) {
    ASSERT_EQ(merge_idx1->Raw<int32_t>()[i], merge_i1[i]);
  }
  for (int32_t i = 0; i < 5; ++i) {
    ASSERT_EQ(merge_idx2->Raw<int32_t>()[i], merge_i2[i]);
  }
}

TEST(GPUniqueMergeOpTest, Execute2) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("GP_UNIQUE_MERGE,0");
  node_proto.set_op("GP_UNIQUE_MERGE");
  node_proto.add_inputs("data,0:0");
  node_proto.add_inputs("");
  node_proto.add_inputs("data,2:0");
  node_proto.add_inputs("");

  // put intput tensor into context
  std::vector<uint64_t> v0 = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::vector<uint64_t> v1 = {2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
  Tensor* data0 = nullptr;
  Tensor* data1 = nullptr;
  TensorShape data_shape({5, 2});
  ctx.Allocate("data,0:0", data_shape, DataType::kUInt64, &data0);
  ctx.Allocate("data,2:0", data_shape, DataType::kUInt64, &data1);
  std::copy(v0.begin(), v0.end(), data0->Raw<uint64_t>());
  std::copy(v1.begin(), v1.end(), data1->Raw<uint64_t>());

  // create op and run
  OpKernel* gp_append_merge;
  CreateOpKernel("GP_UNIQUE_MERGE", &gp_append_merge);
  gp_append_merge->Compute(node_proto, &ctx);

  // check output
  Tensor* output = nullptr;
  Tensor* merge_idx1 = nullptr;
  Tensor* merge_idx2 = nullptr;
  ctx.tensor("GP_UNIQUE_MERGE,0:0", &output);
  ctx.tensor("GP_UNIQUE_MERGE,0:1", &merge_idx1);
  ctx.tensor("GP_UNIQUE_MERGE,0:2", &merge_idx2);
  float result[12] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
  for (int32_t i = 0; i < 12; ++i) {
    ASSERT_EQ(output->Raw<uint64_t>()[i], result[i]);
  }
  int32_t merge_i1[5] = {0, 1, 2, 3, 4};
  int32_t merge_i2[5] = {1, 2, 3, 4, 5};
  for (int32_t i = 0; i < 5; ++i) {
    ASSERT_EQ(merge_idx1->Raw<int32_t>()[i], merge_i1[i]);
  }
  for (int32_t i = 0; i < 5; ++i) {
    ASSERT_EQ(merge_idx2->Raw<int32_t>()[i], merge_i2[i]);
  }
}
}  // namespace euler
