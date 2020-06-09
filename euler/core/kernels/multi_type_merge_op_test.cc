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
#include <iostream>

#include "gtest/gtest.h"
#include "euler/common/logging.h"

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"

namespace euler {

TEST(MultiType, IdxMerge) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("MULTI_TYPE_IDX_MERGE,0");
  node_proto.set_op("MULTI_TYPE_IDX_MERGE");
  node_proto.add_inputs("idx0");  // idx0
  node_proto.add_inputs("");
  node_proto.add_inputs("idx1");  // idx1
  node_proto.add_inputs("");

  Tensor* idx0_t = nullptr;
  Tensor* idx1_t = nullptr;
  ctx.Allocate("idx0", TensorShape({2, 2}), DataType::kInt32, &idx0_t);
  ctx.Allocate("idx1", TensorShape({2, 2}), DataType::kInt32, &idx1_t);
  std::vector<int32_t> idx0 = {0, 2, 2, 4};
  std::vector<int32_t> idx1 = {0, 4, 4, 8};
  std::copy(idx0.begin(), idx0.end(), idx0_t->Raw<int32_t>());
  std::copy(idx1.begin(), idx1.end(), idx1_t->Raw<int32_t>());

  // create op and run
  OpKernel* idx_merge = nullptr;
  CreateOpKernel("MULTI_TYPE_IDX_MERGE", &idx_merge);
  idx_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  std::vector<int32_t> idx_result = {0, 6, 6, 12};
  ctx.tensor("MULTI_TYPE_IDX_MERGE,0:0", &output);
  ASSERT_EQ(idx_result.size(), output->NumElements());
  for (int i = 0; i < output->NumElements(); ++i) {
    ASSERT_EQ(idx_result[i], output->Raw<int32_t>()[i]);
  }
}

TEST(MultiType, DataMerge) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("MULTI_TYPE_DATA_MERGE,0");
  node_proto.set_op("MULTI_TYPE_DATA_MERGE");
  node_proto.add_inputs("data0");
  node_proto.add_inputs("idx0");
  node_proto.add_inputs("");
  node_proto.add_inputs("data1");
  node_proto.add_inputs("idx1");
  node_proto.add_inputs("");

  Tensor* data0_t = nullptr;
  Tensor* idx0_t = nullptr;
  Tensor* data1_t = nullptr;
  Tensor* idx1_t = nullptr;
  ctx.Allocate("data0", TensorShape({4}), DataType::kUInt64, &data0_t);
  ctx.Allocate("idx0", TensorShape({2, 2}), DataType::kInt32, &idx0_t);
  ctx.Allocate("data1", TensorShape({8}), DataType::kUInt64, &data1_t);
  ctx.Allocate("idx1", TensorShape({2, 2}), DataType::kInt32, &idx1_t);
  std::vector<uint64_t> data0 = {1, 2, 3, 4};
  std::vector<int32_t> idx0 = {0, 2, 2, 4};
  std::vector<uint64_t> data1 = {5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> idx1 = {0, 4, 4, 8};
  std::copy(data0.begin(), data0.end(), data0_t->Raw<uint64_t>());
  std::copy(idx0.begin(), idx0.end(), idx0_t->Raw<int32_t>());
  std::copy(data1.begin(), data1.end(), data1_t->Raw<uint64_t>());
  std::copy(idx1.begin(), idx1.end(), idx1_t->Raw<int32_t>());

  // create op and run
  OpKernel* idx_merge = nullptr;
  CreateOpKernel("MULTI_TYPE_DATA_MERGE", &idx_merge);
  idx_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  std::vector<uint64_t> idx_result = {1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12};
  ctx.tensor("MULTI_TYPE_DATA_MERGE,0:0", &output);
  ASSERT_EQ(idx_result.size(), output->NumElements());
  for (int i = 0; i < output->NumElements(); ++i) {
    ASSERT_EQ(idx_result[i], output->Raw<uint64_t>()[i]);
  }
}

}  // namespace euler
