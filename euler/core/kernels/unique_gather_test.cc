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
TEST(UniqueGather, IdUnique) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("ID_UNIQUE,0");
  node_proto.set_op("ID_UNIQUE");
  node_proto.add_inputs("A,0:0");  // id

  Tensor* id_t = nullptr;
  ctx.Allocate("A,0:0", TensorShape({6}), DataType::kUInt64, &id_t);
  std::vector<uint64_t> id = {1, 1, 2, 3, 3, 3};
  std::copy(id.begin(), id.end(), id_t->Raw<uint64_t>());

  OpKernel* id_unique = nullptr;
  CreateOpKernel("ID_UNIQUE", &id_unique);
  id_unique->Compute(node_proto, &ctx);

  Tensor* output0 = nullptr, *output1 = nullptr;
  ctx.tensor("ID_UNIQUE,0:0", &output0);
  ctx.tensor("ID_UNIQUE,0:1", &output1);
  std::vector<uint64_t> unique_id = {1, 2, 3};
  std::vector<int32_t> idx = {0, 0, 1, 2, 2, 2};
  ASSERT_EQ(output0->NumElements(), 3);
  for (int32_t i = 0; i < output0->NumElements(); ++i) {
    ASSERT_EQ(output0->Raw<uint64_t>()[i], unique_id[i]);
  }
  ASSERT_EQ(output1->NumElements(), 6);
  for (int32_t i = 0; i < output1->NumElements(); ++i) {
    ASSERT_EQ(output1->Raw<int32_t>()[i], idx[i]);
  }
}

TEST(UniqueGather, IdUniqueEdge) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("ID_UNIQUE,0");
  node_proto.set_op("ID_UNIQUE");
  node_proto.add_inputs("A,0:0");  // id

  Tensor* id_t = nullptr;
  ctx.Allocate("A,0:0", TensorShape({6, 3}), DataType::kUInt64, &id_t);
  std::vector<uint64_t> id =
    {1, 2, 0, 1, 2, 0, 3, 5, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1};
  std::copy(id.begin(), id.end(), id_t->Raw<uint64_t>());

  OpKernel* id_unique = nullptr;
  CreateOpKernel("ID_UNIQUE", &id_unique);
  id_unique->Compute(node_proto, &ctx);

  Tensor* output0 = nullptr, *output1 = nullptr;
  ctx.tensor("ID_UNIQUE,0:0", &output0);
  ctx.tensor("ID_UNIQUE,0:1", &output1);
  std::vector<uint64_t> unique_id = {1, 2, 0, 3, 5, 1, 3, 3, 1};
  std::vector<int32_t> idx = {0, 0, 1, 2, 2, 2};
  ASSERT_EQ(output0->NumElements(), 9);
  for (int32_t i = 0; i < output0->NumElements(); ++i) {
    ASSERT_EQ(output0->Raw<uint64_t>()[i], unique_id[i]);
  }
  ASSERT_EQ(output1->NumElements(), 6);
  for (int32_t i = 0; i < output1->NumElements(); ++i) {
    ASSERT_EQ(output1->Raw<int32_t>()[i], idx[i]);
  }
}

TEST(UniqueGather, IdxGather) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("IDX_GATHER,0");
  node_proto.set_op("IDX_GATHER");
  node_proto.add_inputs("A,0:0");  // idx
  node_proto.add_inputs("A,0:1");  // gather_idx

  Tensor* idx_t = nullptr, *gather_idx_t = nullptr;
  ctx.Allocate("A,0:0", TensorShape({3, 2}), DataType::kInt32, &idx_t);
  ctx.Allocate("A,0:1", TensorShape({6}), DataType::kInt32, &gather_idx_t);
  std::vector<int32_t> idx = {0, 1, 1, 3, 3, 6};
  std::vector<int32_t> gather_idx = {0, 0, 1, 2, 2, 2};
  std::copy(idx.begin(), idx.end(), idx_t->Raw<int32_t>());
  std::copy(gather_idx.begin(), gather_idx.end(), gather_idx_t->Raw<int32_t>());

  OpKernel* idx_gather = nullptr;
  CreateOpKernel("IDX_GATHER", &idx_gather);
  idx_gather->Compute(node_proto, &ctx);

  Tensor* output0 = nullptr;
  ctx.tensor("IDX_GATHER,0:0", &output0);
  std::vector<int32_t> result = {0, 1, 1, 2, 2, 4, 4, 7, 7, 10, 10, 13};
  ASSERT_EQ(result.size(), output0->NumElements());
  for (int32_t i = 0; i < output0->NumElements(); ++i) {
    ASSERT_EQ(result[i], output0->Raw<int32_t>()[i]);
  }
}

TEST(UniqueGather, DataGather) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("DATA_GATHER,0");
  node_proto.set_op("DATA_GATHER");
  node_proto.add_inputs("A,0:0");  // data
  node_proto.add_inputs("A,0:1");  // idx
  node_proto.add_inputs("A,0:2");  // gather_idx

  Tensor* data_t = nullptr, *idx_t = nullptr, *gather_idx_t = nullptr;
  ctx.Allocate("A,0:0", TensorShape({6}), DataType::kUInt64, &data_t);
  ctx.Allocate("A,0:1", TensorShape({3, 2}), DataType::kInt32, &idx_t);
  ctx.Allocate("A,0:2", TensorShape({6}), DataType::kInt32, &gather_idx_t);
  std::vector<uint64_t> data = {11, 21, 22, 31, 32, 33};
  std::vector<int32_t> idx = {0, 1, 1, 3, 3, 6};
  std::vector<int32_t> gather_idx = {0, 0, 1, 2, 2, 2};
  std::copy(data.begin(), data.end(), data_t->Raw<uint64_t>());
  std::copy(idx.begin(), idx.end(), idx_t->Raw<int32_t>());
  std::copy(gather_idx.begin(), gather_idx.end(), gather_idx_t->Raw<int32_t>());

  OpKernel* data_gather = nullptr;
  CreateOpKernel("DATA_GATHER", &data_gather);
  data_gather->Compute(node_proto, &ctx);

  Tensor* output0 = nullptr;
  ctx.tensor("DATA_GATHER,0:0", &output0);
  std::vector<uint64_t> result =
    {11, 11, 21, 22, 31, 32, 33, 31, 32, 33, 31, 32, 33};
  ASSERT_EQ(result.size(), output0->NumElements());
  for (int32_t i = 0; i < output0->NumElements(); ++i) {
    ASSERT_EQ(result[i], output0->Raw<uint64_t>()[i]);
  }
}
}  // namespace euler
