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

#include "euler/common/data_types.h"
#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"


namespace euler {
TEST(DataMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("DATA_MERGE,0");
  node_proto.set_op("DATA_MERGE");
  node_proto.add_inputs("API_GET_P,0:1");  // data
  node_proto.add_inputs("API_GET_P,0:0");  // idx
  node_proto.add_inputs("ID_SPLIT,2:1");  // merge idx
  node_proto.add_inputs("API_GET_P,1:1");  // data
  node_proto.add_inputs("API_GET_P,1:0");  // idx
  node_proto.add_inputs("ID_SPLIT,2:3");  // merge idx

  // put input tensor into context
  TensorShape shard_idx_shape({2, 2});
  TensorShape shard_data_shape({5});
  TensorShape merge_idx_shape({2});
  Tensor* data0 = nullptr;
  Tensor* idx0 = nullptr;
  Tensor* merge_idx0 = nullptr;
  Tensor* data1 = nullptr;
  Tensor* idx1 = nullptr;
  Tensor* merge_idx1 = nullptr;
  ctx.Allocate("API_GET_P,0:1", shard_data_shape, DataType::kUInt64, &data0);
  ctx.Allocate("API_GET_P,0:0", shard_idx_shape, DataType::kInt32, &idx0);
  ctx.Allocate("ID_SPLIT,2:1", merge_idx_shape, DataType::kInt32, &merge_idx0);
  ctx.Allocate("API_GET_P,1:1", shard_data_shape, DataType::kUInt64, &data1);
  ctx.Allocate("API_GET_P,1:0", shard_idx_shape, DataType::kInt32, &idx1);
  ctx.Allocate("ID_SPLIT,2:3", merge_idx_shape, DataType::kInt32, &merge_idx1);

  std::vector<uint64_t> data0_v = {1, 2, 6, 7, 8};
  std::vector<int32_t> idx0_v = {0, 2, 2, 5};
  std::vector<int32_t> merge_idx0_v = {0, 2};
  std::vector<uint64_t> data1_v = {3, 4, 5, 9, 10};
  std::vector<int32_t> idx1_v = {0, 3, 3, 5};
  std::vector<int32_t> merge_idx1_v = {1, 3};
  std::copy(data0_v.begin(), data0_v.end(), data0->Raw<uint64_t>());
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(merge_idx0_v.begin(), merge_idx0_v.end(),
            merge_idx0->Raw<int32_t>());
  std::copy(data1_v.begin(), data1_v.end(), data1->Raw<uint64_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());
  std::copy(merge_idx1_v.begin(), merge_idx1_v.end(),
            merge_idx1->Raw<int32_t>());

  // create op and run
  OpKernel* data_merge = nullptr;
  CreateOpKernel("DATA_MERGE", &data_merge);
  data_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  std::vector<uint64_t> data_result = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ctx.tensor("DATA_MERGE,0:0", &output);
  for (size_t i = 0; i < data_result.size(); ++i) {
    ASSERT_EQ(data_result[i], output->Raw<uint64_t>()[i]);
  }
}

TEST(DataRowAppendMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("DATA_ROW_APPEND_MERGE,0");
  node_proto.set_op("DATA_ROW_APPEND_MERGE");
  node_proto.add_inputs("data,0:1");  // data
  node_proto.add_inputs("data,0:0");  // idx
  node_proto.add_inputs("");  // empty merge idx
  node_proto.add_inputs("data,1:1");  // data
  node_proto.add_inputs("data,1:0");  // idx
  node_proto.add_inputs("");  // empty merge idx

  // put input tensor into context
  TensorShape shard_idx_shape({2, 2});
  TensorShape shard_data_shape({5});
  TensorShape merge_idx_shape({2});
  Tensor* data0 = nullptr;
  Tensor* idx0 = nullptr;
  Tensor* data1 = nullptr;
  Tensor* idx1 = nullptr;
  ctx.Allocate("data,0:1", {5}, DataType::kUInt64, &data0);
  ctx.Allocate("data,0:0", {2, 2}, DataType::kInt32, &idx0);
  ctx.Allocate("data,1:1", {5}, DataType::kUInt64, &data1);
  ctx.Allocate("data,1:0", {2, 2}, DataType::kInt32, &idx1);

  std::vector<uint64_t> data0_v = {1, 2, 6, 7, 8};
  std::vector<int32_t> idx0_v = {0, 2, 2, 5};
  std::vector<uint64_t> data1_v = {3, 4, 5, 9, 10};
  std::vector<int32_t> idx1_v = {0, 3, 3, 5};
  std::copy(data0_v.begin(), data0_v.end(), data0->Raw<uint64_t>());
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(data1_v.begin(), data1_v.end(), data1->Raw<uint64_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());

  // create op and run
  OpKernel* data_row_app_merge = nullptr;
  CreateOpKernel("DATA_ROW_APPEND_MERGE", &data_row_app_merge);
  data_row_app_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  std::vector<uint64_t> data_result = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ctx.tensor("DATA_ROW_APPEND_MERGE,0:0", &output);
  ASSERT_EQ(data_result.size(), output->NumElements());
  for (size_t i = 0; i < data_result.size(); ++i) {
    ASSERT_EQ(data_result[i], output->Raw<uint64_t>()[i]);
  }
}

TEST(GPDataMergeOpTest, Execute1) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("GP_DATA_MERGE,0");
  node_proto.set_op("GP_DATA_MERGE");
  node_proto.add_inputs("API_GET_P,0:1");  // data
  node_proto.add_inputs("API_GET_P,0:0");  // idx
  node_proto.add_inputs("ID_SPLIT,2:1");  // merge idx
  node_proto.add_inputs("API_GET_P,1:1");  // data
  node_proto.add_inputs("API_GET_P,1:0");  // idx
  node_proto.add_inputs("ID_SPLIT,2:3");  // merge idx

  // put input tensor into context
  TensorShape shard_idx_shape({3, 2});
  TensorShape shard_data_shape({5});
  TensorShape merge_idx_shape({3});
  Tensor* data0 = nullptr;
  Tensor* idx0 = nullptr;
  Tensor* merge_idx0 = nullptr;
  Tensor* data1 = nullptr;
  Tensor* idx1 = nullptr;
  Tensor* merge_idx1 = nullptr;
  ctx.Allocate("API_GET_P,0:1", shard_data_shape, DataType::kUInt64, &data0);
  ctx.Allocate("API_GET_P,0:0", shard_idx_shape, DataType::kInt32, &idx0);
  ctx.Allocate("ID_SPLIT,2:1", merge_idx_shape, DataType::kInt32, &merge_idx0);
  ctx.Allocate("API_GET_P,1:1", shard_data_shape, DataType::kUInt64, &data1);
  ctx.Allocate("API_GET_P,1:0", shard_idx_shape, DataType::kInt32, &idx1);
  ctx.Allocate("ID_SPLIT,2:3", merge_idx_shape, DataType::kInt32, &merge_idx1);

  std::vector<uint64_t> data0_v = {1, 2, 6, 7, 8};
  std::vector<int32_t> idx0_v = {0, 2, 2, 2, 2, 5};
  std::vector<int32_t> merge_idx0_v = {0, 1, 2};
  std::vector<uint64_t> data1_v = {3, 4, 5, 9, 10};
  std::vector<int32_t> idx1_v = {0, 3, 3, 3, 3, 5};
  std::vector<int32_t> merge_idx1_v = {1, 2, 3};
  std::copy(data0_v.begin(), data0_v.end(), data0->Raw<uint64_t>());
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(merge_idx0_v.begin(), merge_idx0_v.end(),
            merge_idx0->Raw<int32_t>());
  std::copy(data1_v.begin(), data1_v.end(), data1->Raw<uint64_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());
  std::copy(merge_idx1_v.begin(), merge_idx1_v.end(),
            merge_idx1->Raw<int32_t>());

  // create op and run
  OpKernel* data_merge = nullptr;
  CreateOpKernel("GP_DATA_MERGE", &data_merge);
  data_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  Tensor* merge_idx_0_t = nullptr;
  Tensor* merge_idx_1_t = nullptr;
  std::vector<uint64_t> data_result = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int32_t> merge_idx_0 = {0, 1, 5, 6, 7};
  std::vector<int32_t> merge_idx_1 = {2, 3, 4, 8, 9};
  ctx.tensor("GP_DATA_MERGE,0:0", &output);
  for (size_t i = 0; i < data_result.size(); ++i) {
    ASSERT_EQ(data_result[i], output->Raw<uint64_t>()[i]);
  }
  ctx.tensor("GP_DATA_MERGE,0:1", &merge_idx_0_t);
  for (size_t i = 0; i < merge_idx_0.size(); ++i) {
    ASSERT_EQ(merge_idx_0[i], merge_idx_0_t->Raw<int32_t>()[i]);
  }
  ctx.tensor("GP_DATA_MERGE,0:2", &merge_idx_1_t);
  for (size_t i = 0; i < merge_idx_1.size(); ++i) {
    ASSERT_EQ(merge_idx_1[i], merge_idx_1_t->Raw<int32_t>()[i]);
  }
}

TEST(GPDataMergeOpTest, Execute2) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("GP_DATA_MERGE,0");
  node_proto.set_op("GP_DATA_MERGE");
  node_proto.add_inputs("API_GET_P,0:1");  // data
  node_proto.add_inputs("API_GET_P,0:0");  // idx
  node_proto.add_inputs("ID_SPLIT,2:1");  // merge idx
  node_proto.add_inputs("API_GET_P,1:1");  // data
  node_proto.add_inputs("API_GET_P,1:0");  // idx
  node_proto.add_inputs("ID_SPLIT,2:3");  // merge idx

  // put input tensor into context
  TensorShape shard_idx_shape({3, 2});
  TensorShape shard_data_shape({8});
  TensorShape merge_idx_shape({3});
  Tensor* data0 = nullptr;
  Tensor* idx0 = nullptr;
  Tensor* merge_idx0 = nullptr;
  Tensor* data1 = nullptr;
  Tensor* idx1 = nullptr;
  Tensor* merge_idx1 = nullptr;
  ctx.Allocate("API_GET_P,0:1", shard_data_shape, DataType::kUInt64, &data0);
  ctx.Allocate("API_GET_P,0:0", shard_idx_shape, DataType::kInt32, &idx0);
  ctx.Allocate("ID_SPLIT,2:1", merge_idx_shape, DataType::kInt32, &merge_idx0);
  ctx.Allocate("API_GET_P,1:1", shard_data_shape, DataType::kUInt64, &data1);
  ctx.Allocate("API_GET_P,1:0", shard_idx_shape, DataType::kInt32, &idx1);
  ctx.Allocate("ID_SPLIT,2:3", merge_idx_shape, DataType::kInt32, &merge_idx1);

  uint64_t d = euler::common::DEFAULT_UINT64;
  std::vector<uint64_t> data0_v = {1, 2, d, d, d, 6, 7, 8};
  std::vector<int32_t> idx0_v = {0, 2, 2, 5, 5, 8};
  std::vector<int32_t> merge_idx0_v = {0, 1, 2};
  std::vector<uint64_t> data1_v = {3, 4, 5, d, d, d, 9, 10};
  std::vector<int32_t> idx1_v = {0, 3, 3, 6, 6, 8};
  std::vector<int32_t> merge_idx1_v = {1, 2, 3};
  std::copy(data0_v.begin(), data0_v.end(), data0->Raw<uint64_t>());
  std::copy(idx0_v.begin(), idx0_v.end(), idx0->Raw<int32_t>());
  std::copy(merge_idx0_v.begin(), merge_idx0_v.end(),
            merge_idx0->Raw<int32_t>());
  std::copy(data1_v.begin(), data1_v.end(), data1->Raw<uint64_t>());
  std::copy(idx1_v.begin(), idx1_v.end(), idx1->Raw<int32_t>());
  std::copy(merge_idx1_v.begin(), merge_idx1_v.end(),
            merge_idx1->Raw<int32_t>());

  // create op and run
  OpKernel* data_merge = nullptr;
  CreateOpKernel("GP_DATA_MERGE", &data_merge);
  data_merge->Compute(node_proto, &ctx);

  // check result
  Tensor* output = nullptr;
  Tensor* merge_idx_0_t = nullptr;
  Tensor* merge_idx_1_t = nullptr;
  std::vector<uint64_t> data_result = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int32_t> merge_idx_0 = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> merge_idx_1 = {2, 3, 4, 5, 6, 7, 8, 9};
  ctx.tensor("GP_DATA_MERGE,0:0", &output);
  for (size_t i = 0; i < data_result.size(); ++i) {
    ASSERT_EQ(data_result[i], output->Raw<uint64_t>()[i]);
  }
  ctx.tensor("GP_DATA_MERGE,0:1", &merge_idx_0_t);
  for (size_t i = 0; i < merge_idx_0.size(); ++i) {
    ASSERT_EQ(merge_idx_0[i], merge_idx_0_t->Raw<int32_t>()[i]);
  }
  ctx.tensor("GP_DATA_MERGE,0:2", &merge_idx_1_t);
  for (size_t i = 0; i < merge_idx_1.size(); ++i) {
    ASSERT_EQ(merge_idx_1[i], merge_idx_1_t->Raw<int32_t>()[i]);
  }
}

}  // namespace euler
