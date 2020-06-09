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
#include "euler/common/data_types.h"

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"

namespace euler {

TEST(GPRegularDataMergeOpTest, Execute) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("GP_REGULAR_DATA_MERGE,0");
  node_proto.set_op("GP_REGULAR_DATA_MERGE");
  node_proto.add_inputs("data,0:0");
  node_proto.add_inputs("idx,1:0");
  node_proto.add_inputs("data,2:0");
  node_proto.add_inputs("idx,3:0");

  // put intput tensor into context
  std::vector<float> v0 = {1, 2, 3, 4, 5};
  std::vector<int32_t> idx0 = {0, 1, 2, 3, 4};
  std::vector<float> v1 = {1, 2, euler::common::DEFAULT_FLOAT, 6, 7};
  std::vector<int32_t> idx1 = {0, 1, 2, 5, 6};
  Tensor* data0 = nullptr, *idx0_t = nullptr;
  Tensor* data1 = nullptr, *idx1_t = nullptr;
  TensorShape data_shape({5});
  ctx.Allocate("data,0:0", data_shape, DataType::kFloat, &data0);
  ctx.Allocate("idx,1:0", data_shape, DataType::kInt32, &idx0_t);
  ctx.Allocate("data,2:0", data_shape, DataType::kFloat, &data1);
  ctx.Allocate("idx,3:0", data_shape, DataType::kInt32, &idx1_t);
  std::copy(v0.begin(), v0.end(), data0->Raw<float>());
  std::copy(idx0.begin(), idx0.end(), idx0_t->Raw<int32_t>());
  std::copy(v1.begin(), v1.end(), data1->Raw<float>());
  std::copy(idx1.begin(), idx1.end(), idx1_t->Raw<int32_t>());

  // create op and run
  OpKernel* gp_regular_data_merge;
  CreateOpKernel("GP_REGULAR_DATA_MERGE", &gp_regular_data_merge);
  gp_regular_data_merge->Compute(node_proto, &ctx);

  // check output
  Tensor* output = nullptr;
  ctx.tensor("GP_REGULAR_DATA_MERGE,0:0", &output);
  float result[7] = {1, 2, 3, 4, 5, 6, 7};
  for (int32_t i = 0; i < 7; ++i) {
    ASSERT_EQ(output->Raw<float>()[i], result[i]);
  }
}

}  // namespace euler
