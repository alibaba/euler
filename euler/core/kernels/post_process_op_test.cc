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

#include <vector>

#include "gtest/gtest.h"

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"

namespace euler {

TEST(PostProcessOpTest, OrderById_asc) {
  OpKernelContext ctx;
  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("POST_PROCESS,0");
  node_proto.set_op("POST_PROCESS");
  node_proto.add_inputs("ids");
  node_proto.add_inputs("weights");
  node_proto.add_post_process("order_by id asc");
  node_proto.add_post_process("limit 10");

  Tensor* node_ids_t = nullptr;
  Tensor* weights_t = nullptr;
  TensorShape ids_shape({15});
  TensorShape weights_shape({15});
  ctx.Allocate("ids", ids_shape, kUInt64, &node_ids_t);
  ctx.Allocate("weights", weights_shape, kFloat, &weights_t);
  std::vector<uint64_t> ids =
    {1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15};
  std::vector<float> weights =
    {1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15};
  std::copy(ids.begin(), ids.end(), node_ids_t->Raw<uint64_t>());
  std::copy(weights.begin(), weights.end(), weights_t->Raw<float>());

  OpKernel* pp_op;
  CreateOpKernel("POST_PROCESS", &pp_op);
  pp_op->Compute(node_proto, &ctx);

  Tensor* output_ids_t = nullptr;
  Tensor* output_weights_t = nullptr;
  ctx.tensor("POST_PROCESS,0:0", &output_ids_t);
  ctx.tensor("POST_PROCESS,0:1", &output_weights_t);
  std::vector<uint64_t> o_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<float> o_weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ASSERT_EQ(o_ids.size(), output_ids_t->NumElements());
  ASSERT_EQ(o_weights.size(), output_weights_t->NumElements());
  for (int32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(o_ids[i], output_ids_t->Raw<uint64_t>()[i]);
    ASSERT_EQ(o_weights[i], output_weights_t->Raw<float>()[i]);
  }
}

TEST(PostProcessOpTest, OrderByWeight_desc) {
  OpKernelContext ctx;
  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("POST_PROCESS,0");
  node_proto.set_op("POST_PROCESS");
  node_proto.add_inputs("ids");
  node_proto.add_inputs("weights");
  node_proto.add_post_process("order_by weights desc");
  node_proto.add_post_process("limit 10");

  Tensor* node_ids_t = nullptr;
  Tensor* weights_t = nullptr;
  TensorShape ids_shape({15});
  TensorShape weights_shape({15});
  ctx.Allocate("ids", ids_shape, kUInt64, &node_ids_t);
  ctx.Allocate("weights", weights_shape, kFloat, &weights_t);
  std::vector<uint64_t> ids =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::vector<float> weights =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::copy(ids.begin(), ids.end(), node_ids_t->Raw<uint64_t>());
  std::copy(weights.begin(), weights.end(), weights_t->Raw<float>());

  OpKernel* pp_op;
  CreateOpKernel("POST_PROCESS", &pp_op);
  pp_op->Compute(node_proto, &ctx);

  Tensor* output_ids_t = nullptr;
  Tensor* output_weights_t = nullptr;
  ctx.tensor("POST_PROCESS,0:0", &output_ids_t);
  ctx.tensor("POST_PROCESS,0:1", &output_weights_t);
  std::vector<uint64_t> o_ids = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6};
  std::vector<float> o_weights = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6};
  ASSERT_EQ(o_ids.size(), output_ids_t->NumElements());
  ASSERT_EQ(o_weights.size(), output_weights_t->NumElements());
  for (int32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(o_ids[i], output_ids_t->Raw<uint64_t>()[i]);
    ASSERT_EQ(o_weights[i], output_weights_t->Raw<float>()[i]);
  }
}
TEST(PostProcessOpTest, Limit) {
  OpKernelContext ctx;
  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("POST_PROCESS,0");
  node_proto.set_op("POST_PROCESS");
  node_proto.add_inputs("ids");
  node_proto.add_inputs("weights");
  node_proto.add_post_process("limit 10");

  Tensor* node_ids_t = nullptr;
  Tensor* weights_t = nullptr;
  TensorShape ids_shape({15});
  TensorShape weights_shape({15});
  ctx.Allocate("ids", ids_shape, kUInt64, &node_ids_t);
  ctx.Allocate("weights", weights_shape, kFloat, &weights_t);
  std::vector<uint64_t> ids =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::vector<float> weights =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::copy(ids.begin(), ids.end(), node_ids_t->Raw<uint64_t>());
  std::copy(weights.begin(), weights.end(), weights_t->Raw<float>());

  OpKernel* pp_op;
  CreateOpKernel("POST_PROCESS", &pp_op);
  pp_op->Compute(node_proto, &ctx);

  Tensor* output_ids_t = nullptr;
  Tensor* output_weights_t = nullptr;
  ctx.tensor("POST_PROCESS,0:0", &output_ids_t);
  ctx.tensor("POST_PROCESS,0:1", &output_weights_t);
  std::vector<uint64_t> o_ids = {9, 7, 5, 3, 1, 15, 14, 13, 12, 11};
  std::vector<float> o_weights = {9, 7, 5, 3, 1, 15, 14, 13, 12, 11};
  ASSERT_EQ(o_ids.size(), output_ids_t->NumElements());
  ASSERT_EQ(o_weights.size(), output_weights_t->NumElements());
  for (int32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(o_ids[i], output_ids_t->Raw<uint64_t>()[i]);
    ASSERT_EQ(o_weights[i], output_weights_t->Raw<float>()[i]);
  }
}

TEST(PostProcessOpTest, OrderByWeight_withoutLimit_desc) {
  OpKernelContext ctx;
  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("POST_PROCESS,0");
  node_proto.set_op("POST_PROCESS");
  node_proto.add_inputs("ids");
  node_proto.add_inputs("weights");
  node_proto.add_post_process("order_by weights desc");

  Tensor* node_ids_t = nullptr;
  Tensor* weights_t = nullptr;
  TensorShape ids_shape({15});
  TensorShape weights_shape({15});
  ctx.Allocate("ids", ids_shape, kUInt64, &node_ids_t);
  ctx.Allocate("weights", weights_shape, kFloat, &weights_t);
  std::vector<uint64_t> ids =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::vector<float> weights =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::copy(ids.begin(), ids.end(), node_ids_t->Raw<uint64_t>());
  std::copy(weights.begin(), weights.end(), weights_t->Raw<float>());

  OpKernel* pp_op;
  CreateOpKernel("POST_PROCESS", &pp_op);
  pp_op->Compute(node_proto, &ctx);

  Tensor* output_ids_t = nullptr;
  Tensor* output_weights_t = nullptr;
  ctx.tensor("POST_PROCESS,0:0", &output_ids_t);
  ctx.tensor("POST_PROCESS,0:1", &output_weights_t);
  std::vector<uint64_t> o_ids =
    {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<float> o_weights =
    {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  ASSERT_EQ(o_ids.size(), output_ids_t->NumElements());
  ASSERT_EQ(o_weights.size(), output_weights_t->NumElements());
  for (int32_t i = 0; i < 15; ++i) {
    ASSERT_EQ(o_ids[i], output_ids_t->Raw<uint64_t>()[i]);
    ASSERT_EQ(o_weights[i], output_weights_t->Raw<float>()[i]);
  }
}

TEST(PostProcessOpTest, OrderByWeight_edgeId_desc) {
  OpKernelContext ctx;
  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("POST_PROCESS,0");
  node_proto.set_op("POST_PROCESS");
  node_proto.add_inputs("ids");
  node_proto.add_inputs("weights");
  node_proto.add_post_process("order_by weights desc");
  node_proto.add_post_process("limit 10");

  Tensor* node_ids_t = nullptr;
  Tensor* weights_t = nullptr;
  TensorShape ids_shape({15, 3});
  TensorShape weights_shape({15});
  ctx.Allocate("ids", ids_shape, kUInt64, &node_ids_t);
  ctx.Allocate("weights", weights_shape, kFloat, &weights_t);
  std::vector<uint64_t> ids = {9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3, 1, 1, 1,
    15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10,
    8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2};
  std::vector<float> weights =
    {9, 7, 5, 3, 1, 15, 14, 13, 12, 11, 10, 8, 6, 4, 2};
  std::copy(ids.begin(), ids.end(), node_ids_t->Raw<uint64_t>());
  std::copy(weights.begin(), weights.end(), weights_t->Raw<float>());

  OpKernel* pp_op;
  CreateOpKernel("POST_PROCESS", &pp_op);
  pp_op->Compute(node_proto, &ctx);

  Tensor* output_ids_t = nullptr;
  Tensor* output_weights_t = nullptr;
  ctx.tensor("POST_PROCESS,0:0", &output_ids_t);
  ctx.tensor("POST_PROCESS,0:1", &output_weights_t);
  std::vector<uint64_t> o_ids = {15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12,
    11, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6};
  std::vector<float> o_weights = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6};
  ASSERT_EQ(o_ids.size(), output_ids_t->NumElements());
  ASSERT_EQ(o_weights.size(), output_weights_t->NumElements());
  for (int32_t i = 0; i < 30; ++i) {
    ASSERT_EQ(o_ids[i], output_ids_t->Raw<uint64_t>()[i]);
  }
  for (int32_t i = 0; i < 10; ++i) {
    ASSERT_EQ(o_weights[i], output_weights_t->Raw<float>()[i]);
  }
}

}  // namespace euler
