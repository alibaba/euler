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

#include <algorithm>
#include <iostream>

#include "gtest/gtest.h"

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"

namespace euler {

TEST(GetNBFilterTest, Executor) {
  OpKernelContext ctx;

  DAGNodeProto proto;
  proto.set_name("API_GET_NB_FILTER,0");
  proto.set_op("API_GET_NB_FILTER");
  /*
   * 1,2,3
   * 4,5
   * 6,7,8
   */
  proto.add_inputs("index");
  proto.add_inputs("id");
  proto.add_inputs("weight");
  proto.add_inputs("type");
  proto.add_inputs("filter");
  proto.add_post_process("order_by id asc");

  std::vector<int32_t> idx = {0, 3, 3, 5, 5, 8};
  std::vector<uint64_t> id = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> w = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> t = {1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<uint64_t> f = {1, 3, 5, 7};
  std::vector<Tensor*> tensors(5);
  ctx.Allocate("index", TensorShape({3, 2}), kInt32, &(tensors[0]));
  ctx.Allocate("id", TensorShape({8}), kUInt64, &(tensors[1]));
  ctx.Allocate("weight", TensorShape({8}), kFloat, &(tensors[2]));
  ctx.Allocate("type", TensorShape({8}), kInt32, &(tensors[3]));
  ctx.Allocate("filter", TensorShape({4}), kUInt64, &(tensors[4]));
  std::copy(idx.begin(), idx.end(), tensors[0]->Raw<int32_t>());
  std::copy(id.begin(), id.end(), tensors[1]->Raw<uint64_t>());
  std::copy(w.begin(), w.end(), tensors[2]->Raw<float>());
  std::copy(t.begin(), t.end(), tensors[3]->Raw<int32_t>());
  std::copy(f.begin(), f.end(), tensors[4]->Raw<uint64_t>());

  OpKernel* op = nullptr;
  CreateOpKernel("API_GET_NB_FILTER", &op);
  op->Compute(proto, &ctx);

  Tensor* o_idx_t = nullptr;
  Tensor* o_id_t = nullptr;
  Tensor* o_w_t = nullptr;
  Tensor* o_t_t = nullptr;
  ctx.tensor("API_GET_NB_FILTER,0:0", &o_idx_t);
  ctx.tensor("API_GET_NB_FILTER,0:1", &o_id_t);
  ctx.tensor("API_GET_NB_FILTER,0:2", &o_w_t);
  ctx.tensor("API_GET_NB_FILTER,0:3", &o_t_t);
  std::vector<int32_t> o_idx = {0, 2, 2, 3, 3, 4};
  std::vector<uint64_t> o_id = {1, 3, 5, 7};
  std::vector<float> o_w = {1, 3, 5, 7};
  std::vector<int32_t> o_t = {1, 1, 1, 1};
  ASSERT_EQ(o_idx.size(), o_idx_t->NumElements());
  ASSERT_EQ(o_id.size(), o_id_t->NumElements());
  ASSERT_EQ(o_w.size(), o_w_t->NumElements());
  ASSERT_EQ(o_t.size(), o_t_t->NumElements());
  for (size_t i = 0 ; i < o_idx.size(); ++i) {
    ASSERT_EQ(o_idx[i], o_idx_t->Raw<int32_t>()[i]);
  }
  for (size_t i = 0; i < o_id.size(); ++i) {
    ASSERT_EQ(o_id[i], o_id_t->Raw<uint64_t>()[i]);
    ASSERT_EQ(o_w[i], o_w_t->Raw<float>()[i]);
    ASSERT_EQ(o_t[i], o_t_t->Raw<int32_t>()[i]);
  }
}

}  // namespace euler

