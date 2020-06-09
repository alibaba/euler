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
#include <string>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/api/api.h"
#include "euler/common/str_util.h"

namespace euler {

class GetAdj: public OpKernel {
 public:
  explicit GetAdj(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void GetAdj::Compute(const DAGNodeProto& node_def,
                     OpKernelContext* ctx) {
  Tensor* adj_edges_t = nullptr;
  Tensor* edge_types_t = nullptr;
  ctx->tensor(node_def.inputs(0), &adj_edges_t);
  ctx->tensor(node_def.inputs(1), &edge_types_t);

  size_t edge_num = adj_edges_t->NumElements() / 2;
  std::vector<int32_t> edge_types(edge_types_t->NumElements());
  std::copy(edge_types_t->Raw<int32_t>(),
            edge_types_t->Raw<int32_t>() + edge_types.size(),
            edge_types.begin());

  // output
  Tensor* o_adj_t = nullptr;
  TensorShape shape({edge_num, 1});
  ctx->Allocate(OutputName(node_def, 0), shape, DataType::kInt32, &o_adj_t);
  for (size_t i = 0; i < edge_num; ++i) {
    bool exist = false;
    for (int32_t e_type : edge_types) {
      EdgeId eid(adj_edges_t->Raw<uint64_t>()[i * 2],
                 adj_edges_t->Raw<uint64_t>()[i * 2 + 1],
                 e_type);
      exist = exist || EdgeExist(eid);
    }
    o_adj_t->Raw<int32_t>()[i] = exist == true ? 1 : 0;
  }
}

REGISTER_OP_KERNEL("API_GET_ADJ", GetAdj);

}  // namespace euler
