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
#include "euler/core/kernels/common.h"

namespace euler {

class GetEdgeSumWeight: public OpKernel {
 public:
  explicit GetEdgeSumWeight(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void GetEdgeSumWeight::Compute(const DAGNodeProto& node_def,
                               OpKernelContext* ctx) {
  Tensor* root_t = nullptr;
  Tensor* edge_types_t = nullptr;
  ctx->tensor(node_def.inputs(0), &root_t);
  ctx->tensor(node_def.inputs(1), &edge_types_t);

  size_t root_num = root_t->NumElements();
  std::vector<int32_t> edge_types(edge_types_t->NumElements());
  std::copy(edge_types_t->Raw<int32_t>(),
            edge_types_t->Raw<int32_t>() + edge_types.size(),
            edge_types.begin());

  // output
  Tensor* o_root_t = nullptr;
  Tensor* o_root_w_t = nullptr;
  TensorShape shape({root_num, 1});
  ctx->Allocate(OutputName(node_def, 0), shape, DataType::kUInt64, &o_root_t);
  ctx->Allocate(OutputName(node_def, 1), shape, DataType::kFloat, &o_root_w_t);

  std::copy(root_t->Raw<uint64_t>(), root_t->Raw<uint64_t>() + root_num,
            o_root_t->Raw<uint64_t>());
  for (size_t i = 0; i < root_num; ++i) {
    std::vector<uint64_t> roots = {root_t->Raw<uint64_t>()[i]};
    IdWeightPairVec nb = GetFullNeighbor(roots, edge_types);
    float sum_weight = 0;
    for (auto& iw : nb[0]) {
      sum_weight += std::get<1>(iw);
    }
    o_root_w_t->Raw<float>()[i] = sum_weight;
  }
}

REGISTER_OP_KERNEL("API_GET_EDGE_SUM_WEIGHT", GetEdgeSumWeight);

}  // namespace euler
