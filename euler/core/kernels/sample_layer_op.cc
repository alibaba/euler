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
class SampleLayer: public OpKernel {
 public:
  explicit SampleLayer(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void SampleLayer::Compute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx) {
  Tensor* l_root_t = nullptr;
  Tensor* edge_types_t = nullptr;
  ctx->tensor(node_def.inputs(0), &l_root_t);
  ctx->tensor(node_def.inputs(1), &edge_types_t);
  int64_t default_node = atol(node_def.inputs(2).c_str());

  size_t l_root_num = l_root_t->NumElements();
  std::vector<int32_t> edge_types(edge_types_t->NumElements());
  std::copy(edge_types_t->Raw<int32_t>(),
            edge_types_t->Raw<int32_t>() + edge_types.size(),
            edge_types.begin());

  // output
  Tensor* o_nb_t = nullptr;
  Tensor* o_w_t = nullptr;
  Tensor* o_t_t = nullptr;
  TensorShape shape({l_root_num, 1});
  ctx->Allocate(OutputName(node_def, 0), shape, DataType::kUInt64, &o_nb_t);
  ctx->Allocate(OutputName(node_def, 1), shape, DataType::kFloat, &o_w_t);
  ctx->Allocate(OutputName(node_def, 2), shape, DataType::kInt32, &o_t_t);
  int32_t addr = 0;
  for (size_t i = 0; i < l_root_num; ++i) {
    std::vector<uint64_t> roots =
        {l_root_t->Raw<uint64_t>()[i]};
    IdWeightPairVec nb = SampleNeighbor(roots, edge_types, 1);
    if (nb[0].empty()) {
      o_nb_t->Raw<uint64_t>()[addr] = default_node;
      o_w_t->Raw<float>()[addr] = 0;
      o_t_t->Raw<int32_t>()[addr] = 0;
    } else {
      o_nb_t->Raw<uint64_t>()[addr] = std::get<0>(nb[0][0]);
      o_w_t->Raw<float>()[addr] = std::get<1>(nb[0][0]);
      o_t_t->Raw<int32_t>()[addr] = std::get<2>(nb[0][0]);
    }
    ++addr;
  }
}

REGISTER_OP_KERNEL("API_SAMPLE_L", SampleLayer);

}  // namespace euler
