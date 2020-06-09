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
#include "euler/core/kernels/common.h"
#include "euler/common/str_util.h"

namespace euler {
class GatherResult: public OpKernel {
 public:
  explicit GatherResult(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void GatherResult::Compute(const DAGNodeProto& node_def,
                           OpKernelContext* ctx) {
  Tensor* adj_idx_t = nullptr;
  Tensor* adj_nb_t = nullptr;
  Tensor* l_nb_t = nullptr;
  ctx->tensor(node_def.inputs(0), &adj_idx_t);
  ctx->tensor(node_def.inputs(1), &adj_nb_t);
  ctx->tensor(node_def.inputs(2), &l_nb_t);

  ctx->AddAlias(OutputName(node_def, 0), adj_idx_t);
  ctx->AddAlias(OutputName(node_def, 1), adj_nb_t);
  ctx->AddAlias(OutputName(node_def, 2), l_nb_t);
}

REGISTER_OP_KERNEL("API_GATHER_RESULT", GatherResult);

}  // namespace euler
