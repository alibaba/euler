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

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"

namespace euler {
class As: public OpKernel {
 public:
  explicit As(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void As::Compute(const DAGNodeProto& node_def,
                 OpKernelContext* ctx) {
  std::string alias_name = node_def.op_alias();
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    ctx->AddAlias(OutputName(alias_name, i), t);
  }
}

REGISTER_OP_KERNEL("AS", As);

}  // namespace euler
