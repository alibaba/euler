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
#include "euler/common/str_util.h"
#include "euler/client/query_proxy.h"

namespace euler {
class BroadCastSplit: public OpKernel {
 public:
  explicit BroadCastSplit(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void BroadCastSplit::Compute(const DAGNodeProto& node_def,
                             OpKernelContext* ctx) {
  int32_t split_num = QueryProxy::GetInstance()->GetShardNum();
  /* get input tensor */
  Tensor* input_t = nullptr;
  ctx->tensor(node_def.inputs(0), &input_t);

  /* output */
  for (int32_t i = 0; i < split_num * 2; i += 2) {
    std::string output_name = OutputName(node_def, i);
    ctx->AddAlias(output_name, input_t);
  }
}

REGISTER_OP_KERNEL("BROAD_CAST_SPLIT", BroadCastSplit);

}  // namespace euler
