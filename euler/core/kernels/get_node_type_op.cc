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

#include <stdlib.h>

#include <string>

#include "euler/core/kernels/common.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"
#include "euler/core/api/api.h"
#include "euler/core/graph/graph.h"

namespace euler {

class GetNodeTypeOp : public OpKernel {
 public:
  explicit GetNodeTypeOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void GetNodeTypeOp::Compute(const DAGNodeProto& node_def,
                            OpKernelContext* ctx) {
  if (node_def.inputs_size() == 0) {
    EULER_LOG(ERROR) << "Node ids must be specified!";
    return;
  }

  NodeIdVec node_ids;
  auto s = GetNodeIds(node_def, 0, ctx, &node_ids);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Invalid node_ids argment specified!";
    return;
  }

  auto res = GetNodeType(node_ids);

  // Fill output
  auto outname = OutputName(node_def, 0);
  Tensor* output = nullptr;
  TensorShape shape({ res.size(), 1 });
  s = ctx->Allocate(outname, shape, DataType::kInt32, &output);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor " << outname << " failed!";
    return;
  }

  std::copy(res.begin(), res.end(), output->Raw<int32_t>());
}

REGISTER_OP_KERNEL("API_GET_NODE_T", GetNodeTypeOp);

}  // namespace euler
