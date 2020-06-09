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

namespace euler {

class GetNodeOp : public OpKernel {
 public:
  explicit GetNodeOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void GetNodeOp::Compute(const DAGNodeProto& node_def, OpKernelContext* ctx) {
  if (node_def.inputs_size() == 0 && node_def.dnf_size() == 0) {
    EULER_LOG(ERROR) << "Node ids or filter must be specified for GetNodeOp";
    return;
  }

  NodeIdVec node_ids;
  if (node_def.inputs_size() > 0) {
    auto s = GetNodeIds(node_def, 0, ctx, &node_ids);
    if (!s.ok()) {
      EULER_LOG(ERROR) << "Retrieve argment node_ids failed!";
      return;
    }
  }

  euler::FilerByIndex(node_def, ctx, &node_ids);

  // post proccess
  for (auto& post : node_def.post_process()) {
    auto vec = Split(post, " ");
    if (vec[0] == "order_by") {
      if (vec.size() < 2 || vec.size() > 3 ||  vec[1] != "id") {
        EULER_LOG(ERROR) << "Invalid post process: " << post;
        continue;
      }

      int sign = 1;
      if (vec.size() == 3 && vec[2] == "desc") {
        sign = -1;
      }

      auto cmp = [sign] (NodeId a, NodeId b) {
        return (a < b ? 1 : -1) * sign > 0;
      };
      std::sort(node_ids.begin(), node_ids.end(), cmp);
    } else if (vec[0] == "limit") {
      if (vec.size() != 2) {
        EULER_LOG(ERROR) << "Invalid post process: " << post;
        continue;
      }
      node_ids.resize(atoi(vec[1].c_str()));
    }
  }

  // Fill output tensor
  std::string output_name = OutputName(node_def, 0);
  Tensor* output = nullptr;
  TensorShape shape({ node_ids.size() });
  auto s = ctx->Allocate(output_name, shape, DataType::kUInt64, &output);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor failed!";
    return;
  }

  std::copy(node_ids.begin(), node_ids.end(), output->Raw<uint64_t>());
}

REGISTER_OP_KERNEL("API_GET_NODE", GetNodeOp);

}  // namespace euler
