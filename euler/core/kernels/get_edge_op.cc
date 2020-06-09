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

class GetEdgeOp : public OpKernel {
 public:
  explicit GetEdgeOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void GetEdgeOp::Compute(const DAGNodeProto& node_def, OpKernelContext* ctx) {
  if (node_def.inputs_size() == 0 && node_def.dnf_size() == 0) {
    EULER_LOG(ERROR) << "Edge ids or filter must be specified for GetEdgeOp";
    return;
  } else if (node_def.inputs_size() != 0 &&
             node_def.dnf_size() == 0 && node_def.post_process_size() == 0) {
    std::string output_name = OutputName(node_def, 0);
    Tensor* input = nullptr;
    ctx->tensor(node_def.inputs(0), &input);
    ctx->AddAlias(output_name, input);
    return;
  }

  auto& graph = Graph::Instance();
  std::vector<Graph::UID> uids;
  if (node_def.inputs_size() > 0) {
    EdgeIdVec edge_ids;
    auto s = GetEdgeIds(node_def, 0, ctx, &edge_ids);
    if (!s.ok()) {
      EULER_LOG(ERROR) << "Invalid argment 'edge_ids', status: " << s;
      return;
    }

    uids.reserve(edge_ids.size());
    for (auto& eid : edge_ids) {
      uids.push_back(graph.EdgeIdToUID(eid));
    }
  }

  euler::FilerByIndex(node_def, ctx, &uids);

  // post proccess
  for (auto& post : node_def.post_process()) {
    auto vec = Split(post, " ");
    if (vec[0] == "limit") {
      if (vec.size() != 2) {
        EULER_LOG(ERROR) << "Invalid post process: " << post;
        continue;
      }
      uids.resize(atoi(vec[1].c_str()));
    }
  }

  // Fill output tensor
  std::string output_name = OutputName(node_def, 0);
  Tensor* output = nullptr;
  TensorShape shape({ uids.size(), 3 });
  auto s = ctx->Allocate(output_name, shape, DataType::kUInt64, &output);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor failed!";
    return;
  }

  auto data = output->Raw<uint64_t>();
  for (auto& uid : uids) {
    auto eid = graph.UIDToEdgeId(uid);
    data[0] = std::get<0>(eid);
    data[1] = std::get<1>(eid);
    data[2] = std::get<2>(eid);
    data += 3;
  }
}

REGISTER_OP_KERNEL("API_GET_EDGE", GetEdgeOp);

}  // namespace euler
