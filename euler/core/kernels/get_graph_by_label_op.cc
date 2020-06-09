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
class GetGraphByLabelOp: public OpKernel {
 public:
  explicit GetGraphByLabelOp(const std::string& name): OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void GetGraphByLabelOp::Compute(
    const DAGNodeProto& node_def, OpKernelContext* ctx) {
  Tensor* labels_t = nullptr;
  auto s = ctx->tensor(node_def.inputs(0), &labels_t);
  if (!s.ok()) {
    EULER_LOG(FATAL) << "graph label input error";
  }
  std::vector<std::string> labels;
  labels.reserve(labels_t->NumElements());
  for (int32_t i = 0; i < labels_t->NumElements(); ++i) {
    labels.push_back(*(labels_t->Raw<std::string*>()[i]));
  }
  std::vector<std::shared_ptr<IndexResult>> results =
      QueryIndex("graph_label", labels);

  size_t total_size = 0;
  std::vector<std::vector<uint64_t>> node_id_results;
  node_id_results.reserve(results.size());
  for (std::shared_ptr<IndexResult> result : results) {
    if (result != nullptr) {
      node_id_results.push_back(result->GetIds());
      total_size += node_id_results.back().size();
    } else {
      node_id_results.push_back({});
    }
  }

  std::string index_name = OutputName(node_def, 0);
  std::string data_name = OutputName(node_def, 1);
  Tensor* index_tensor = nullptr;
  Tensor* data_tensor = nullptr;
  ctx->Allocate(index_name, {results.size(), 2},
                DataType::kInt32, &index_tensor);
  ctx->Allocate(data_name, {total_size}, DataType::kUInt64,
                &data_tensor);
  int32_t offset = 0;
  for (size_t i = 0; i < node_id_results.size(); ++i) {
    index_tensor->Raw<int32_t>()[i * 2] = offset;
    index_tensor->Raw<int32_t>()[i * 2 + 1] =
        offset + node_id_results[i].size();
    std::copy(node_id_results[i].begin(),
              node_id_results[i].end(),
              data_tensor->Raw<uint64_t>() + offset);
    offset += node_id_results[i].size();
  }
}

REGISTER_OP_KERNEL("API_GET_GRAPH_BY_LABEL", GetGraphByLabelOp);

}  // namespace euler
