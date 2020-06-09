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

#include <string>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"
#include "euler/core/api/api.h"

namespace euler {

class SampleEdgeOp: public OpKernel {
 public:
  explicit SampleEdgeOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void SampleEdgeOp::Compute(const DAGNodeProto& node_def, OpKernelContext* ctx) {
  if (node_def.inputs_size() != 2) {
    EULER_LOG(ERROR) << "Invalid input arguments";
    return;
  }

  Tensor* edge_type_t = nullptr;
  auto s = ctx->tensor(node_def.inputs(0), &edge_type_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Retrieve edge_type input failed!";
    return;
  }

  Tensor* count_t = nullptr;
  s = ctx->tensor(node_def.inputs(1), &count_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Retrieve count input failed!";
    return;
  }

  std::vector<int> edge_types(edge_type_t->NumElements());
  std::copy(edge_type_t->Raw<int32_t>(),
            edge_type_t->Raw<int32_t>() + edge_type_t->NumElements(),
            edge_types.begin());
  int count = *count_t->Raw<int32_t>();
  auto edge_id_vec = euler::SampleEdge(edge_types, count);
  if (edge_id_vec.size() != static_cast<size_t>(count)) {
    EULER_LOG(ERROR) << "Expect sample count: " << count
                     << "， real got:" << edge_id_vec.size();
    return;
  }

  std::string output_name = OutputName(node_def, 0);
  TensorShape shape({ static_cast<size_t>(count), 3ul });
  Tensor* output = nullptr;
  s = ctx->Allocate(output_name, shape, DataType::kInt64, &output);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor failed!";
    return;
  }

  auto data = output->Raw<int64_t>();
  for (auto& edge_id : edge_id_vec) {
    *data++ = std::get<0>(edge_id);
    *data++ = std::get<1>(edge_id);
    *data++ = std::get<2>(edge_id);
  }
}

REGISTER_OP_KERNEL("API_SAMPLE_EDGE", SampleEdgeOp);

}  // namespace euler
