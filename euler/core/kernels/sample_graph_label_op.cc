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
#include <vector>
#include <cmath>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"
#include "euler/common/random.h"
#include "euler/common/str_util.h"
#include "euler/client/query_proxy.h"

namespace euler {

class SampleGraphLabelOp: public OpKernel {
 public:
  explicit SampleGraphLabelOp(const std::string& name):
      OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void SampleGraphLabelOp::Compute(
    const DAGNodeProto& node_def, OpKernelContext* ctx) {

  Tensor* count_tensor = nullptr;
  ctx->tensor(node_def.inputs(0), &count_tensor);
  int32_t count = count_tensor->Raw<int32_t>()[0];

  const std::vector<std::string>& graph_label =
      QueryProxy::GetInstance()->GetGraphLabel();

  std::vector<std::string> results; results.reserve(count);
  if (graph_label.empty()) {
    EULER_LOG(FATAL) << "graph label set is empty!";
  }
  for (int32_t i = 0; i < count; ++i) {
    int32_t idx = floor(common::ThreadLocalRandom() * graph_label.size());
    results.push_back(graph_label[idx]);
  }
  std::string result_str = Join(results, ",");

  // output
  std::string output_name = OutputName(node_def.name(), 0);
  Tensor* output = nullptr;
  ctx->Allocate(output_name, {result_str.size()},
      DataType::kInt8, &output);
  std::copy(result_str.begin(), result_str.end(),
            output->Raw<char>());
}

REGISTER_OP_KERNEL("API_SAMPLE_GRAPH_LABEL", SampleGraphLabelOp);

}  // namespace euler
