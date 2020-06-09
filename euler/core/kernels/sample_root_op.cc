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
#include "euler/common/logging.h"
#include "euler/common/fast_weighted_collection.h"

namespace euler {

class SampleRoot: public OpKernel {
 public:
  explicit SampleRoot(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void SampleRoot::Compute(const DAGNodeProto& node_def,
                         OpKernelContext* ctx) {
  Tensor* roots_t = nullptr;
  Tensor* weights_t = nullptr;
  Tensor* n_t = nullptr;  // root num
  Tensor* m_t = nullptr;  // nb num
  ctx->tensor(node_def.inputs(0), &roots_t);
  ctx->tensor(node_def.inputs(1), &weights_t);
  ctx->tensor(node_def.inputs(2), &n_t);
  ctx->tensor(node_def.inputs(3), &m_t);
  uint64_t default_node = atol(node_def.inputs(4).c_str());

  int32_t n = n_t->Raw<int32_t>()[0];
  int32_t m = m_t->Raw<int32_t>()[0];
  int32_t batch = roots_t->NumElements() / n;
  if (batch == 0) EULER_LOG(FATAL) << "batch size is zero!";

  std::vector<std::vector<uint64_t>> roots(batch);
  for (int32_t i = 0; i < batch; ++i) {
    roots[i].resize(n);
    std::copy(roots_t->Raw<uint64_t>() + i * n,
              roots_t->Raw<uint64_t>() + (i + 1) * n,
              roots[i].begin());
  }
  std::vector<std::vector<float>> weights(batch);
  for (int32_t i = 0; i < batch; ++i) {
    weights[i].resize(n);
    std::copy(weights_t->Raw<float>() + i * n,
              weights_t->Raw<float>() + (i + 1) * n,
              weights[i].begin());
  }
  std::vector<common::FastWeightedCollection<uint64_t>> fwcs(batch);
  for (int32_t i = 0; i < batch; ++i) {
    fwcs[i].Init(roots[i], weights[i]);
  }

  Tensor* results_t = nullptr;
  TensorShape results_shape({static_cast<size_t>(batch * m)});
  std::string output_name = OutputName(node_def, 0);
  ctx->Allocate(output_name, results_shape, DataType::kUInt64, &results_t);
  for (int32_t i = 0; i < batch; ++i) {
    std::vector<uint64_t> result(m);
    if (fwcs[i].GetSumWeight() == 0) {
      for (int32_t j = 0; j < m; ++j) {
        result[j] = default_node;
      }
    } else {
      for (int32_t j = 0; j < m; ++j) {
        result[j] = fwcs[i].Sample().first;
      }
    }
    std::copy(result.begin(), result.end(),
              results_t->Raw<uint64_t>() + i * m);
  }
}

REGISTER_OP_KERNEL("API_SAMPLE_ROOT", SampleRoot);

}  // namespace euler
