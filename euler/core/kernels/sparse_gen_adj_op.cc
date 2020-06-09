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

namespace euler {

class SparseGenAdj: public OpKernel {
 public:
  explicit SparseGenAdj(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void SparseGenAdj::Compute(const DAGNodeProto& node_def,
                           OpKernelContext* ctx) {
  Tensor* roots_t = nullptr;
  Tensor* l_nb_t = nullptr;
  Tensor* n_t = nullptr;
  ctx->tensor(node_def.inputs(0), &roots_t);
  ctx->tensor(node_def.inputs(1), &l_nb_t);
  ctx->tensor(node_def.inputs(2), &n_t);

  int32_t n = n_t->Raw<int32_t>()[0];
  int32_t batch = roots_t->NumElements() / n;
  if (batch == 0) EULER_LOG(FATAL) << "batch size is zero!";

  Tensor* root_batch_t = nullptr;
  TensorShape shape({static_cast<size_t>(roots_t->NumElements()), 2});
  ctx->Allocate(OutputName(node_def, 0), shape,
                DataType::kUInt64, &root_batch_t);
  for (int32_t i = 0; i < batch; ++i) {
    for (int32_t j = 0; j < n; ++j) {
      int32_t cnt = i * n + j;
      // root id
      root_batch_t->Raw<uint64_t>()[cnt * 2] = roots_t->Raw<uint64_t>()[cnt];
      // batch number
      root_batch_t->Raw<uint64_t>()[cnt * 2 + 1] = i;
    }
  }
  ctx->AddAlias(OutputName(node_def, 1), l_nb_t);
}

REGISTER_OP_KERNEL("API_SPARSE_GEN_ADJ", SparseGenAdj);

}  // namespace euler
