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
#include <unordered_map>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"
#include "euler/common/logging.h"

namespace euler {
class IdxGather: public OpKernel {
 public:
  explicit IdxGather(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void IdxGather::Compute(const DAGNodeProto& node_def,
                        OpKernelContext* ctx) {
  // get input tensor
  Tensor* idx_t = nullptr, *gather_idx_t = nullptr;
  ctx->tensor(node_def.inputs(0), &idx_t);
  ctx->tensor(node_def.inputs(1), &gather_idx_t);
  // output
  std::string output_name = OutputName(node_def, 0);
  TensorShape shape({static_cast<uint64_t>(gather_idx_t->NumElements()), 2});
  Tensor* output = nullptr;
  ctx->Allocate(output_name, shape, kInt32, &output);
  int32_t base_addr = 0;
  std::vector<int32_t> interval(2);
  for (int32_t i = 0; i < gather_idx_t->NumElements(); ++i) {
    int32_t idx = gather_idx_t->Raw<int32_t>()[i] * 2;
    interval[0] = base_addr;
    interval[1] = base_addr + idx_t->Raw<int32_t>()[idx + 1] -
                  idx_t->Raw<int32_t>()[idx];
    std::copy(interval.begin(), interval.end(),
              output->Raw<int32_t>() + i * 2);
    base_addr = interval[1];
  }
}

REGISTER_OP_KERNEL("IDX_GATHER", IdxGather);

}  // namespace euler
