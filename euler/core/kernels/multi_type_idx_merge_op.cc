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

namespace euler {

class MultiTypeIdxMerge: public OpKernel {
 public:
  explicit MultiTypeIdxMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void MultiTypeIdxMerge::Compute(const DAGNodeProto& node_def,
                                OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> idxs;
  size_t types_num = 0;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    if (node_def.inputs(i) != "") ctx->tensor(node_def.inputs(i), &t);
    if (i % 2 == 0) {
      idxs.push_back(t);
      types_num = t->Shape().Dims()[0];
    }
  }

  /* merge output result */
  std::string output_name = OutputName(node_def, 0);
  Tensor* idx_result = nullptr;
  TensorShape idx_shape({types_num, 2});
  ctx->Allocate(output_name, idx_shape, DataType::kInt32, &idx_result);
  int32_t base_addr = 0;
  for (size_t i = 0; i < types_num; ++i) {
    int32_t cnt = 0;
    for (size_t j = 0; j < idxs.size(); ++j) {  // each shard
      cnt += idxs[j]->Raw<int32_t>()[i * 2 + 1] -
             idxs[j]->Raw<int32_t>()[i * 2];
    }
    idx_result->Raw<int32_t>()[i * 2] = base_addr;
    idx_result->Raw<int32_t>()[i * 2 + 1] = base_addr + cnt;
    base_addr = idx_result->Raw<int32_t>()[i * 2 + 1];
  }
}

REGISTER_OP_KERNEL("MULTI_TYPE_IDX_MERGE", MultiTypeIdxMerge);

}  // namespace euler
