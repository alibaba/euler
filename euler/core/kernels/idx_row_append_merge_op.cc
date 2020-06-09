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

class IdxRowAppendMergeOp: public OpKernel {
 public:
  explicit IdxRowAppendMergeOp(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void IdxRowAppendMergeOp::Compute(const DAGNodeProto& node_def,
                                  OpKernelContext* ctx) {
  std::vector<Tensor*> input_datas;
  int32_t row = -1;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    if (i % 2 == 0) {  // input data
      Tensor* t = nullptr;
      ctx->tensor(node_def.inputs(i), &t);
      input_datas.push_back(t);
      if (row == -1) row = t->NumElements() / 2;
      if (row != t->NumElements() / 2) {
        EULER_LOG(FATAL) << "row size not match";
      }
    }
  }

  // output
  std::string output_name = OutputName(node_def, 0);
  Tensor* output_tensor = nullptr;
  ctx->Allocate(output_name, {static_cast<size_t>(row), 2},
                DataType::kInt32, &output_tensor);
  int32_t offset = 0;
  for (int32_t i = 0; i < row; ++i) {
    int32_t size = 0;
    for (size_t j = 0; j < input_datas.size(); ++j) {
      size += input_datas[j]->Raw<int32_t>()[i * 2 + 1] -
              input_datas[j]->Raw<int32_t>()[i * 2];
    }
    output_tensor->Raw<int32_t>()[i * 2] = offset;
    output_tensor->Raw<int32_t>()[i * 2 + 1] = offset + size;
    offset += size;
  }
}

REGISTER_OP_KERNEL("IDX_ROW_APPEND_MERGE", IdxRowAppendMergeOp);

}  // namespace euler
