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

class DataRowAppendMergeOp: public OpKernel {
 public:
  explicit DataRowAppendMergeOp(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

#define ROW_MERGE(DATA_TYPE) {                               \
  int32_t offset = 0;                                        \
  for (int32_t i = 0; i < row; ++i) {                        \
    for (size_t j = 0; j < data_idxs.size(); ++j) {          \
      int32_t begin = data_idxs[j]->Raw<int32_t>()[i * 2];   \
      int32_t end = data_idxs[j]->Raw<int32_t>()[i * 2 + 1]; \
      std::copy(datas[j]->Raw<DATA_TYPE>() + begin,          \
                datas[j]->Raw<DATA_TYPE>() + end,            \
                output_tensor->Raw<DATA_TYPE>() + offset);   \
      offset += end - begin;                                 \
    }                                                        \
  }                                                          \
}

void DataRowAppendMergeOp::Compute(const DAGNodeProto& node_def,
                                   OpKernelContext* ctx) {
  std::vector<Tensor*> datas;
  std::vector<Tensor*> data_idxs;
  int32_t row = -1;
  size_t count = 0;
  DataType data_type = kUInt64;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    if (i % 3 == 0) {  // data
      Tensor* t = nullptr;
      ctx->tensor(node_def.inputs(i), &t);
      datas.push_back(t);
      count += t->NumElements();
      data_type = t->Type();
    } else if (i % 3 == 1) {  // data idx
      Tensor* t = nullptr;
      ctx->tensor(node_def.inputs(i), &t);
      data_idxs.push_back(t);

      if (row == -1) row = t->NumElements() / 2;
      if (row != t->NumElements() / 2) {
        EULER_LOG(FATAL) << "row size not match";
      }
    }
  }

  // output
  std::string output_name = OutputName(node_def, 0);
  Tensor* output_tensor = nullptr;
  ctx->Allocate(output_name, {count},
                data_type, &output_tensor);

  if (data_type == DataType::kUInt64) {
    ROW_MERGE(uint64_t);
  } else if (data_type == DataType::kFloat) {
    ROW_MERGE(float);
  } else if (data_type == DataType::kInt8) {
    ROW_MERGE(char);
  } else if (data_type == DataType::kInt32) {
    ROW_MERGE(int32_t);
  } else {
    EULER_LOG(ERROR) << "error data type";
  }
}

REGISTER_OP_KERNEL("DATA_ROW_APPEND_MERGE", DataRowAppendMergeOp);

}  // namespace euler
