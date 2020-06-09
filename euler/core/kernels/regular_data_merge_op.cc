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
#include "euler/common/data_types.h"

namespace euler {
class RegularDataMerge: public OpKernel {
 public:
  explicit RegularDataMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

#define REGULE_MERGE_RESULT(DATA_TYPE, DEFAULT_VALUE) {               \
  std::string output_name = OutputName(node_def, 0);                  \
  Tensor* data_result = nullptr;                                      \
  std::vector<size_t> idx_shape_v = datas[0]->Shape().Dims();         \
  idx_shape_v[0] = ids_num;                                           \
  TensorShape idx_shape(idx_shape_v);                                 \
  ctx->Allocate(output_name, idx_shape, data_type, &data_result);     \
  for (int32_t i = 0; i < data_result->NumElements(); ++i) {          \
    data_result->Raw<DATA_TYPE>()[i] = DEFAULT_VALUE;                 \
  }                                                                   \
  for (size_t i = 0; i < datas.size(); ++i) {                         \
    Tensor* merge_t = merge_idxs[i];                                  \
    for (size_t j = 0; j < datas[i]->Shape().Dims()[0]; ++j) {        \
      int32_t result_addr = merge_t->Raw<int32_t>()[j] * offset;      \
      if (datas[i]->Raw<DATA_TYPE>()[j * offset] != DEFAULT_VALUE) {  \
        std::copy(datas[i]->Raw<DATA_TYPE>() + j * offset,            \
                  datas[i]->Raw<DATA_TYPE>() + (j + 1) * offset,      \
                  data_result->Raw<DATA_TYPE>() + result_addr);       \
      }                                                               \
    }                                                                 \
  }                                                                   \
}

void RegularDataMerge::Compute(const DAGNodeProto& node_def,
                               OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> datas;
  std::vector<Tensor*> merge_idxs;
  size_t ids_num = 0;
  int32_t offset = 1;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    if (i % 2 == 0) {  // id
      datas.push_back(t);
      ids_num += t->Shape().Dims()[0];
    } else {  // merge_idx
      merge_idxs.push_back(t);
    }
  }
  for (size_t i = 1; i < datas[0]->Shape().Dims().size(); ++i) {
    offset *= datas[0]->Shape().Dims()[i];
  }

  DataType data_type = datas[0]->Type();
  /* merge output result */
  if (data_type == DataType::kUInt64) {
    REGULE_MERGE_RESULT(uint64_t, euler::common::DEFAULT_UINT64);
  } else if (data_type == DataType::kFloat) {
    REGULE_MERGE_RESULT(float, euler::common::DEFAULT_FLOAT);
  } else if (data_type == DataType::kInt8) {
    REGULE_MERGE_RESULT(char, euler::common::DEFAULT_CHAR);
  } else if (data_type == DataType::kInt32) {
    REGULE_MERGE_RESULT(int32_t, euler::common::DEFAULT_INT32);
  } else {
    EULER_LOG(ERROR) << "error data type";
  }
}

REGISTER_OP_KERNEL("REGULAR_DATA_MERGE", RegularDataMerge);
REGISTER_OP_KERNEL("GP_REGULAR_DATA_MERGE", RegularDataMerge);
}  // namespace euler
