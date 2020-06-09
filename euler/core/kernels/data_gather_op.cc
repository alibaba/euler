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
class DataGather: public OpKernel {
 public:
  explicit DataGather(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

#define GATHER_DATA(DATA_TYPE) {                              \
  int32_t base_addr = 0;                                      \
  for (int32_t i = 0; i < gather_idx_t->NumElements(); ++i) { \
    int32_t idx = gather_idx_t->Raw<int32_t>()[i] * 2;        \
    int32_t begin = idx_t->Raw<int32_t>()[idx];               \
    int32_t end = idx_t->Raw<int32_t>()[idx + 1];             \
    std::copy(                                                \
        data_t->Raw<DATA_TYPE>() + begin,                     \
        data_t->Raw<DATA_TYPE>() + end,                       \
        output->Raw<DATA_TYPE>() + base_addr);                \
    base_addr += end - begin;                                 \
  }                                                           \
}

void DataGather::Compute(const DAGNodeProto& node_def,
                         OpKernelContext* ctx) {
  // get input
  Tensor* data_t = nullptr, *idx_t = nullptr, *gather_idx_t = nullptr;
  ctx->tensor(node_def.inputs(0), &data_t);
  ctx->tensor(node_def.inputs(1), &idx_t);
  ctx->tensor(node_def.inputs(2), &gather_idx_t);

  // output
  std::string output_name = OutputName(node_def, 0);
  DataType data_type = data_t->Type();
  size_t size = 0;
  for (int32_t i = 0; i < gather_idx_t->NumElements(); ++i) {
    int32_t idx = gather_idx_t->Raw<int32_t>()[i] * 2;
    size += idx_t->Raw<int32_t>()[idx + 1] - idx_t->Raw<int32_t>()[idx];
  }

  TensorShape shape({size});
  Tensor* output = nullptr;
  ctx->Allocate(output_name, shape, data_type, &output);
  if (data_type == DataType::kUInt64) {
    GATHER_DATA(uint64_t);
  } else if (data_type == DataType::kFloat) {
    GATHER_DATA(float);
  } else if (data_type == DataType::kInt8) {
    GATHER_DATA(char);
  } else if (data_type == DataType::kInt32) {
    GATHER_DATA(int32_t);
  } else {
    EULER_LOG(ERROR) << "error data type";
  }
}

REGISTER_OP_KERNEL("DATA_GATHER", DataGather);
}  // namespace euler
