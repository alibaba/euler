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

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"

namespace euler {
class Reshape: public OpKernel {
 public:
  explicit Reshape(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void Reshape::Compute(const DAGNodeProto& node_def,
                      OpKernelContext* ctx) {
  Tensor* batch_nb_idx_t = nullptr;
  ctx->tensor(node_def.inputs(0), &batch_nb_idx_t);
  std::string shape_str = node_def.inputs(1);
  std::vector<std::string> shape_str_vec = Split(shape_str, ',');
  int32_t unknow_idx = -1;
  size_t other_size = 1;
  std::vector<size_t> shape(shape_str_vec.size());
  for (size_t i = 0; i < shape_str_vec.size(); ++i) {
    if (shape_str_vec[i] == "?" && unknow_idx == -1) {
      unknow_idx = i;
    } else if (shape_str_vec[i] == "?" && unknow_idx != -1) {
      EULER_LOG(FATAL) << "reshape reg error: " << shape_str;
    } else {
      shape[i] = atol(shape_str_vec[i].c_str());
      other_size *= shape[i];
    }
  }
  if (unknow_idx != -1) {
    shape[unknow_idx] = batch_nb_idx_t->NumElements() / other_size;
    if (shape[unknow_idx] * other_size != static_cast<size_t>(
            batch_nb_idx_t->NumElements())) {
      EULER_LOG(FATAL) << "reshape reg error: " << shape_str;
    }
  } else {
    if (other_size != static_cast<size_t>(
            batch_nb_idx_t->NumElements())) {
      EULER_LOG(FATAL) << "reshape reg error: " << shape_str;
    }
  }
  Tensor* output_t = nullptr;
  TensorShape tensor_shape(shape);
  ctx->Allocate(OutputName(node_def, 0), tensor_shape,
                batch_nb_idx_t->Type(), &output_t);
  if (batch_nb_idx_t->Type() == DataType::kUInt64) {
    std::copy(batch_nb_idx_t->Raw<uint64_t>(),
              batch_nb_idx_t->Raw<uint64_t>() + batch_nb_idx_t->NumElements(),
              output_t->Raw<uint64_t>());
  } else if (batch_nb_idx_t->Type() == DataType::kFloat) {
    std::copy(batch_nb_idx_t->Raw<float>(),
              batch_nb_idx_t->Raw<float>() + batch_nb_idx_t->NumElements(),
              output_t->Raw<float>());
  } else if (batch_nb_idx_t->Type() == DataType::kInt32) {
    std::copy(batch_nb_idx_t->Raw<int32_t>(),
              batch_nb_idx_t->Raw<int32_t>() + batch_nb_idx_t->NumElements(),
              output_t->Raw<int32_t>());
  } else {
    EULER_LOG(FATAL) << "data type not support";
  }
}

REGISTER_OP_KERNEL("API_RESHAPE", Reshape);

}  // namespace euler
