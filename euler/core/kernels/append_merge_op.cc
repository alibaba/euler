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

class AppendMerge: public OpKernel {
 public:
  explicit AppendMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void AppendMerge::Compute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx) {
  // get input tensor and merge idx
  int32_t total_num = 0;
  std::vector<Tensor*> input_datas;
  std::vector<Tensor*> merge_idxs;
  DataType data_type = kUInt64;
  std::vector<size_t> shape_vec;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    if (node_def.inputs(i) != "") ctx->tensor(node_def.inputs(i), &t);
    if (i % 2 == 0) {  // input data
      input_datas.push_back(t);
      total_num += t->NumElements();
      data_type = t->Type();
      shape_vec = t->Shape().Dims();
    } else {  // merge idx
      merge_idxs.push_back(t);
    }
  }

  // calculate shape
  int32_t other_dim = 1;
  for (size_t i = 1; i < shape_vec.size(); ++i) {
    other_dim *= shape_vec[i];
  }
  shape_vec[0] = total_num / other_dim;

  // merge output result
  std::string output_name = OutputName(node_def, 0);
  Tensor* output_tensor = nullptr;
  TensorShape shape(shape_vec);
  ctx->Allocate(output_name, shape, data_type,
                &output_tensor);
  int32_t output_offset = 0;
  for (size_t i = 0; i < input_datas.size(); ++i) {
    if (merge_idxs[i] != nullptr) {
      output_offset = merge_idxs[i]->Raw<int32_t>()[0];
    } else {
      output_offset += i == 0 ? 0 : input_datas[i - 1]->NumElements();
    }
    if (data_type == kUInt64) {
      std::copy(input_datas[i]->Raw<uint64_t>(),
                input_datas[i]->Raw<uint64_t>() + input_datas[i]->NumElements(),
                output_tensor->Raw<uint64_t>() + output_offset);
    } else if (data_type == kInt64) {
      std::copy(input_datas[i]->Raw<int64_t>(),
                input_datas[i]->Raw<int64_t>() + input_datas[i]->NumElements(),
                output_tensor->Raw<int64_t>() + output_offset);
    } else if (data_type == kFloat) {
      std::copy(input_datas[i]->Raw<float>(),
                input_datas[i]->Raw<float>() + input_datas[i]->NumElements(),
                output_tensor->Raw<float>() + output_offset);
    } else if (data_type == kInt32) {
      std::copy(input_datas[i]->Raw<int32_t>(),
                input_datas[i]->Raw<int32_t>() + input_datas[i]->NumElements(),
                output_tensor->Raw<int32_t>() + output_offset);
    } else {
      EULER_LOG(FATAL) << "type not support yet " << data_type;
    }
  }

  /* graph partition version append merge */
  // add merge index for next merge
  if (name() == "GP_APPEND_MERGE") {
    int32_t split_num = static_cast<int32_t>(input_datas.size());
    int32_t offset = 0;
    for (int32_t i = 0; i < split_num; ++i) {
      std::string output_name = OutputName(node_def, i + 1);
      Tensor* output_tensor = nullptr;
      size_t size = input_datas[i]->Shape().Dims()[0];
      ctx->Allocate(output_name, {size}, kInt32,
                    &output_tensor);
      if (merge_idxs[i] != nullptr) {
        offset = merge_idxs[i]->Raw<int32_t>()[0];
      } else {
        offset += i == 0 ? 0 : input_datas[i - 1]->NumElements();
      }
      for (size_t j = 0; j < size; ++j) {
        output_tensor->Raw<int32_t>()[j] = offset + static_cast<int32_t>(j);
      }
    }
  }
}

REGISTER_OP_KERNEL("APPEND_MERGE", AppendMerge);
REGISTER_OP_KERNEL("GP_APPEND_MERGE", AppendMerge);

}  // namespace euler
