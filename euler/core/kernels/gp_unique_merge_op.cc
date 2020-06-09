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

class GPUniqueMerge: public OpKernel {
 public:
  explicit GPUniqueMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
 private:
  std::string DataToString(uint64_t* begin, size_t offset) {
    std::string result = "";
    for (size_t i = 0; i < offset; ++i) {
      result += std::to_string(*(begin + i));
    }
    return result;
  }
};

void GPUniqueMerge::Compute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx) {
  // get input tensor and merge idx
  std::vector<Tensor*> input_datas;
  DataType data_type = kUInt64;
  size_t total_size = 0;
  std::vector<size_t> data_shape;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    if (i % 2 == 0) {  // input data
      ctx->tensor(node_def.inputs(i), &t);
      input_datas.push_back(t);
      if (t->Type() != kUInt64) {
        EULER_LOG(FATAL) << "unique merge not support this kind of data";
      }
      total_size += t->NumElements();
      data_shape = t->Shape().Dims();
    }
  }
  size_t offset = 1;
  for (size_t i = 1; i < input_datas[0]->Shape().Dims().size(); ++i) {
    offset *= input_datas[0]->Shape().Dims()[i];
  }

  // merge output result
  std::unordered_map<std::string, int32_t> unique_map(total_size / offset);
  std::vector<uint64_t> unique_result;
  unique_result.reserve(total_size);
  int32_t unique_cnt = 0;
  for (size_t i = 0; i < input_datas.size(); ++i) {
    for (int32_t j = 0; j < input_datas[i]->NumElements(); j += offset) {
      std::string key = DataToString(input_datas[i]->Raw<uint64_t>() + j,
                                     offset);
      if (unique_map.find(key) == unique_map.end()) {
        unique_map[key] = unique_cnt++;
        for (size_t k = 0; k < offset; ++k) {
          unique_result.push_back(input_datas[i]->Raw<uint64_t>()[j + k]);
        }
      }
    }
  }

  std::string output_name = OutputName(node_def, 0);
  Tensor* output_tensor = nullptr;
  data_shape[0] = unique_map.size();
  TensorShape result_shape(data_shape);
  ctx->Allocate(output_name, result_shape, data_type,
                &output_tensor);
  std::copy(unique_result.begin(), unique_result.end(),
            output_tensor->Raw<uint64_t>());

  // add merge index for next merge
  int32_t split_num = input_datas.size();
  for (int32_t i = 0; i < split_num; ++i) {
    std::string output_name = OutputName(node_def, i + 1);
    Tensor* output_tensor = nullptr;
    size_t size = input_datas[i]->Shape().Dims()[0];
    ctx->Allocate(output_name, {size}, kInt32,
                  &output_tensor);
    for (size_t j = 0; j < size; ++j) {
      std::string key = DataToString(
          input_datas[i]->Raw<uint64_t>() + j * offset,
          offset);
      output_tensor->Raw<int32_t>()[j] =
          unique_map[key];
    }
  }
}

REGISTER_OP_KERNEL("GP_UNIQUE_MERGE", GPUniqueMerge);

}  // namespace euler
