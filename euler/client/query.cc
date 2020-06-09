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

#include "euler/client/query.h"

#include <string>

#include "euler/core/framework/op_kernel.h"

namespace euler {

Query::Query(const std::string& gremlin) {
  gremlin_ = gremlin;
  ctx_ = std::make_shared<OpKernelContext>();
}

Query::Query(
    const std::string& op_name,
    const std::string& alias,
    int32_t output_num,
    const std::vector<std::string>& input_tensor_names,
    const std::vector<std::string>& norm_attr_names): Query(op_name) {
  op_name_ = op_name;
  output_num_ = output_num;
  alias_ = alias;
  input_tensor_names_ = input_tensor_names;
  norm_attr_names_ = norm_attr_names;
}

Tensor* Query::AllocInput(const std::string& name,
                          const TensorShape& shape,
                          const DataType& type) {
  Tensor* tensor = nullptr;
  ctx_->Allocate(name, shape, type, &tensor);
  return tensor;
}

std::unordered_map<std::string, Tensor*>
Query::GetResult(
    const std::vector<std::string>& result_names) {
  std::unordered_map<std::string, Tensor*> results;
  for (auto& result_name : result_names) {
    Tensor* output = nullptr;
    if (ctx_->tensor(result_name, &output).ok()) {
      results[result_name] = output;
    } else {
      EULER_LOG(ERROR) << "result " << result_name << " not exist!";
    }
  }
  return results;
}

Tensor* Query::GetResult(const std::string& result_name) {
  Tensor* t = nullptr;
  ctx_->tensor(result_name, &t);
  return t;
}

}  // namespace euler
