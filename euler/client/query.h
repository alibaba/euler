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

#ifndef EULER_CLIENT_QUERY_H_
#define EULER_CLIENT_QUERY_H_

#include <unordered_map>
#include <memory>
#include <vector>
#include <string>

#include "euler/common/logging.h"

namespace euler {

class OpKernelContext;
class Tensor;
class TensorShape;
enum DataType: int32_t;

class Query {
 public:
  explicit Query(const std::string& gremlin);

  Query(
      const std::string& op_name,
      const std::string& alias,
      int32_t output_num,
      const std::vector<std::string>& input_tensor_names,
      const std::vector<std::string>& norm_attr_names);

  Tensor* AllocInput(const std::string& name,
                     const TensorShape& shape,
                     const DataType& type);

  std::unordered_map<std::string, Tensor*> GetResult(
      const std::vector<std::string>& result_names);

  Tensor* GetResult(const std::string& result_name);

  bool SingleOpQuery() {
    return !op_name_.empty();
  }

 private:
  std::shared_ptr<OpKernelContext> ctx_;
  std::string gremlin_;

  std::string op_name_;
  int32_t output_num_;
  std::string alias_;
  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> norm_attr_names_;

  friend class QueryProxy;
};
}  // namespace euler

#endif  // EULER_CLIENT_QUERY_H_
