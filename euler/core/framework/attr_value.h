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

#ifndef EULER_CORE_FRAMEWORK_ATTR_VALUE_H_
#define EULER_CORE_FRAMEWORK_ATTR_VALUE_H_

#include <functional>
#include <string>
#include <vector>

#include "euler/core/framework/types.h"
#include "euler/common/status.h"
#include "euler/core/framework/tensor_shape.h"

namespace euler {

struct AttrValue {
  enum Type {
    kNone,
    kString,
    kInt,
    kFloat,
    kBool,
    kDataType,
    kTensorShape,
    kDataTypeList
  };

  Type attr_type;
  std::string s;
  int64_t i;
  float f;
  bool b;
  DataType type;
  TensorShape shape;
  std::vector<DataType> type_list;
};

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_ATTR_VALUE_H_
