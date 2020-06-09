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

#ifndef EULER_CORE_FRAMEWORK_TENSOR_UTIL_H_
#define EULER_CORE_FRAMEWORK_TENSOR_UTIL_H_

#include "euler/common/status.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/framework/tensor.pb.h"
#include "euler/core/framework/tensor_shape.h"
#include "euler/core/framework/tensor_shape.pb.h"
#include "euler/core/framework/types.h"
#include "euler/core/framework/types.pb.h"

namespace euler {

Status Encode(const Tensor& tensor, TensorProto* proto);
Status Decode(const TensorProto& proto, Tensor* tensor);
TensorShape ProtoToTensorShape(const TensorShapeProto& proto);
DataType ProtoToDataType(DataTypeProto proto);

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_TENSOR_UTIL_H_
