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

#include "euler/core/framework/tensor_util.h"

#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

namespace euler {

namespace {

Status EncodeString(const Tensor& tensor, TensorProto* proto) {
  std::string buffer;
  auto ptr = tensor.Raw<std::string*>();
  for (int i = 0; i < tensor.NumElements(); ++i) {
    uint32_t len = (*ptr)->size();
    buffer.append(reinterpret_cast<char*>(&len), sizeof(len));
    buffer.append(**ptr);
    ++ptr;
  }
  proto->set_tensor_content(buffer);
  return Status::OK();
}

Status DecodeString(const TensorProto& proto, Tensor* tensor) {
  auto pstr = tensor->Raw<std::string*>();
  auto start = proto.tensor_content().c_str();
  auto end = start + proto.tensor_content().size();
  for (int i = 0; i < tensor->NumElements(); ++i) {
    if (start + sizeof(uint32_t) > end) {
      return Status::Internal("Invalid tensor proto");
    }
    uint32_t len = *reinterpret_cast<const uint32_t*>(start);
    start += sizeof(uint32_t);
    (*pstr)->resize(len);
    if (start + len > end) {
      return Status::Internal("Invalid tensor proto");
    }
    (*pstr)->assign(start, len);
    pstr++;
    start += len;
  }
  return Status::OK();
}

}  // namespace

Status Encode(const Tensor& tensor, TensorProto* proto) {
  if (!tensor.Initialized()) {
    return Status::FailedPrecondition("Tensor should be properly initialized");
  }

  proto->set_dtype(static_cast<DataTypeProto>(tensor.Type()));
  for (auto& dim : tensor.Shape().Dims()) {
    proto->mutable_tensor_shape()->mutable_dims()->Add(dim);
  }

  if (tensor.Type() == DataType::kString) {
    return EncodeString(tensor, proto);
  }

  auto ptr = tensor.Raw<char>();
  auto bytes = tensor.TotalBytes();
  auto content = proto->mutable_tensor_content();
  content->assign(ptr, bytes);
  return Status::OK();
}

Status Decode(const TensorProto& proto, Tensor* tensor) {
  if (!tensor->Initialized()) {
    return Status::FailedPrecondition("Tensor should be properly initialized");
  }

  if (tensor->Type() == DataType::kString) {
    return DecodeString(proto, tensor);
  }

  auto bytes = tensor->TotalBytes();
  auto ptr = tensor->Raw<char>();
  auto& content = proto.tensor_content();
  if (content.size() != bytes) {
    return Status::Internal("Dismatched proto and tensor output");
  }
  memcpy(ptr, content.c_str(), bytes);
  return Status::OK();
}

TensorShape ProtoToTensorShape(const TensorShapeProto& proto) {
  std::vector<size_t> dims(proto.dims_size());
  std::copy(proto.dims().begin(), proto.dims().end(),
            dims.begin());
  return TensorShape(dims);
}

DataType ProtoToDataType(DataTypeProto proto) {
  switch (proto) {
    case DataTypeProto::DT_INT8:
      return DataType::kInt8;
    case  DataTypeProto::DT_INT16:
      return DataType::kInt16;
    case DataTypeProto::DT_INT32:
      return DataType::kInt32;
    case DataTypeProto::DT_INT64:
      return DataType::kInt64;
    case DataTypeProto::DT_UINT8:
      return DataType::kUInt8;
    case  DataTypeProto::DT_UINT16:
      return DataType::kUInt16;
    case DataTypeProto::DT_UINT32:
      return DataType::kUInt32;
    case DataTypeProto::DT_UINT64:
      return DataType::kUInt64;
    case DataTypeProto::DT_FLOAT:
      return DataType::kFloat;
    case DataTypeProto::DT_DOUBLE:
      return DataType::kDouble;
    case DataTypeProto::DT_BOOL:
      return DataType::kBool;
    case DataTypeProto::DT_STRING:
      return DataType::kString;
    default:
      return DataType::kFloat;
  }
}

}  // namespace euler
