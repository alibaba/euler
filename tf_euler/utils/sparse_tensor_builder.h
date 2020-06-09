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

#ifndef TF_EULER_UTILS_SPARSE_TENSOR_BUILDER_H_
#define TF_EULER_UTILS_SPARSE_TENSOR_BUILDER_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

template <typename T, int R>
class SparseTensorBuilder {
 public:
  SparseTensorBuilder() : dense_shape_(R) {}

  void emplace(const std::array<int64, R>& index, const T& value) {
    for (size_t i = 0; i < R; ++i) {
      int64 ix = index[i];
      indices_.emplace_back(ix);
      if (ix + 1 > dense_shape_[i]) {
        dense_shape_[i] = ix + 1;
      }
    }
    values_.emplace_back(value);
  }

  Tensor indices() {
    Tensor tensor(DT_INT64, {static_cast<int64>(indices_.size() / R), R});
    CopyVectorToTensor(indices_, &tensor);
    return tensor;
  }

  Tensor values() {
    DataTypeToEnum<T> mapper;
    Tensor tensor(mapper.value, {static_cast<int64>(values_.size())});
    CopyVectorToTensor(values_, &tensor);
    return tensor;
  }

  Tensor dense_shape() {
    Tensor tensor(DT_INT64, {static_cast<int64>(dense_shape_.size())});
    CopyVectorToTensor(dense_shape_, &tensor);
    return tensor;
  }

 private:
  template <typename VT>
  void CopyVectorToTensor(const std::vector<VT>& vec, Tensor* tensor) {
    auto flat = tensor->flat<VT>();
    memcpy(flat.data(), vec.data(), flat.size() * sizeof(VT));
  }

 private:
  std::vector<int64> indices_;
  std::vector<T> values_;
  std::vector<int64> dense_shape_;
};

}  // namespace tensorflow

#endif  // TF_EULER_UTILS_SPARSE_TENSOR_BUILDER_H_
