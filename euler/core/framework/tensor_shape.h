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


#ifndef EULER_CORE_FRAMEWORK_TENSOR_SHAPE_H_
#define EULER_CORE_FRAMEWORK_TENSOR_SHAPE_H_

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace euler {

class TensorShape {
 public:
  TensorShape(): num_elements_(1), dims_(0) { }

  TensorShape(const std::vector<size_t>& dims): dims_(dims) {  // NOLINT
    ComputeNumElements();
  }

  TensorShape(std::vector<size_t>&& dims)  // NOLINT
      : dims_(std::move(dims)) {
    ComputeNumElements();
  }

  TensorShape(std::initializer_list<size_t> dims)  // NOLINT
      : dims_(dims) {
    ComputeNumElements();
  }

  TensorShape(const TensorShape& other)
      : num_elements_(other.num_elements_),
        dims_(other.dims_) {
  }

  const std::vector<std::size_t>& Dims() const {
    return dims_;
  }

  size_t Size() const {
    return dims_.size();
  }

  size_t NumElements() const {
    return num_elements_;
  }

  bool IsScalar() const {
    return dims_.empty();
  }

  void Set(size_t id, size_t dim) {
    dims_[id] = dim;
    ComputeNumElements();
  }

  size_t operator[](size_t id) const {
    return dims_[id];
  }

  bool operator==(const TensorShape& rhs) const {
    if (dims_.size() != rhs.dims_.size()) {
      return false;
    }
    for (size_t i = 0; i < dims_.size(); i++) {
      if (dims_[i] !=  rhs.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const TensorShape& rhs) const {
    return !(*this == rhs);
  }

  std::string DebugString() const {
    std::string ret = "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (i != 0) {
        ret += ", ";
      }
      ret += std::to_string(dims_[i]);
    }
    ret += "]";
    return ret;
  }

 private:
  void ComputeNumElements() {
    num_elements_ = 1;
    for (size_t dim : dims_) {
      num_elements_ *= dim;
    }
  }

  size_t num_elements_;
  std::vector<size_t> dims_;
};


inline std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << "(";
  auto dims = shape.Dims();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << dims[i];
  }
  os << ")";
  return os;
}


}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_TENSOR_SHAPE_H_
