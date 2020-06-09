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


#ifndef EULER_CORE_FRAMEWORK_TENSOR_H_
#define EULER_CORE_FRAMEWORK_TENSOR_H_

#include <string>

#include "euler/common/logging.h"
#include "euler/common/refcount.h"
#include "euler/core/framework/tensor_shape.h"
#include "euler/core/framework/allocator.h"
#include "euler/core/framework/types.h"

namespace euler {

class Buffer : public RefCounted {
 public:
  Buffer(Allocator* allocator, size_t size);
  Buffer(Allocator* allocator, void* begin, size_t size, bool own);
  Buffer(Allocator* allocator, void* begin, size_t size, Buffer* parent);
  ~Buffer();

  void* begin() const {
    return begin_;
  }

  size_t size() const {
    return size_;
  }

  bool own() const {
    return own_;
  }

 private:
  RefCountedPtr<Allocator> allocator_;
  void* begin_;
  size_t size_;
  bool own_;
  RefCountedPtr<Buffer> parent_;
};

class Tensor {
 public:
  Tensor();
  Tensor(Allocator* allocator, const TensorShape& shape, DataType type);
  Tensor(const TensorShape& shape, DataType type, Buffer* buffer);
  Tensor(Allocator* allocator, const TensorShape& shape, DataType type,
         void* data, bool own);

  ~Tensor();

  bool Initialized() const {
    return state_.get() != nullptr;
  }

  template <typename T>
  T* Raw() const {
    EULER_CHECK(Initialized()) << "Tensor Not Initialized";
    return reinterpret_cast<T*>(state_->buffer->begin());
  }

  const TensorShape& Shape() const {
    EULER_CHECK(Initialized());
    return state_->shape;
  }

  int NumElements() const {
    EULER_CHECK(Initialized());
    return state_->shape.NumElements();
  }

  DataType Type() const {
    EULER_CHECK(Initialized());
    return state_->type;
  }

  size_t TotalBytes() const {
    return state_->shape.NumElements() * SizeOfType(state_->type);
  }

  template <typename T>
  T Scalar() const {
    EULER_CHECK(Shape().IsScalar()) << "tensor is not scalar";
    return *(Raw<T>());
  }

  Buffer* GetBuffer() const {
    EULER_CHECK(Initialized());
    return state_->buffer.get();
  }

 private:
  class State : public RefCounted {
   public:
    State(const RefCountedPtr<Buffer>& buffer_,
          const TensorShape& shape_, DataType type_)
      : buffer(buffer_), shape(shape_), type(type_) {}
    RefCountedPtr<Buffer> buffer;
    TensorShape shape;
    DataType type;
  };
  RefCountedPtr<State> state_;
};

template <>
inline std::string Tensor::Scalar<std::string>() const {
  if (Shape().Size() == 0) {
    return "";
  }

  return std::string(reinterpret_cast<char*>(Raw<int8_t>()),
                     Shape().NumElements());
}

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_TENSOR_H_
