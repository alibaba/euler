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

#include <cstring>

#include "euler/core/framework/tensor.h"
#include "euler/common/logging.h"

namespace euler {

Buffer::Buffer(Allocator* allocator, size_t size)
  : allocator_(allocator), begin_(allocator->Allocate(size)),
    size_(size), own_(true), parent_(nullptr) {
  // memset(begin_, 0x00, size_);
}

Buffer::Buffer(Allocator* allocator, void* begin, size_t size, bool own)
  : allocator_(allocator), begin_(begin),
    size_(size), own_(own), parent_(nullptr) {}

Buffer::Buffer(Allocator* allocator, void* begin, size_t size, Buffer* parent)
  : allocator_(allocator), begin_(begin),
    size_(size), own_(false), parent_(parent) {}

Buffer::~Buffer() {
  if (own_) {
    allocator_->Deallocate(begin_);
  }
}

Tensor::Tensor(): state_(nullptr) {}

Tensor::Tensor(Allocator* allocator, const TensorShape& shape, DataType type)
  : state_(RefCountedPtr<State>::Create(
        RefCountedPtr<Buffer>::Create(
            allocator, shape.NumElements() * SizeOfType(type)),
        shape, type)) {
  if (type == DataType::kString) {
    auto p = Raw<std::string*>();
    for (int i = 0; i < NumElements(); ++i) {
      *p++ = new std::string;
    }
  }
}

Tensor::Tensor(const TensorShape& shape, DataType type, Buffer* buffer)
  : state_(RefCountedPtr<State>::Create(
        RefCountedPtr<Buffer>(buffer), shape, type)) { }

Tensor::Tensor(Allocator* allocator, const TensorShape& shape, DataType type,
               void* data, bool own)
  : state_(RefCountedPtr<State>::Create(
        RefCountedPtr<Buffer>::Create(
            allocator, data, shape.NumElements() * SizeOfType(type),
            own), shape, type)) { }

Tensor::~Tensor() {
  if (Initialized() && state_->type == DataType::kString &&
      state_->buffer->own()) {
    auto p = Raw<std::string*>();
    for (int i = 0; i < NumElements(); ++i) {
      delete *p;
      p++;
    }
  }
}

}  // namespace euler
