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


#ifndef EULER_COMMON_REFCOUNT_H_
#define EULER_COMMON_REFCOUNT_H_

#include <assert.h>

#include <atomic>
#include <utility>

namespace euler {

class RefCounted {
 public:
  RefCounted();

  explicit RefCounted(int ref);

  void Ref() const;

  bool Unref() const;

  bool RefCountIsOne() const;

 protected:
  virtual ~RefCounted();

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

inline RefCounted::RefCounted() : ref_(1) {}

inline RefCounted::RefCounted(int ref) : ref_(ref) {}

inline RefCounted::~RefCounted() { }

inline void RefCounted::Ref() const {
  assert(ref_.load() >= 1);
  ref_.fetch_add(1, std::memory_order_relaxed);
}

inline bool RefCounted::Unref() const {
  assert(ref_.load() > 0);
  if (RefCountIsOne() || ref_.fetch_sub(1) == 1) {
    delete this;
    return true;
  } else {
    return false;
  }
}

inline bool RefCounted::RefCountIsOne() const {
  return (ref_.load(std::memory_order_acquire) == 1);
}

template<typename T>
class RefCountedPtr {
 public:
  RefCountedPtr() : ptr_(nullptr) {}

  explicit RefCountedPtr(T* ptr) : ptr_(ptr) {
    Ref();
  }

  RefCountedPtr(const RefCountedPtr& rptr) : ptr_(rptr.ptr_) {
    Ref();
  }

  RefCountedPtr(RefCountedPtr&& rptr) : ptr_(rptr.ptr_) {
    rptr.ptr_ = nullptr;
  }

  RefCountedPtr& operator=(T* ptr) {
    Unref();
    ptr_ = ptr;
    Ref();
    return *this;
  }

  RefCountedPtr& operator=(const RefCountedPtr& rptr) {
    Unref();
    ptr_ = rptr.ptr_;
    Ref();
    return *this;
  }

  RefCountedPtr& operator=(RefCountedPtr&& rptr) {
    std::swap(ptr_, rptr.ptr_);
    return *this;
  }

  ~RefCountedPtr() {
    if (ptr_ != nullptr) {
      ptr_->Unref();
    }
  }

  std::add_lvalue_reference<T> operator*() const {
    return *ptr_;
  }

  T* operator->() const {
    return ptr_;
  }

  T* get() const {
    return ptr_;
  }

  template <typename... Targs>
  static RefCountedPtr Create(Targs&&... args) {
    return RefCountedPtr(new T(std::forward<Targs>(args)...), 0);
  }

 private:
  RefCountedPtr(T* ptr, int x) : ptr_(ptr) {
    (void)x;
  }
  void Ref() {
    if (ptr_ != nullptr) {
      ptr_->Ref();
    }
  }

  void Unref() {
    if (ptr_ != nullptr) {
      ptr_->Unref();
    }
  }

  T* ptr_;
};

}  // namespace euler

#endif  // EULER_COMMON_REFCOUNT_H_
