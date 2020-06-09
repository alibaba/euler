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


#ifndef EULER_CORE_FRAMEWORK_ALLOCATOR_H_
#define EULER_CORE_FRAMEWORK_ALLOCATOR_H_

#include <functional>
#include <unordered_map>
#include <string>

#include "euler/common/mutex.h"
#include "euler/common/refcount.h"
#include "euler/common/singleton.h"

namespace euler {

class Allocator : public RefCounted {
 public:
  virtual void* Allocate(size_t size) = 0;
  virtual void Deallocate(void* buf) = 0;
};

class AllocatorManager : public Singleton<AllocatorManager> {
 public:
  Allocator* Get(const std::string& name,
                 std::function<Allocator*()> creator);
 private:
  Mutex mu_;
  std::unordered_map<std::string, RefCountedPtr<Allocator>> allocators_;
};

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_ALLOCATOR_H_
