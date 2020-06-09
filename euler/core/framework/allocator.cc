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


#include "euler/core/framework/allocator.h"

namespace euler {

Allocator* AllocatorManager::Get(const std::string& name,
                                 std::function<Allocator*()> creator) {
  MutexLock l(&mu_);
  auto iter = allocators_.find(name);
  if (iter != allocators_.end()) {
    return iter->second.get();
  }
  Allocator* new_allocator = creator();
  allocators_[name] = RefCountedPtr<Allocator>(new_allocator);
  new_allocator->Unref();
  return new_allocator;
}

}  // namespace euler
