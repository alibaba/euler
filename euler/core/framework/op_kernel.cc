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


#include "euler/core/framework/op_kernel.h"

#include <stdlib.h>
#include <unordered_set>
#include <unordered_map>

#include "euler/common/logging.h"
#include "euler/common/signal.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/types.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/framework/tensor_util.h"

namespace euler {

namespace {

class OpKernelCache {
 public:
  ~OpKernelCache() {
    MutexLock lock(&mu_);
    for (auto& item : cache_) {
      delete item.second;
    }
    cache_.clear();
  }

  void Get(const std::string& op_name, OpKernel** op) {
    *op = nullptr;
    MutexLock lock(&mu_);
    auto it = cache_.find(op_name);
    if (it != cache_.end()) {
      *op = it->second;
    }
  }

  void Cache(OpKernel* op) {
    MutexLock lock(&mu_);
    cache_.insert({op->name(), op});
  }

 private:
  Mutex mu_;
  std::unordered_map<std::string, OpKernel*> cache_;
};

OpKernelCache* GlobalOpKernelCache() {
  static OpKernelCache instance;
  return &instance;
}

}  // namespace

OpKernel::OpKernel(const std::string& name): name_(name) { }
OpKernel::~OpKernel() { }

void AsyncOpKernel::Compute(const DAGNodeProto& node_def,
                            OpKernelContext* ctx) {
  Signal sig;
  AsyncCompute(node_def, ctx, [&sig] () { sig.Notify(); });
  sig.Wait();
}

OpKernelContext::~OpKernelContext() {
  MutexLock l(&mu_);
  std::unordered_set<Tensor*> set;
  for (auto& item : tensor_map_) {
    if (set.find(item.second) == set.end()) {
      set.insert(item.second);
      delete item.second;
    }
  }
  tensor_map_.clear();
}

class MallocAllocator: public Allocator {
 public:
  void* Allocate(size_t size) override {
    return malloc(size);
  }
  void Deallocate(void* ptr) override {
    free(ptr);
  }
};

static Allocator* ContextTensorAllocator() {
  static MallocAllocator allocator;
  return &allocator;
}

Status OpKernelContext::Allocate(const std::string& name,
                                 const TensorShape& shape,
                                 DataType type, Tensor** tensor) {
  *tensor = new Tensor(ContextTensorAllocator(), shape, type);
  if (!(*tensor)->Initialized()) {
    return Status::Internal("Allocate memory for tensor failed, shape:",
                            shape.DebugString(), ", DataType: ", type);
  }
  {
    MutexLock l(&mu_);
    if (!tensor_map_.insert({name, *tensor}).second) {
      delete (*tensor);
      *tensor = nullptr;
      return Status::Internal("Tensor: ", name, " Exists!");
    }
  }
  return Status::OK();
}

Status OpKernelContext::Allocate(const TensorProto& proto) {
  Tensor* tensor = nullptr;
  TensorShape tensor_shape =
      ProtoToTensorShape(proto.tensor_shape());
  auto s = Allocate(proto.name(), tensor_shape,
                    ProtoToDataType(proto.dtype()), &tensor);
  RETURN_IF_ERROR(s);
  return Decode(proto, tensor);
}

Status OpKernelContext::Allocate(const std::string& name,
                                 const TensorProto& proto) {
  Tensor* tensor = nullptr;
  TensorShape tensor_shape =
      ProtoToTensorShape(proto.tensor_shape());
  auto s = Allocate(name, tensor_shape,
                    ProtoToDataType(proto.dtype()), &tensor);
  RETURN_IF_ERROR(s);
  return Decode(proto, tensor);
}


Status OpKernelContext::AddAlias(const std::string& name,
                                 Tensor* tensor) {
  MutexLock l(&mu_);
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    return Status::Internal("Tensor '", name, "' exists!");
  }
  tensor_map_.insert({name, tensor});
  return Status::OK();
}

Status OpKernelContext::tensor(const std::string& name, Tensor** tensor) {
  *tensor = nullptr;
  {
    MutexLock l(&mu_);
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
      return Status::NotFound("Tensor '", name, "' not found!");
    }
    *tensor = it->second;
  }
  return Status::OK();
}

Status OpKernelContext::RemoveAlias(const std::string& name) {
  MutexLock l(&mu_);
  auto it =  tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    tensor_map_.erase(it);
  }
  return Status::OK();
}

Status OpKernelContext::Deallocate(const std::string& name) {
  MutexLock l(&mu_);
  auto it = tensor_map_.find(name);
  if (it != tensor_map_.end()) {
    delete (it->second);
    tensor_map_.erase(it);
  }
  return Status::OK();
}

typedef std::unordered_map<std::string,
                           OpKernelRegistrar::Factory> KernelRegistry;

void* GlobalKernelRegistry() {
  static KernelRegistry* registry = new KernelRegistry;
  return registry;
}

static KernelRegistry* GlobalKernelRegistryTyped() {
  return reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
}

void OpKernelRegistrar::Register(const std::string& name, Factory factory) {
  if (!GlobalKernelRegistryTyped()->insert({name, factory}).second) {
    EULER_LOG(FATAL) << "Register OpKernel '" << name << "' failed!";
  }
}

Status LookupOpKernel(const std::string& name) {
  auto& registry = *GlobalKernelRegistryTyped();
  if (registry.find(name) == registry.end()) {
    return Status::NotFound("No OpKernel '", name, "' registered");
  }
  return Status::OK();
}

Status CreateOpKernel(const std::string& name, OpKernel** kernel) {
  auto& registry = *GlobalKernelRegistryTyped();
  auto it = registry.find(name);
  if (it == registry.end()) {
    return Status::NotFound("No OpKernel '", name, "' registered");
  }

  auto cache = GlobalOpKernelCache();
  cache->Get(name, kernel);
  if (*kernel == nullptr) {
    *kernel = it->second(name);
    cache->Cache(*kernel);
  }

  return Status::OK();
}

std::string OutputName(const std::string& name, int i) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s:%d", name.c_str(), i);
  return buf;
}

std::string OutputName(const DAGNodeProto& node_def, int i) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s:%d", node_def.name().c_str(), i);
  return buf;
}

}  // namespace euler
