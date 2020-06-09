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


#ifndef EULER_CORE_FRAMEWORK_OP_KERNEL_H_
#define EULER_CORE_FRAMEWORK_OP_KERNEL_H_

#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

#include "euler/core/framework/types.h"
#include "euler/common/status.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/framework/tensor_shape.h"
#include "euler/core/framework/attr_value.h"
#include "euler/core/framework/tensor.pb.h"
#include "euler/common/logging.h"

namespace euler {

class DAGNodeProto;
class OpKernelContext;

class OpKernel {
 public:
  // OpKernel can not have any state and so is thread-safe,
  // all state should be persistent in OpKernelContext.
  // An OpKernel can be cached and reused and used concurrently.
  explicit OpKernel(const std::string& name);
  virtual ~OpKernel();

  // Computation in OpKernel can be synchronous or asynchronous.
  // All OpKernel's Compute() methods must be non-stateful and
  // thread-safe, the side effect of the Compute shoud be
  // persisted in the OpKernelContext object.
  //
  // An asynchronous OpKernel must subclass AsyncOpKernel.
  virtual void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx) = 0;

  const std::string& name() const { return name_; }

 private:
  std::string name_;
};

class AsyncOpKernel: public OpKernel {
 public:
  using OpKernel::OpKernel;

  typedef std::function<void()> DoneCallback;

  // An asynchronous OpKernel can run in synchronous mode too,
  // which just call and wait the AsyncCompute() method to complete.
  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
  virtual void AsyncCompute(const DAGNodeProto& node_def,
                            OpKernelContext* ctx, DoneCallback callback) = 0;
};

class OpKernelContext {
 public:
  // OpKernelContext maintains the state in the OpKernel computation.
  //
  // An OpKernelContext object lives through the entire execution of
  // the op graph.
  //
  // All OpKernel retrieve their tensor inputs from the OpKernelContext
  // and allocate their output in the OpKernelContext.
  ~OpKernelContext();

  Status Allocate(const std::string& name, const TensorShape& shape,
                  DataType type, Tensor** tensor);

  Status Allocate(const TensorProto& tensor_def);

  Status Allocate(const std::string& name, const TensorProto& tensor_def);

  Status AddAlias(const std::string& name, Tensor* tensor);

  Status tensor(const std::string& name, Tensor** tensor);

  // Remove an alias but do not deallocate tensor
  Status RemoveAlias(const std::string& name);

  // !!! Attention: iff tensor has alias, it's not safe to be deallocated
  Status Deallocate(const std::string& name);

 private:
  Mutex mu_;
  std::unordered_map<std::string, Tensor*> tensor_map_;  // Guard by mu_
};

#define REGISTER_OP_KERNEL(name, cls) \
  REGISTER_OP_KERNEL_UNIQ_HELPER(__COUNTER__, name, cls)

#define REGISTER_OP_KERNEL_UNIQ_HELPER(counter, name, cls) \
  REGISTER_OP_KERNEL_UNIQ(counter, name, cls)

#define REGISTER_OP_KERNEL_UNIQ(counter, name, cls)           \
  static ::euler::OpKernelRegistrar                           \
      registrar__##counter##__obj(                            \
          name,                                               \
          [] (const std::string& op) -> euler::OpKernel* {    \
            return new cls(op);                               \
          });

class OpKernelRegistrar {
 public:
  typedef OpKernel* (*Factory)(const std::string& name);

  OpKernelRegistrar(const std::string& name, Factory factory) {
    Register(name, factory);
  }

 private:
  void Register(const std::string& name, Factory factory);
};

void* GlobalKernelRegistry();

Status LookupOpKernel(const std::string& name);
Status CreateOpKernel(const std::string& name, OpKernel** kernel);

std::string OutputName(const std::string& name, int i);
std::string OutputName(const DAGNodeProto& node_def, int i);

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_OP_KERNEL_H_
