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

#ifndef EULER_CORE_FRAMEWORK_EXECUTOR_H_
#define EULER_CORE_FRAMEWORK_EXECUTOR_H_

#include <atomic>
#include <vector>

#include "euler/core/dag/dag.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/common/mutex.h"

namespace euler {

class ThreadPool;

class Executor {
 public:
  typedef std::function<void()> DoneCallback;

  Executor(DAG* dag, ThreadPool* thread_pool, OpKernelContext* ctx);
  void Run();
  void Run(DoneCallback callback);

 private:
  void RunInternal();
  void Run(DAGNode* node);
  void RunDone(DAGNode* node);

 private:
  DAG* dag_;
  ThreadPool* thread_pool_;
  OpKernelContext* ctx_;
  std::vector<std::atomic<int>> ref_;
  DoneCallback callback_;
  std::atomic<int> remain_node_;
};

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_EXECUTOR_H_
