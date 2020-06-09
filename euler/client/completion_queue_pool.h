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

#ifndef EULER_CLIENT_COMPLETION_QUEUE_POOL_H_
#define EULER_CLIENT_COMPLETION_QUEUE_POOL_H_

#include <vector>

#include "grpcpp/grpcpp.h"

namespace euler {

class GrpcCQTag {
 public:
  virtual void OnCompleted(bool ok) = 0;
};

class CompletionQueuePool {
 public:
  static CompletionQueuePool *GetInstance() {
    static CompletionQueuePool
      completion_queue_pool(std::thread::hardware_concurrency() * 2);
    return &completion_queue_pool;
  }

  CompletionQueuePool(CompletionQueuePool const&) = delete;
  void operator=(CompletionQueuePool const&) = delete;

  grpc::CompletionQueue *NextCompletionQueue() {
    std::lock_guard<std::mutex> lock(mu_);
    return threads_[next_round_robin_assignment_++ %
                    threads_.size()].completion_queue();
  }

 private:
  explicit CompletionQueuePool(size_t thread_count)
      : threads_(thread_count), next_round_robin_assignment_(0) { }

  class GrpcThread {
   public:
    GrpcThread() : thread_(&GrpcThread::CompleteGrpcCall, this) { }

    void CompleteGrpcCall() {
      void* tag;
      bool ok = false;

      while (completion_queue_.Next(&tag, &ok)) {
        GrpcCQTag* cq_tag = static_cast<GrpcCQTag*>(tag);
        cq_tag->OnCompleted(ok);
      }
    }

    ~GrpcThread() {
      completion_queue_.Shutdown();
      thread_.join();
    }

    grpc::CompletionQueue *completion_queue() { return &completion_queue_; }

   private:
    grpc::CompletionQueue completion_queue_;
    std::thread thread_;
  };

  std::vector<GrpcThread> threads_;
  size_t next_round_robin_assignment_;
  std::mutex mu_;
};


}  // namespace euler
#endif  // EULER_CLIENT_COMPLETION_QUEUE_POOL_H_
