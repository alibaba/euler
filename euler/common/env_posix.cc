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

#include <dlfcn.h>

#include <thread>  // NOLINT
#include <atomic>
#include <string>
#include <deque>
#include <vector>

#include "euler/common/env.h"
#include "euler/common/macros.h"
#include "euler/common/mutex.h"
#include "euler/common/logging.h"

namespace euler {

namespace {

class StdThread: public Thread {
 public:
  StdThread(const std::string& name, ThreadFunc fn)
      : name_(name), thread_(fn) {
  }

  ~StdThread() override {
    if (Joinable()) {
      Join();
    }
  }

  bool Joinable() override {
    return thread_.joinable();
  }

  void Join() override {
    if (Joinable()) {
      thread_.join();
    }
  }

 private:
  std::string name_;
  std::thread thread_;
};

template <typename T>
class JobQueue {
 public:
  T Pop() {
    MutexLock l(&mu_);
    cv_.Wait(&l , [this] () { return shutdown_.load() || !jobs_.empty(); });
    if (shutdown_.load()) {
      return default_element_;
    }
    auto element = jobs_.front();
    jobs_.pop_front();
    return element;
  }

  void Push(const T& element) {
    MutexLock l(&mu_);
    jobs_.push_back(element);
    cv_.SignalAll();
  }

  void Shutdown() {
    MutexLock l(&shutdown_mu_);
    if (shutdown_.load()) {
      return;
    }
    shutdown_.store(true, std::memory_order_release);
    cv_.SignalAll();
  }

 private:
  std::atomic<bool> shutdown_;
  Mutex mu_;
  CondVar cv_;
  Mutex shutdown_mu_;
  T default_element_;
  std::deque<T> jobs_;
};

class StdThreadPool: public ThreadPool {
 public:
  StdThreadPool(const std::string& name, int num_threads)
      : shutdown_(false), counter_(0) {
    for (int i = 0; i < num_threads; ++i) {
      job_queues_.emplace_back(new JobQueue<ThreadFunc>());
    }
    for (int i = 0; i < num_threads; ++i) {
      threads_.emplace_back(new StdThread(name, [this, i] () { Loop(i); }));
    }
  }

  ~StdThreadPool() override {
    Shutdown();
  }

  void Schedule(ThreadFunc fn) override {
    job_queues_[++counter_ % job_queues_.size()]->Push(fn);
  }

  void Shutdown() override {
    MutexLock l(&shutdown_mu_);
    if (shutdown_.load()) {
      return;
    }

    shutdown_.store(true, std::memory_order_release);

    for (auto& job_queue : job_queues_) {
      job_queue->Shutdown();
    }

    for (auto& thread : threads_) {
      thread->Join();
    }
  }

  void Join() override {
    for (auto& thread : threads_) {
      thread->Join();
    }
  }

 private:
  void Loop(int i) {
    auto& job_queue = job_queues_[i % job_queues_.size()];
    while (!shutdown_.load(std::memory_order_acquire)) {
      ThreadFunc fn = job_queue->Pop();
      if (shutdown_.load()) {
        break;
      }
      fn();
    }
  }

 private:
  std::atomic<bool> shutdown_;
  std::atomic<size_t> counter_;
  Mutex shutdown_mu_;
  std::vector<std::unique_ptr<StdThread>> threads_;
  std::vector<std::unique_ptr<JobQueue<ThreadFunc>>> job_queues_;
};

}  // namespace

class PosixEnv: public Env {
 public:
  Thread* StartThread(const std::string& name, ThreadFunc fn) override;
  ThreadPool* StartThreadPool(
      const std::string& name, int num_threads) override;

  Status LoadLibrary(const char* library_filename, void** handle) override;

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override;
};

Thread* PosixEnv::StartThread(const std::string &name, ThreadFunc fn) {
  return new StdThread(name, fn);
}

ThreadPool* PosixEnv::StartThreadPool(
    const std::string &name, int num_threads) {
  return new StdThreadPool(name, num_threads);
}

Status PosixEnv::LoadLibrary(const char* library_filename, void** handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    return Status::NotFound(dlerror());
  }
  return Status::OK();
}

Status PosixEnv::GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) {
  *symbol = dlsym(handle, symbol_name);
  if (!*symbol) {
    return Status::NotFound(dlerror());
  }
  return Status::OK();
}

#if defined(POSIX)

Env* Env::Default() {
  return new PosixEnv;
}

#endif  // defined(POSIX)

}  // namespace euler
