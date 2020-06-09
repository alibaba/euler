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

#ifndef EULER_COMMON_MUTEX_H_
#define EULER_COMMON_MUTEX_H_

#include <mutex>               // NOLINT
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT

#include "euler/common/macros.h"

namespace euler {

class CondVar;

class Mutex {
 public:
  Mutex() = default;
  ~Mutex() = default;

  void Lock() { mu_.lock(); }
  void Unlock() { mu_.unlock(); }
  void AssertHeld() { }

 private:
  friend class CondVar;
  std::mutex mu_;

  DISALLOW_COPY_AND_ASSIGN(Mutex);
};

class MutexLock {
 public:
  explicit MutexLock(Mutex* mu) : mu_(mu)  {
    this->mu_->Lock();
  }

  ~MutexLock() {
    mu_->Unlock();
  }

 private:
  friend class CondVar;
  Mutex* const mu_;

  DISALLOW_COPY_AND_ASSIGN(MutexLock);
};

class CondVar {
 public:
  CondVar() { }
  ~CondVar() { }

  void Wait(MutexLock* l) {
    std::unique_lock<std::mutex> lock(l->mu_->mu_, std::adopt_lock);
    cv_.wait(lock);
    lock.release();
  }

  template <class Predicate>
  void Wait(MutexLock* l, Predicate pred) {
    std::unique_lock<std::mutex> lock(l->mu_->mu_, std::adopt_lock);
    cv_.wait(lock, pred);
    lock.release();
  }

  bool WaitFor(MutexLock* l, int64_t timeout_in_us) {
    std::unique_lock<std::mutex> lock(l->mu_->mu_, std::adopt_lock);
    auto ret = cv_.wait_for(lock, std::chrono::microseconds(timeout_in_us));
    lock.release();

    return ret == std::cv_status::timeout;
  }

  void Signal() { cv_.notify_one(); }
  void SignalAll() { cv_.notify_all(); }

 private:
  std::condition_variable cv_;

  DISALLOW_COPY_AND_ASSIGN(CondVar);
};

}  // namespace euler

#endif  // EULER_COMMON_MUTEX_H_
