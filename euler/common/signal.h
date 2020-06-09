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

#ifndef EULER_COMMON_SIGNAL_H_
#define EULER_COMMON_SIGNAL_H_

#include <assert.h>
#include <stdint.h>

#include <atomic>              // NOLINT

#include <condition_variable>  // NOLINT

#include "euler/common/mutex.h"
#include "euler/common/macros.h"

namespace euler {

class Signal {
 public:
  Signal() : notified_(0) { }
  ~Signal() {
    MutexLock l(&mu_);
  }

  void Notify() {
    MutexLock l(&mu_);
    assert(!Notified());
    notified_.store(true, std::memory_order_release);
    cv_.SignalAll();
  }

  bool Notified() const {
    return notified_.load(std::memory_order_acquire);
  }

  void Wait() {
    if (!Notified()) {
      MutexLock l(&mu_);
      while (!Notified()) {
        cv_.Wait(&l);
      }
    }
  }

 private:
  friend bool WaitWithTimeout(Signal* n, int64_t timeout_in_us);

  bool WaitWithTimeout(int64_t timeout_in_us) {
    bool notified =  Notified();
    if (!notified) {
      MutexLock l(&mu_);
      do {
        notified = Notified();
      } while (!notified && !cv_.WaitFor(&l, timeout_in_us));
    }
    return notified;
  }

 private:
  Mutex mu_;
  CondVar cv_;
  std::atomic<bool> notified_;

  DISALLOW_COPY_AND_ASSIGN(Signal);
};

inline bool WaitWithTimeout(Signal* s, int64_t timeout_in_us) {
  return s->WaitWithTimeout(timeout_in_us);
}

}  // namespace euler

#endif  // EULER_COMMON_SIGNAL_H_
