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


#ifndef EULER_SERVICE_WORKER_H_
#define EULER_SERVICE_WORKER_H_

#include <functional>

#include "euler/proto/worker.pb.h"
#include "euler/common/status.h"
#include "euler/common/signal.h"
#include "euler/common/env.h"

namespace euler {

struct WorkerEnv {
  Env* env;
  ThreadPool* compute_pool;
};

typedef std::function<void(const Status&)> Callback;

class Worker {
 protected:
  explicit Worker(WorkerEnv* env): env_(env) {
  }

  virtual ~Worker() { }

 public:
#define DECLARE_METHOD(Method)                                          \
  virtual void Method##Async(const Method##Request* request,            \
                             Method##Reply* reply, Callback) = 0;       \
                                                                        \
  Status Method(const Method##Request* request, Method##Reply* reply) { \
    return CallAndWait(&Worker::Method##Async, request, reply);         \
  }

  DECLARE_METHOD(Ping);
  DECLARE_METHOD(Execute);

#undef DECLARE_METHOD  // DECLARE_METHOD

  WorkerEnv* env() const {
    return env_;
  }

 protected:
  WorkerEnv* const env_;

 private:
  template <typename Method, typename Req, typename Resp>
  Status CallAndWait(Method func, const Req* req, Resp* resp) {
    Status ret;
    Signal sig;
    (this->*func)(req, resp, [&ret, &sig](const Status& s) {
      ret = s;
      sig.Notify();
    });
    sig.Wait();
    return ret;
  }
};

}  // namespace euler

#endif  // EULER_SERVICE_WORKER_H_
