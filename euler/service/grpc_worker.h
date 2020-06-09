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

#ifndef EULER_SERVICE_GRPC_WORKER_H_
#define EULER_SERVICE_GRPC_WORKER_H_

#include <memory>

#include "euler/service/worker.h"

namespace euler {

class GrpcWorker: public Worker {
 public:
  explicit GrpcWorker(WorkerEnv* env);

#define DECLARE_METHOD(Method)                                  \
  void Method##Async(const Method##Request* request,            \
                     Method##Reply* reply, Callback) override;  \

  DECLARE_METHOD(Ping);
  DECLARE_METHOD(Execute);

#undef DECLARE_METHOD  // DECLARE_METHOD
};

std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* worker_env);

}  // namespace euler

#endif  // EULER_SERVICE_GRPC_WORKER_H_
