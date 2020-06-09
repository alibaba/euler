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

#include "euler/service/grpc_worker.h"

#include <vector>
#include <string>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/executor.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/framework/tensor_util.h"
#include "euler/common/logging.h"
#include "euler/core/api/api.h"

namespace euler {

GrpcWorker::GrpcWorker(WorkerEnv* env): Worker(env) {
}

void GrpcWorker::PingAsync(const PingRequest* request,
                           PingReply* reply, Callback done) {
  EULER_LOG(DEBUG) << "Received request: " << request->content();
  reply->set_content("Pong");
  done(Status::OK());
}

void GrpcWorker::ExecuteAsync(const ExecuteRequest* request,
                              ExecuteReply* reply, Callback done) {
  auto context = new OpKernelContext;
  for (auto& input : request->inputs()) {
    auto s = context->Allocate(input);
    if (!s.ok()) {
      auto msg = ToString("Allocate input tensor '", input.name(), "' failed!");
      EULER_LOG(ERROR) << msg;
      done(Status::Internal(msg));

      delete context;
      return;
    }
  }

  auto dag = DAG::NewFromProto(request->graph()).release();
  if (dag == nullptr) {
    auto msg = ToString("Convert graph proto to DAG failed, proto:",
                        request->graph().DebugString());
    EULER_LOG(ERROR) << msg;
    done(Status::Internal(msg));

    delete dag;
    delete context;
    return;
  }

  auto executor = new Executor(dag, env()->compute_pool, context);

  auto callback = [request, reply, done, context, dag, executor] () {
    for (auto& output : request->outputs()) {
      auto outpb = reply->mutable_outputs()->Add();
      outpb->set_name(output);
      Tensor* t = nullptr;
      auto s = context->tensor(output, &t);
      if (!s.ok() || t == nullptr) {
        auto msg = ToString("No output tensor '", output, "'");
        EULER_LOG(ERROR) << msg;
        done(Status::Internal(msg));

        delete dag;
        delete context;
        delete executor;
        return;
      }

      Encode(*t, outpb);
    }

    done(Status::OK());
    delete dag;
    delete context;
    delete executor;
  };

  executor->Run(callback);
}

std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* worker_env) {
  return std::unique_ptr<GrpcWorker>(new GrpcWorker(worker_env));
}

}  // namespace euler
