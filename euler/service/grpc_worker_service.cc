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


#include "euler/service/grpc_worker_service.h"

#include <string>
#include <vector>
#include <utility>

#include "euler/common/logging.h"
#include "euler/common/macros.h"
#include "euler/service/async_service_interface.h"
#include "euler/service/grpc_call.h"
#include "euler/service/grpc_euler_service.h"
#include "euler/service/grpc_worker.h"

namespace euler {

namespace {

inline ::grpc::Status ToGrpcStatus(const Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  }

  if (s.error_message().size() > 3072) {
    std::string msg = s.error_message().substr(0, 3072) + " ... [truncated]";
    EULER_LOG(ERROR) << "Truncated error message: " << s.DebugString();
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()), msg);
  }
  return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                        s.error_message());
}

class GrpcWorkerService: public AsyncServiceInterface {
  static constexpr const size_t kGrpcWorkerServiceThreadCount = 8;

 public:
  GrpcWorkerService(GrpcWorker* worker, ::grpc::ServerBuilder* builder)
      : is_shutdown_(false) {
    builder->RegisterService(&euler_service_);
    for (size_t i = 0; i < kGrpcWorkerServiceThreadCount; i++) {
      threads_.emplace_back(
          new GrpcWorkerServiceThread(worker, builder, &euler_service_));
    }
  }

  ~GrpcWorkerService() {
    Shutdown();
  }

  void Shutdown() override {
    MutexLock l(&shutdown_mu_);
    if (is_shutdown_) {
      return;
    }

    for (auto& worker_thread : threads_) {
      worker_thread->Shutdown();
    }
    threads_.clear();
    is_shutdown_ = true;
  }

#define ENQUEUE_REQUEST(method)                                 \
  do {                                                          \
    MutexLock l(&shutdown_mu_);                                 \
    if (!is_shutdown_) {                                        \
      Call<GrpcWorkerServiceThread, EulerService::AsyncService, \
           method##Request, method##Reply>::                    \
          EnqueueRequestForMethod(                              \
              euler_service_, cq_.get(),                        \
              static_cast<int>(EulerServiceMethod::k##method),  \
              &GrpcWorkerServiceThread::method##Handler);       \
    }                                                           \
  } while (0)

  void Loop() override {
    for (auto& worker_thread : threads_) {
      worker_thread->Start();
    }
  }

 private:
  class GrpcWorkerServiceThread {
   public:
    GrpcWorkerServiceThread(
        GrpcWorker* worker, ::grpc::ServerBuilder* builder,
        EulerService::AsyncService* euler_service)
        : worker_(worker),
          euler_service_(euler_service),
          is_shutdown_(false) {
      cq_ = builder->AddCompletionQueue();
    }

    void Start() {
      thread_.reset(worker_->env()->env->StartThread(
          "grpc_worker_service", [this]() { HandleRPCsLoop(); }));
    }

    ~GrpcWorkerServiceThread() {
      Shutdown();
    }

    void Join() { thread_->Join(); }

    void Shutdown() {
      MutexLock l(&shutdown_mu_);
      if (is_shutdown_) {
        return;
      }
      is_shutdown_ = true;
      cq_->Shutdown();
      Join();
    }

   private:
    void HandleRPCsLoop() {
      ENQUEUE_REQUEST(Ping);
      ENQUEUE_REQUEST(Execute);
      ENQUEUE_REQUEST(Execute);
      ENQUEUE_REQUEST(Execute);

      void* tag;
      bool ok;

      while (cq_->Next(&tag, &ok)) {
        UntypedCall<GrpcWorkerServiceThread>::Tag* callback_tag =
            static_cast<UntypedCall<GrpcWorkerServiceThread>::Tag*>(tag);
        callback_tag->OnCompleted(this, ok);
      }
    }

   private:
    void Schedule(std::function<void()> f) {
      worker_->env()->compute_pool->Schedule(std::move(f));
    }

    template <class Req, class Resp>
    using WorkCall =
        Call<GrpcWorkerServiceThread, EulerService::AsyncService, Req, Resp>;

    // Handlers for rpc requests

#define DECLARE_HANDLER(Method)                                            \
    void Method##Handler(WorkCall<Method##Request, Method##Reply>* call) { \
      Schedule([this, call]() {                                            \
        Status s = worker_->Method(&call->request, &call->response);       \
        call->SendResponse(ToGrpcStatus(s));                               \
      });                                                                  \
      ENQUEUE_REQUEST(Method);                                             \
    }

    DECLARE_HANDLER(Ping);

#undef DECLARE_HANDLER  // DECLARE_HANDLER

    void ExecuteHandler(WorkCall<ExecuteRequest, ExecuteReply>* call) {
      Schedule([this, call] () {
        auto done = [call] (const Status& s) {
          call->SendResponse(ToGrpcStatus(s));
        };
        worker_->ExecuteAsync(&call->request, &call->response, done);
      });
      ENQUEUE_REQUEST(Execute);
    }

#undef ENQUEUE_REQUEST  // ENQUEUE_REQUEST

   private:
    GrpcWorker* const worker_ = nullptr;  // Not owned.
    std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
    std::unique_ptr<Thread> thread_;
    EulerService::AsyncService* const euler_service_;

    Mutex shutdown_mu_;
    bool is_shutdown_;
    DISALLOW_COPY_AND_ASSIGN(GrpcWorkerServiceThread);
  };

 private:
  EulerService::AsyncService euler_service_;
  std::vector<std::unique_ptr<GrpcWorkerServiceThread>> threads_;

  Mutex shutdown_mu_;
  bool is_shutdown_;

  DISALLOW_COPY_AND_ASSIGN(GrpcWorkerService);
};

}  // namespace

std::unique_ptr<AsyncServiceInterface> NewGrpcWorkerService(
    GrpcWorker* worker, ::grpc::ServerBuilder* builder) {
  return std::unique_ptr<AsyncServiceInterface>(
      new GrpcWorkerService(worker, builder));
}

}  // namespace euler
