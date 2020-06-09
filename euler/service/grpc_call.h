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


#ifndef EULER_SERVICE_GRPC_CALL_H_
#define EULER_SERVICE_GRPC_CALL_H_

#include <map>
#include <utility>

#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/server_builder.h"

#include "euler/common/mutex.h"
#include "euler/common/refcount.h"

namespace euler {

template <class Service>
class UntypedCall : public RefCounted {
 public:
  virtual ~UntypedCall() {}

  virtual void RequestReceived(Service* service, bool ok) = 0;

  class Tag {
   public:
    enum Callback { kRequestReceived, kResponseSent};

    Tag(UntypedCall* call, Callback cb) : call_(call), callback_(cb) {}

    void OnCompleted(Service* service, bool ok) {
      switch (callback_) {
        case kRequestReceived:
          call_->RequestReceived(service, ok);
          break;
        case kResponseSent:
          break;
      }
      call_->Unref();
    }

   private:
    UntypedCall* const call_;  // `this` owns one reference.
    Callback callback_;
  };
};

template <class Service, class GrpcService, class RequestMessage,
          class ResponseMessage>
class Call : public UntypedCall<Service> {
 public:
  using EnqueueFunction = void (GrpcService::*)(
      ::grpc::ServerContext*, RequestMessage*,
      ::grpc::ServerAsyncResponseWriter<ResponseMessage>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*);

  using HandleRequestFunction = void (Service::*)(
      Call<Service, GrpcService, RequestMessage, ResponseMessage>*);

  explicit Call(HandleRequestFunction handle_request_function)
      : handle_request_function_(handle_request_function), responder_(&ctx_) {}

  virtual ~Call() {}

  void RequestReceived(Service* service, bool ok) override {
    if (ok) {
      this->Ref();
      (service->*handle_request_function_)(this);
    }
  }

  void SendResponse(::grpc::Status status) {
    this->Ref();  // Ref for grpc; released in Tag callback.
    responder_.Finish(response, status, &response_sent_tag_);
    this->Unref();
  }

  static void EnqueueRequestForMethod(
      GrpcService* grpc_service, ::grpc::ServerCompletionQueue* cq,
      int method_id, HandleRequestFunction handle_request_function) {
    auto call = new Call<Service, GrpcService, RequestMessage, ResponseMessage>(
        handle_request_function);
    grpc_service->RequestAsyncUnary(method_id, &call->ctx_, &call->request,
                                    &call->responder_, cq, cq,
                                    &call->request_received_tag_);
  }

  RequestMessage request;
  ResponseMessage response;

  const std::multimap<::grpc::string_ref, ::grpc::string_ref>& client_metadata()
      const {
    return ctx_.client_metadata();
  }

 private:
  HandleRequestFunction handle_request_function_;
  ::grpc::ServerContext ctx_;
  ::grpc::ServerAsyncResponseWriter<ResponseMessage> responder_;

  typedef typename UntypedCall<Service>::Tag Tag;
  Tag request_received_tag_{this, Tag::kRequestReceived};
  Tag response_sent_tag_{this, Tag::kResponseSent};

  Mutex mu_;
};

}  // namespace euler

#endif  // EULER_SERVICE_GRPC_CALL_H_
