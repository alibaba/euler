/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "euler/client/grpc_channel.h"

#include "glog/logging.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "euler/client/grpc_thread_pool.h"

namespace euler {
namespace client {

namespace {

class GrpcClosure final : public GrpcCQTag {
 public:
  explicit GrpcClosure(GrpcContext *ctx) : ctx_(ctx) { }

  void OnCompleted(bool ok) override {
    Status status(Status::OK);
    if (!ok) {
      status = Status(StatusCode::RPC_ERROR, "gRpc CQ not ok.");
    } else if (!ctx_->status.ok()) {
      status = Status(StatusCode::RPC_ERROR,
                      "gRpc error: " + ctx_->status.error_message());
    } else {
      grpc::ProtoBufferReader reader(&ctx_->response_buf);
      if (!ctx_->response->ParseFromZeroCopyStream(&reader)) {
        status = Status(StatusCode::PROTO_ERROR, "Bad response.");
      }
    }
    ctx_->done(status);
    delete this;
  }

 private:
  GrpcContext *ctx_;
};

}  // namespace

bool GrpcContext::Initialize(const google::protobuf::Message &request) {
  bool own_buffer;
  grpc::Status s = grpc::GenericSerialize<
    grpc::ProtoBufferWriter,
    google::protobuf::Message>(request, &request_buf, &own_buffer);
  return s.ok();
}

GrpcChannel::GrpcChannel(const std::string &host_port,
                         std::shared_ptr<grpc::Channel> raw_channel)
    : RpcChannel(host_port),
      stub_(raw_channel),
      cq_(GrpcThreadPool::GetInstance()->NextCompletionQueue()) { }

void GrpcChannel::IssueRpcCall(RpcContext *ctx) {
  GrpcContext *grpc_ctx = dynamic_cast<GrpcContext *>(ctx);
  if (!grpc_ctx) {
    ctx->done(Status(StatusCode::INVALID_ARGUMENT, "Wrong RpcContext."));
    return;
  }

  grpc_ctx->context.reset(new grpc::ClientContext);
  grpc_ctx->response_reader = stub_.PrepareUnaryCall(
      grpc_ctx->context.get(), grpc_ctx->method, grpc_ctx->request_buf, cq_);
  grpc_ctx->response_reader->StartCall();
  grpc_ctx->response_reader->Finish(&grpc_ctx->response_buf, &grpc_ctx->status,
                                    new GrpcClosure(grpc_ctx));
}

}  // namespace client
}  // namespace euler
