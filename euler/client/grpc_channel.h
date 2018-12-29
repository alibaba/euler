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

#ifndef EULER_CLIENT_GRPC_CHANNEL_H_
#define EULER_CLIENT_GRPC_CHANNEL_H_

#include <string>

#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/grpcpp.h"
#include "euler/client/rpc_manager.h"

namespace euler {
namespace client {

struct GrpcContext: public RpcContext {
  GrpcContext(const std::string &method,
              google::protobuf::Message *respone,
              std::function<void(const Status &)> done)
      : RpcContext(method, respone, done) { }

  bool Initialize(const google::protobuf::Message &request);

  grpc::ByteBuffer request_buf;
  grpc::ByteBuffer response_buf;
  grpc::Status status;
  std::unique_ptr<grpc::ClientContext> context;
  std::unique_ptr<grpc::GenericClientAsyncResponseReader> response_reader;
};

class GrpcChannel: public RpcChannel {
 public:
  GrpcChannel(const std::string& host_port,
              std::shared_ptr<grpc::Channel> raw_channel);

  void IssueRpcCall(RpcContext *ctx) override;

 private:
  grpc::GenericStub stub_;
  grpc::CompletionQueue *cq_;
};

}  // namespace client
}  // namespace euler

#endif  // EULER_CLIENT_GRPC_CHANNEL_H_
