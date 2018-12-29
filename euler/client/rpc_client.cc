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

#include "euler/client/rpc_client.h"

#include "euler/client/impl_register.h"

namespace euler {
namespace client {

bool RpcClientBase::Initialize(std::shared_ptr<common::ServerMonitor> monitor,
                               size_t shard_index, const GraphConfig &config) {
  config.Get("num_retries", &num_retries_);
  return rpc_manager_ && rpc_manager_->Initialize(monitor, shard_index, config);
}

void RpcClientBase::IssueRpcCall(const std::string &method,
                                 const google::protobuf::Message &request,
                                 google::protobuf::Message *respone,
                                 std::function<void(const Status &)> done) {
  RpcContext *ctx = rpc_manager_->CreateContext(method, respone, nullptr);
  ctx->done = [ctx, done, this](const Status &status) {
    if (!status.ok()) {
      rpc_manager_->MoveToBadHost(ctx->destination->host_port());
    }

    if (status.ok() ||
        (num_retries_ > 0 && ++ctx->num_failures == num_retries_)) {
      done(status);
      delete ctx;
    } else {
      DoIssueRpcCall(ctx);
    }
  };
  if (!(ctx && ctx->Initialize(request))) {
    done(Status(StatusCode::PROTO_ERROR, "Bad request."));
  }
  DoIssueRpcCall(ctx);
}

void RpcClientBase::DoIssueRpcCall(RpcContext *ctx) {
  ctx->destination = rpc_manager_->GetChannel();
  ctx->destination->IssueRpcCall(ctx);
}

std::unique_ptr<RpcClient> NewRpcClient(
    std::shared_ptr<common::ServerMonitor> monitor, size_t shard_index,
    const GraphConfig &config) {
  std::unique_ptr<RpcClient> rpc_client(ImplFactory<RpcClient>::New());
  if (rpc_client && rpc_client->Initialize(monitor, shard_index, config)) {
    return rpc_client;
  } else {
    return nullptr;
  }
}

REGISTER_IMPL(RpcClient, RpcClientBase);

}  // namespace client
}  // namespace euler
