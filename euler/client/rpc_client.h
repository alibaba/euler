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

#ifndef EULER_CLIENT_RPC_CLIENT_H_
#define EULER_CLIENT_RPC_CLIENT_H_

#include <string>

#include "google/protobuf/message.h"

#include "euler/client/graph_config.h"
#include "euler/client/impl_register.h"
#include "euler/client/rpc_manager.h"
#include "euler/client/status.h"
#include "euler/common/server_monitor.h"

namespace euler {
namespace client {

class RpcClient {
 public:
  virtual bool Initialize(std::shared_ptr<common::ServerMonitor> monitor,
                          size_t shard_index, const GraphConfig &config) = 0;
  virtual void IssueRpcCall(const std::string &method,
                            const google::protobuf::Message &request,
                            google::protobuf::Message *respone,
                            std::function<void(const Status &)> done) = 0;
  virtual ~RpcClient() = default;
};

class RpcClientBase : public RpcClient {
 public:
  RpcClientBase() : rpc_manager_(ImplFactory<RpcManager>::New()),
                    num_retries_(kRpcRetryCount) { }

  bool Initialize(std::shared_ptr<common::ServerMonitor> monitor,
                  size_t shard_index, const GraphConfig &config) override;
  void IssueRpcCall(const std::string &method,
                    const google::protobuf::Message &request,
                    google::protobuf::Message *respone,
                    std::function<void(const Status &)> done) override;

 private:
  static constexpr const int kRpcRetryCount = 10;

  void DoIssueRpcCall(RpcContext *ctx);

  std::unique_ptr<RpcManager> rpc_manager_;
  int num_retries_;
};

std::unique_ptr<RpcClient> NewRpcClient(
    std::shared_ptr<common::ServerMonitor> monitor, size_t shard_index,
    const GraphConfig &config = GraphConfig());

}  // namespace client
}  // namespace euler

#endif  // EULER_CLIENT_RPC_CLIENT_H_
