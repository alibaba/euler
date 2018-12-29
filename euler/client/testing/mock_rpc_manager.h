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

#ifndef EULER_CLIENT_TESTING_MOCK_RPC_MANAGER_H_
#define EULER_CLIENT_TESTING_MOCK_RPC_MANAGER_H_

#include <vector>
#include <string>

#include "gmock/gmock.h"

#include "euler/client/rpc_manager.h"

namespace euler {
namespace client {
namespace testing {

struct MockRpcContext : public RpcContext {
  MockRpcContext(const std::string &method,
                 google::protobuf::Message *response,
                 std::function<void(const Status &)> done)
      : RpcContext(method, response, done) { }

  bool Initialize(const google::protobuf::Message &/*request*/) override {
    return true;
  }
};

class MockRpcChannel : public RpcChannel {
 public:
  static std::vector<MockRpcChannel *> channels;

  explicit MockRpcChannel(const std::string &host_port)
      : RpcChannel(host_port) {
    channels.emplace_back(this);
  }

  ~MockRpcChannel() override {
    auto iter = std::find(channels.begin(), channels.end(), this);
    channels.erase(iter);
  }

  MOCK_METHOD1(IssueRpcCall, void(RpcContext *));
};

class MockRpcManager : public RpcManager {
 public:
  std::unique_ptr<RpcChannel> CreateChannel(
      const std::string &host_port, int /*tag*/) override {
    return std::unique_ptr<RpcChannel>(new MockRpcChannel(host_port));
  }

  RpcContext *CreateContext(const std::string &method,
                            google::protobuf::Message *respone,
                            std::function<void(const Status &)> done) override {
    return new MockRpcContext(method, respone, done);
  }
};

}  // namespace testing
}  // namespace client
}  // namespace euler

#endif  // EULER_CLIENT_TESTING_MOCK_RPC_MANAGER_H_
