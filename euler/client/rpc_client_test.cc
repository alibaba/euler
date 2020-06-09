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

#include <memory>

#include "gtest/gtest.h"
#include "euler/client/testing/echo.h"
#include "euler/client/testing/mock_rpc_manager.h"
#include "euler/client/testing/simple_server_monitor.h"
#include "euler/client/rpc_client.h"

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::InvokeWithoutArgs;
using euler::testing::EchoRequest;
using euler::testing::EchoResponse;
using euler::testing::EchoServiceImpl;
using euler::testing::MockRpcChannel;
using euler::testing::MockRpcContext;
using euler::testing::MockRpcManager;
using euler::testing::NewSimpleMonitor;

namespace euler {
namespace {

class RpcClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_TRUE(NewSimpleMonitor(&regs_, &monitor_));

    REGISTER_IMPL(RpcManager, MockRpcManager);
  }

  std::shared_ptr<ServerRegister> regs_;
  std::shared_ptr<ServerMonitor> monitor_;
};

TEST_F(RpcClientTest, TestRetrySuccess) {
  int num_retries = 3;
  GraphConfig config;
  config.Add("num_retries", num_retries);

  auto rpc_client = NewRpcClient(monitor_, 0, config);
  EXPECT_TRUE(rpc_client);

  for (int i = 0; i < num_retries; ++i) {
    regs_->RegisterShard(0, std::to_string(i), Meta(), Meta());
  }

  EchoResponse response;
  for (MockRpcChannel *channel : MockRpcChannel::channels) {
    EXPECT_CALL(*channel, IssueRpcCall(_))
        .WillOnce(Invoke([&](RpcContext *ctx) {
                    if (ctx->num_failures < num_retries - 1) {
                      ctx->done(Status(ErrorCode::RPC_ERROR, ""));
                    } else {
                       response.set_message("hello euler");
                       ctx->done(Status::OK());
                    }
                  }));
  }

  bool finished = false;
  {
    std::string method;
    EchoRequest request;
    rpc_client->IssueRpcCall(
        method, request, &response, [&](const Status &status) {
          EXPECT_TRUE(status.ok());
          finished = true;
        });
  }

  EXPECT_TRUE(finished);
  EXPECT_EQ("hello euler", response.message());
}

TEST_F(RpcClientTest, TestRetryFailure) {
  int num_retries = 3;
  GraphConfig config;
  config.Add("num_retries", num_retries);

  auto rpc_client = NewRpcClient(monitor_, 0, config);
  EXPECT_TRUE(rpc_client);

  for (int i = 0; i < num_retries; ++i) {
    regs_->RegisterShard(0, std::to_string(i), Meta(), Meta());
  }

  EchoResponse response;
  for (MockRpcChannel *channel : MockRpcChannel::channels) {
     EXPECT_CALL(*channel, IssueRpcCall(_))
         .WillOnce(Invoke([&](RpcContext *ctx) {
                     ctx->done(Status(ErrorCode::RPC_ERROR, ""));
                   }));
  }

  bool finished = false;
  {
    std::string method;
    EchoRequest request;
    rpc_client->IssueRpcCall(
        method, request, &response, [&](const Status &status) {
          EXPECT_FALSE(status.ok());
          finished = true;
        });
  }

  EXPECT_TRUE(finished);
}

}  // namespace
}  // namespace euler
