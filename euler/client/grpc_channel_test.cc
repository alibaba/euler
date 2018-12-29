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

#include <thread>

#include "gtest/gtest.h"

#include "euler/client/grpc_manager.h"
#include "euler/client/testing/echo.h"

namespace euler {
namespace client {
namespace {

class GrpcChannelTest : public ::testing::Test {
 protected:
  void SetUp() override { service_.BuildAndStart(); }

  void TearDown() override { service_.Shutdown(); }

  testing::EchoServiceImpl service_;
};

TEST_F(GrpcChannelTest, IssueRpcCall) {
  std::atomic_bool finished(false);

  std::string method = "/euler.client.testing.EchoService/Echo";
  testing::EchoRequest request;
  testing::EchoResponse response;
  request.set_message("hello euler");
  GrpcManager grpc_manager;
  std::unique_ptr<RpcContext> ctx(grpc_manager.CreateContext(
      method, &response,
      [request, &response, &finished](const Status &status) {
        EXPECT_TRUE(status.ok());
        EXPECT_EQ(request.message(), response.message());
        finished = true;
      }));
  EXPECT_TRUE(ctx->Initialize(request));
  std::unique_ptr<RpcChannel> channel =
      grpc_manager.CreateChannel(service_.host_port(), 0);
  channel->IssueRpcCall(ctx.get());

  while (!finished) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace
}  // namespace client
}  // namespace euler
