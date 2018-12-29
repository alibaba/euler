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

#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "euler/client/testing/echo.h"
#include "euler/common/zk_service.h"
#include "euler/common/server_monitor.h"
#include "euler/common/server_register.h"

using euler::client::testing::EchoRequest;
using euler::client::testing::EchoResponse;
using euler::client::testing::EchoServiceImpl;

namespace euler {
namespace client {
namespace {

class RpcClientEnd2EndTest : public ::testing::Test {
 protected:
  static common::ZkService zk_service_;
  static const std::string zk_path_;
  static const int num_shards_;
  static EchoServiceImpl service_;
  static std::shared_ptr<common::ServerRegister> register_;

  static void SetUpTestCase() {
    EXPECT_TRUE(zk_service_.Start());
    service_.BuildAndStart();
    register_ = common::GetServerRegister(zk_service_.HostPort(),
                                          zk_path_);
    EXPECT_TRUE(register_);
    for (int i = 0; i < num_shards_; ++i) {
      EXPECT_TRUE(register_->RegisterShard(i, service_.host_port(),
                                           common::Meta(), common::Meta()));
    }
  }

  static void TearDownTestCase() {
    for (int i = 0; i < num_shards_; ++i) {
      EXPECT_TRUE(register_->DeregisterShard(i, service_.host_port()));
    }
    service_.Shutdown();
    EXPECT_TRUE(zk_service_.Shutdown());
  }
};

common::ZkService RpcClientEnd2EndTest::zk_service_;
const std::string RpcClientEnd2EndTest::zk_path_ =
    "/euler_client_rpc_client_end2end_test";
const int RpcClientEnd2EndTest::num_shards_ = 3;
EchoServiceImpl RpcClientEnd2EndTest::service_;
std::shared_ptr<common::ServerRegister> RpcClientEnd2EndTest::register_;

TEST_F(RpcClientEnd2EndTest, RpcClientEnd2EndSimple) {
  auto monitor = common::GetServerMonitor(
      RpcClientEnd2EndTest::zk_service_.HostPort(), zk_path_);
  EXPECT_TRUE(monitor);

  auto rpc_client = NewRpcClient(monitor, 0);
  EXPECT_TRUE(rpc_client);

  std::atomic_bool finished(false);
  EchoResponse response;
  {
    std::string method = "/euler.client.testing.EchoService/Echo";
    EchoRequest request;
    request.set_message("hello euler");
    rpc_client->IssueRpcCall(
        method, request, &response, [&finished](const Status &status) {
          EXPECT_TRUE(status.ok());
          finished = true;
        });
  }

  while (!finished) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_EQ("hello euler", response.message());
}

TEST_F(RpcClientEnd2EndTest, RpcClientEnd2EndMultiRound) {
  auto monitor = common::GetServerMonitor(
      RpcClientEnd2EndTest::zk_service_.HostPort(), zk_path_);
  EXPECT_TRUE(monitor);

  auto rpc_client = NewRpcClient(monitor, 0);
  EXPECT_TRUE(rpc_client);

  int num_times = 10;
  std::atomic_int counter(0);
  std::vector<EchoResponse> responses(num_times);

  {
    std::string method = "/euler.client.testing.EchoService/Echo";
    EchoRequest request;
    request.set_message("hello euler");

    for (int i = 0; i < num_times; ++i) {
      EchoResponse *response = &responses[i];
      rpc_client->IssueRpcCall(
          method, request, response, [&counter](const Status &status) {
            EXPECT_TRUE(status.ok());
            ++counter;
          });
    }
  }

  while (counter < num_times) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  for (const auto &response : responses) {
    EXPECT_EQ("hello euler", response.message());
  }
}

TEST_F(RpcClientEnd2EndTest, RpcClientEnd2EndMultiShards) {
  auto monitor = common::GetServerMonitor(
      RpcClientEnd2EndTest::zk_service_.HostPort(), zk_path_);
  EXPECT_TRUE(monitor);

  std::vector<std::unique_ptr<RpcClient>> rpc_clients;
  for (int i = 0; i < num_shards_; ++i) {
    auto rpc_client = NewRpcClient(monitor, i);
    EXPECT_TRUE(rpc_client);
    rpc_clients.emplace_back(std::move(rpc_client));
  }

  std::atomic_int counter(0);
  std::vector<EchoResponse> responses(num_shards_);

  {
    std::string method = "/euler.client.testing.EchoService/Echo";
    EchoRequest request;
    request.set_message("hello euler");
    for (int i = 0; i < num_shards_; ++i) {
      auto &rpc_client = rpc_clients[i];
      EchoResponse *response = &responses[i];
      rpc_client->IssueRpcCall(
          method, request, response, [&counter](const Status &status) {
            EXPECT_TRUE(status.ok());
            ++counter;
          });
    }
  }

  while (counter < num_shards_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  for (const auto &response : responses) {
    EXPECT_EQ("hello euler", response.message());
  }
}

}  // namespace
}  // namespace client
}  // namespace euler
