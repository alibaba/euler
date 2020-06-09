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

#include "euler/client/rpc_client.h"

#include <string>

#include "gtest/gtest.h"
#include "euler/common/logging.h"
#include "euler/client/testing/echo.h"
#include "euler/common/mutex.h"
#include "euler/common/server_monitor.h"
#include "euler/common/server_register.h"

using euler::testing::EchoRequest;
using euler::testing::EchoResponse;
using euler::testing::EchoServiceImpl;

namespace euler {
namespace {

Mutex mu;

class RpcClientEnd2EndTest : public ::testing::Test {
 protected:
  static const char zk_path_[];
  static const int num_shards_;
  EchoServiceImpl service_;
  std::shared_ptr<ServerRegister> register_;

  void SetUp() {
    mu.Lock();
    service_.BuildAndStart();
    register_ = GetServerRegister("127.0.0.1:2181", zk_path_);
    EXPECT_TRUE(register_);
    for (int i = 0; i < num_shards_; ++i) {
      EXPECT_TRUE(register_->RegisterShard(i, service_.host_port(),
                                           Meta(), Meta()));
    }
  }

  void TearDown() {
    for (int i = 0; i < num_shards_; ++i) {
      EXPECT_TRUE(register_->DeregisterShard(i, service_.host_port()));
    }
    service_.Shutdown();
    sleep(3);  // Wait for port to cleanup
    mu.Unlock();
  }
};

const char RpcClientEnd2EndTest::zk_path_[] =
    "/euler_client_rpc_client_end2end_test";
const int RpcClientEnd2EndTest::num_shards_ = 3;

TEST_F(RpcClientEnd2EndTest, RpcClientEnd2EndSimple) {
  auto monitor = GetServerMonitor(
      "127.0.0.1:2181", zk_path_);
  EXPECT_TRUE(monitor);

  auto rpc_client = NewRpcClient(monitor, 0);
  EXPECT_TRUE(rpc_client);

  std::atomic_bool finished(false);
  EchoResponse response;
  {
    std::string method = "/euler.testing.EchoService/Echo";
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
  EULER_LOG(INFO) << response.message();
  EXPECT_EQ("hello euler", response.message());
}

TEST_F(RpcClientEnd2EndTest, RpcClientEnd2EndMultiRound) {
  auto monitor = GetServerMonitor(
      "127.0.0.1:2181", zk_path_);
  EXPECT_TRUE(monitor);

  auto rpc_client = NewRpcClient(monitor, 0);
  EXPECT_TRUE(rpc_client);

  int num_times = 10;
  std::atomic_int counter(0);
  std::vector<EchoResponse> responses(num_times);

  {
    std::string method = "/euler.testing.EchoService/Echo";
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
  auto monitor = GetServerMonitor(
      "127.0.0.1:2181", zk_path_);
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
    std::string method = "/euler.testing.EchoService/Echo";
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
}  // namespace euler
