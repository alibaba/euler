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

#include <chrono>
#include <future>
#include <thread>

#include "gtest/gtest.h"
#include "glog/logging.h"

#include "euler/common/server_monitor.h"
#include "euler/common/server_register.h"
#include "euler/common/zk_service.h"

namespace euler {
namespace common {

class ServerMonitorTest : public testing::Test {
 protected:
  static ZkService zk_service_;
  static const std::string zk_path_;

  static void SetUpTestCase() {
    EXPECT_TRUE(zk_service_.Start());
  }

  static void TearDownTestCase() {
    EXPECT_TRUE(zk_service_.Shutdown());
  }

  void SetUp() override {
    monitor_ = GetServerMonitor(zk_service_.HostPort(), zk_path_);
    EXPECT_TRUE(monitor_);

    register_ = GetServerRegister(zk_service_.HostPort(), zk_path_);
    EXPECT_TRUE(register_);
  }

  void TearDown() override { }

  std::shared_ptr<ServerMonitor> monitor_;
  std::shared_ptr<ServerRegister> register_;
};

ZkService ServerMonitorTest::zk_service_;
const std::string ServerMonitorTest::zk_path_ =
    "/euler_common_server_monitor_test";

TEST_F(ServerMonitorTest, Asynchronous) {
  int count = 0;
  ShardCallback callback(
      [&count](const Server &server) {
        EXPECT_EQ(std::to_string(count++), server);
      },
      [&count](const Server &server) {
        EXPECT_EQ(std::to_string(--count), server);
      });
  EXPECT_TRUE(monitor_->SetShardCallback(0, &callback));

  Meta meta{{"key", "value"}};
  Meta shard_meta{{"shard_key", "shard_value"}};

  std::async(std::launch::async, [this, meta, shard_meta]() {
    for (int i = 0; i < 3; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      EXPECT_TRUE(register_->RegisterShard(0, std::to_string(i),
                                           meta, shard_meta));
    }
  });

  std::string value;

  EXPECT_TRUE(monitor_->GetMeta("key", &value));
  EXPECT_EQ(meta["key"], value);

  EXPECT_TRUE(monitor_->GetShardMeta(0, "shard_key", &value));
  EXPECT_EQ(shard_meta["shard_key"], value);

  // To make sure all changes are received.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_EQ(3, count);

  for (int i = 2; i >= 0; --i) {
    EXPECT_TRUE(register_->DeregisterShard(0, std::to_string(i)));
  }

  // To make sure all changes are received.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_EQ(0, count);

  EXPECT_TRUE(monitor_->UnsetShardCallback(0, &callback));
}

TEST_F(ServerMonitorTest, AddBeforeListen) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(register_->RegisterShard(0, std::to_string(i), Meta(), Meta()));
  }

  // To make sure all changes are received.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  int count = 0;
  ShardCallback callback(
      [&count](const Server &) { ++count; },
      [&count](const Server &) { --count; });
  EXPECT_TRUE(monitor_->SetShardCallback(0, &callback));

  EXPECT_EQ(3, count);

  EXPECT_TRUE(monitor_->UnsetShardCallback(0, &callback));
  for (int i = 2; i >= 0; --i) {
    EXPECT_TRUE(register_->DeregisterShard(0, std::to_string(i)));
  }

  // To make sure all changes are received.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(3, count);
}

}  // namespace common
}  // namespace euler
