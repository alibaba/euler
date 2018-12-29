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

#include "euler/client/rpc_manager.h"

#include <thread>

#include "gtest/gtest.h"
#include "euler/client/testing/mock_rpc_manager.h"
#include "euler/client/testing/simple_server_monitor.h"

namespace euler {
namespace client {
namespace {

class RpcManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_TRUE(testing::NewSimpleMonitor(&regs_, &monitor_));
  }

  void ParallelQuery(int num_threads,
                     int num_times_per_replica, int num_replicas,
                     std::vector<std::atomic_int> *counters) {
    EXPECT_EQ(0, num_replicas * num_times_per_replica % num_threads);
    int num_times = num_replicas * num_times_per_replica / num_threads;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(
        [num_times, counters, this] {
          std::vector<int> local_counters(counters->size());
          for (int i = 0; i < num_times; ++i) {
            const std::string &host_port =
                rpc_manager_.GetChannel()->host_port();
            ++local_counters[std::stoi(host_port)];
          }
          for (size_t i = 0; i < counters->size(); ++i) {
            (*counters)[i] += local_counters[i];
          }
        });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  }

  testing::MockRpcManager rpc_manager_;
  std::shared_ptr<common::ServerRegister> regs_;
  std::shared_ptr<common::ServerMonitor> monitor_;
};

TEST_F(RpcManagerTest, Balance) {
  EXPECT_TRUE(rpc_manager_.Initialize(monitor_, 0));

  int num_replicas = 3;
  std::vector<std::atomic_int> counters(num_replicas);

  for (int i = 0; i < num_replicas; ++i) {
    EXPECT_TRUE(regs_->RegisterShard(0, std::to_string(i),
                                    common::Meta(), common::Meta()));
  }

  int num_times_per_replica = 100000;

  ParallelQuery(32, num_times_per_replica, num_replicas, &counters);

  for (const auto &counter : counters) {
    EXPECT_EQ(num_times_per_replica, counter);
  }
}

TEST_F(RpcManagerTest, ServerChanges) {
  EXPECT_TRUE(rpc_manager_.Initialize(monitor_, 0));

  std::vector<std::atomic_int> counters(3);

  EXPECT_TRUE(regs_->RegisterShard(0, "0", common::Meta(), common::Meta()));
  EXPECT_TRUE(regs_->RegisterShard(0, "1", common::Meta(), common::Meta()));

  int num_times_per_replica = 20000;

  ParallelQuery(32, num_times_per_replica, 2, &counters);

  EXPECT_EQ(num_times_per_replica, counters[0]);
  EXPECT_EQ(num_times_per_replica, counters[1]);

  EXPECT_TRUE(regs_->RegisterShard(0, "2", common::Meta(), common::Meta()));

  ParallelQuery(32, num_times_per_replica, 3, &counters);

  EXPECT_EQ(num_times_per_replica * 2, counters[0]);
  EXPECT_EQ(num_times_per_replica * 2, counters[1]);
  EXPECT_EQ(num_times_per_replica    , counters[2]);

  EXPECT_TRUE(regs_->DeregisterShard(0, "1"));

  ParallelQuery(32, num_times_per_replica, 2, &counters);

  EXPECT_EQ(num_times_per_replica * 3, counters[0]);
  EXPECT_EQ(num_times_per_replica * 2, counters[1]);
  EXPECT_EQ(num_times_per_replica * 2, counters[2]);
}

TEST_F(RpcManagerTest, BlackList) {
  GraphConfig config;
  config.Add("bad_host_cleanup_interval", 1);
  config.Add("bad_host_timeout", 5);
  EXPECT_TRUE(rpc_manager_.Initialize(monitor_, 0, config));

  std::vector<std::atomic_int> counters(3);

  EXPECT_TRUE(regs_->RegisterShard(0, "0", common::Meta(), common::Meta()));
  EXPECT_TRUE(regs_->RegisterShard(0, "1", common::Meta(), common::Meta()));
  EXPECT_TRUE(regs_->RegisterShard(0, "2", common::Meta(), common::Meta()));

  int num_times_per_replica = 20000;

  ParallelQuery(32, num_times_per_replica, 3, &counters);

  EXPECT_EQ(num_times_per_replica, counters[0]);
  EXPECT_EQ(num_times_per_replica, counters[1]);
  EXPECT_EQ(num_times_per_replica, counters[2]);

  rpc_manager_.MoveToBadHost("2");

  ParallelQuery(32, num_times_per_replica, 2, &counters);

  EXPECT_EQ(num_times_per_replica * 2, counters[0]);
  EXPECT_EQ(num_times_per_replica * 2, counters[1]);
  EXPECT_EQ(num_times_per_replica    , counters[2]);

  // To ensure bad host is recovered by rpc manager.
  std::this_thread::sleep_for(std::chrono::seconds(6));

  ParallelQuery(32, num_times_per_replica, 3, &counters);

  EXPECT_EQ(num_times_per_replica * 3, counters[0]);
  EXPECT_EQ(num_times_per_replica * 3, counters[1]);
  EXPECT_EQ(num_times_per_replica * 2, counters[2]);
}

}  // namespace
}  // namespace client
}  // namespace euler
