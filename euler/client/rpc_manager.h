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

#ifndef EULER_CLIENT_RPC_MANAGER_H_
#define EULER_CLIENT_RPC_MANAGER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <utility>

#include "google/protobuf/message.h"
#include "euler/client/graph_config.h"
#include "euler/common/status.h"
#include "euler/common/server_monitor.h"

namespace euler {

class RpcChannel;

struct RpcContext {
  RpcContext(const std::string &method,
             google::protobuf::Message *response,
             std::function<void(const Status &)> done)
      : method(method), response(response), done(done), num_failures(0) { }

  virtual bool Initialize(const google::protobuf::Message &request) = 0;
  virtual ~RpcContext() = default;

  std::string method;
  google::protobuf::Message *response;
  std::function<void(const Status &)> done;

  std::shared_ptr<RpcChannel> destination;
  int num_failures;
};

class RpcChannel {
 public:
  explicit RpcChannel(const std::string &host_port) : host_port_(host_port) { }

  virtual ~RpcChannel() = default;

  virtual void IssueRpcCall(RpcContext *ctx) = 0;

  std::string host_port() { return host_port_; }

 private:
  std::string host_port_;
};

class RpcManager {
 public:
  RpcManager()
      : num_channels_per_host_(1),
        bad_host_cleanup_interval_(1),
        bad_host_timeout_(10),
        next_replica_index_(0),
        shutdown_(false),
        bad_hosts_cleaner_(std::thread(&RpcManager::CleanupBadHosts, this)),
        shard_callback_(
            std::bind(&RpcManager::AddChannel, this, std::placeholders::_1),
            std::bind(&RpcManager::RemoveChannel, this, std::placeholders::_1)
        ) { }

  bool Initialize(std::shared_ptr<ServerMonitor> monitor,
                  size_t shard_index,
                  const GraphConfig &config = GraphConfig());
  virtual ~RpcManager();

  virtual std::unique_ptr<RpcChannel> CreateChannel(
      const std::string &host_port, int tag) = 0;

  virtual RpcContext *CreateContext(
      const std::string &method,
      google::protobuf::Message *respone,
      std::function<void(const Status &)> done) = 0;

  std::shared_ptr<RpcChannel> GetChannel();
  void MoveToBadHost(const std::string &host_port);

 private:
  using TimePoint = std::chrono::time_point<std::chrono::system_clock>;
  using Duration = std::chrono::seconds;
  // Bad host and detected time.
  using BadHost = std::pair<std::string, TimePoint>;

  void AddChannel(const std::string &host_port);
  void RemoveChannel(const std::string &host_port);
  void CleanupBadHosts();
  void DoAddChannel(const std::string &host_port);
  void DoRemoveChannel(const std::string &host_port);
  void DoCleanupBadHosts(TimePoint now);

  int num_channels_per_host_;
  Duration bad_host_cleanup_interval_;
  Duration bad_host_timeout_;

  std::vector<std::shared_ptr<RpcChannel>> channels_;
  std::vector<BadHost> bad_hosts_;
  size_t next_replica_index_;
  std::mutex mu_;
  std::condition_variable cv_;

  bool shutdown_;
  std::thread bad_hosts_cleaner_;

  std::shared_ptr<ServerMonitor> monitor_;
  size_t shard_index_;
  ShardCallback shard_callback_;
};

}  // namespace euler

#endif  // EULER_CLIENT_RPC_MANAGER_H_
