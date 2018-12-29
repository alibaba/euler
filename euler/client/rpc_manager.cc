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

#include "glog/logging.h"
#include "euler/client/impl_register.h"
#include "euler/client/grpc_manager.h"

namespace euler {
namespace client {

bool RpcManager::Initialize(std::shared_ptr<common::ServerMonitor> monitor,
                            size_t shard_index, const GraphConfig &config) {
  if (monitor_) {
    return true;
  }

  config.Get("num_channels_per_host", &num_channels_per_host_);
  int value_int;
  if (config.Get("bad_host_cleanup_interval", &value_int)) {
    bad_host_cleanup_interval_ = Duration(value_int);
  }
  if (config.Get("bad_host_timeout", &value_int)) {
    bad_host_timeout_ = Duration(value_int);
  }

  bool success = monitor->SetShardCallback(shard_index, &shard_callback_);
  if (success) {
    monitor_ = monitor;
    shard_index_ = shard_index;
  } else {
    LOG(ERROR) << "Fail to listen on ServerMonitor.";
  }
  return success;
}

RpcManager::~RpcManager() {
  shutdown_ = true;
  bad_hosts_cleaner_.join();
  if (monitor_) {
    monitor_->UnsetShardCallback(shard_index_, &shard_callback_);
  }
}

std::shared_ptr<RpcChannel> RpcManager::GetChannel() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this]{ return !channels_.empty(); });
  return channels_[next_replica_index_++ % channels_.size()];
}

void RpcManager::MoveToBadHost(const std::string &host_port) {
  std::lock_guard<std::mutex> lock(mu_);
  DoRemoveChannel(host_port);
  // MoveToBadHost may be call many times for the same host.
  if (std::find_if(bad_hosts_.begin(), bad_hosts_.end(),
                   [host_port](const BadHost &bad_host) {
                     return bad_host.first == host_port;
                   }) == bad_hosts_.end()) {
    bad_hosts_.emplace_back(host_port, std::chrono::system_clock::now());
  }
}

void RpcManager::AddChannel(const std::string &host_port) {
  {
    std::lock_guard<std::mutex> lock(mu_);
#if 0
    for (auto& channel : channels_) {
      if (channel->host_port() == host_port) {
        return;
      }
    }
#endif
    DoAddChannel(host_port);
  }
  cv_.notify_all();
}

void RpcManager::RemoveChannel(const std::string &host_port) {
  std::lock_guard<std::mutex> lock(mu_);
  DoRemoveChannel(host_port);
  bad_hosts_.erase(
      std::remove_if(bad_hosts_.begin(), bad_hosts_.end(),
                     [host_port](const BadHost &bad_host) {
                       return bad_host.first == host_port;
                     }),
      bad_hosts_.end());
}

void RpcManager::CleanupBadHosts() {
  while (!shutdown_) {
    std::this_thread::sleep_for(bad_host_cleanup_interval_);
    TimePoint now = std::chrono::system_clock::now();
    {
      std::lock_guard<std::mutex> lock(mu_);
      DoCleanupBadHosts(now);
    }
    cv_.notify_all();
  }
}

void RpcManager::DoAddChannel(const std::string &host_port) {
  for (int tag = 0; tag < num_channels_per_host_; ++tag) {
    channels_.emplace_back(CreateChannel(host_port, tag));
  }
}

void RpcManager::DoRemoveChannel(const std::string &host_port) {
  channels_.erase(
      std::remove_if(channels_.begin(), channels_.end(),
                     [host_port](const std::shared_ptr<RpcChannel> &channel) {
                       return channel->host_port() == host_port;
                     }),
      channels_.end());
}

void RpcManager::DoCleanupBadHosts(TimePoint now) {
  auto iter = std::partition(
      bad_hosts_.begin(), bad_hosts_.end(),
      [now, this](const BadHost &bad_host) {
        return now - bad_host.second < bad_host_timeout_;
      });
  std::for_each(iter, bad_hosts_.end(), [this](const BadHost &bad_host) {
    DoAddChannel(bad_host.first);
  });
  bad_hosts_.erase(iter, bad_hosts_.end());
}

REGISTER_IMPL(RpcManager, GrpcManager);

}  // namespace client
}  // namespace euler
