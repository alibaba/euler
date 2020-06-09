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

#include "euler/common/zk_server_register.h"

#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "euler/common/logging.h"
#include "euler/common/str_util.h"
#include "euler/common/zk_util_cache.h"
#include "euler/common/server_meta.pb.h"

namespace euler {

namespace {

std::string MetaToBytes(const Meta& meta, const Meta& shard_meta) {
  ServerMeta metas;

  for (auto& me : meta) {
    auto item = metas.add_items();
    item->set_type(0);
    item->set_key(me.first);
    item->set_value(me.second);
  }

  for (auto& me : shard_meta) {
    auto item = metas.add_items();
    item->set_type(1);
    item->set_key(me.first);
    item->set_value(me.second);
  }

  std::string content;
  metas.SerializeToString(&content);
  return content;
}

std::string ShardToBytes(size_t shard_index, const Server server) {
  return JoinString({std::to_string(shard_index), server}, "#");
}

void ZkLogCallback(const char * /*message*/) { }

}  // namespace

bool ZkServerRegister::Initialize() {
  {
    std::lock_guard<std::mutex> lock(zk_mu_);

    if (zk_handle_) {
      return true;
    }

    zhandle_t *zh = zookeeper_init2(zk_addr_.c_str(), Watcher, 60000, nullptr,
                                    this, 0, ZkLogCallback);
    if (zh == nullptr) {
      EULER_LOG(ERROR) << "Fail to initialize ZK connection.";
      return false;
    }
    zk_handle_ = zh;
  }

  int rc = zoo_create(zk_handle_, zk_path_.c_str(), "", 0, &ZOO_OPEN_ACL_UNSAFE,
                      0, nullptr, 0);
  if (rc != ZOK && rc != ZNODEEXISTS) {
    EULER_LOG(ERROR) << "ZK error when creating root node: "
                     << zerror(rc) << ".";
  }

  return true;
}

ZkServerRegister::~ZkServerRegister() {
  std::lock_guard<std::mutex> lock(zk_mu_);
  zookeeper_close(zk_handle_);
  zk_handle_ = nullptr;
}

bool ZkServerRegister::RegisterShard(size_t shard_index, const Server &server,
                                     const Meta &meta, const Meta &shard_meta) {
  std::string shard_zk_child = ShardToBytes(shard_index, server);
  std::string shard_zk_path = JoinString({zk_path_, shard_zk_child}, "/");
  std::string shard_meta_bytes = MetaToBytes(meta, shard_meta);

  int rc = zoo_create(zk_handle_, shard_zk_path.c_str(),
                      shard_meta_bytes.c_str(), shard_meta_bytes.size(),
                      &ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL, nullptr, 0);
  if (rc != ZOK) {
    EULER_LOG(ERROR) << "ZK error when creating meta: " << zerror(rc) << ".";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    registered_.emplace(shard_zk_path, shard_meta_bytes);
  }
  return true;
}

bool ZkServerRegister::DeregisterShard(size_t shard_index,
                                       const Server &server) {
  std::string shard_zk_child = ShardToBytes(shard_index, server);
  std::string shard_zk_path = JoinString({zk_path_, shard_zk_child}, "/");

  int rc = zoo_delete(zk_handle_, shard_zk_path.c_str(), -1);
  if (rc != ZOK) {
    EULER_LOG(ERROR) << "ZK error when deleting meta: " << zerror(rc) << ".";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    registered_.erase(shard_zk_child);
  }
  return true;
}

void ZkServerRegister::Watcher(zhandle_t *zh, int /*type*/, int state,
                               const char * /*path*/, void *data) {
  if (state == ZOO_EXPIRED_SESSION_STATE) {
    zookeeper_close(zh);

    ZkServerRegister *self = static_cast<ZkServerRegister *>(data);
    {
      std::lock_guard<std::mutex> lock(self->zk_mu_);

      self->zk_handle_ = nullptr;
      while (self->zk_handle_ == nullptr) {
        EULER_LOG(WARNING) << "Reconnecting ZK ...";
        self->zk_handle_ = zookeeper_init2(self->zk_addr_.c_str(), Watcher,
                                           60000, nullptr, self, 0,
                                           ZkLogCallback);
      }
    }

    {
      std::lock_guard<std::mutex> lock(self->mu_);

      for (const auto &registered : self->registered_) {
        const std::string &shard_zk_path = registered.first;
        const std::string &shard_meta_bytes = registered.second;
        int rc = zoo_create(self->zk_handle_, shard_zk_path.c_str(),
                            shard_meta_bytes.c_str(), shard_meta_bytes.size(),
                            &ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL, nullptr, 0);
        if (rc != ZOK) {
          EULER_LOG(ERROR) << "ZK error when creating meta: "
                           << zerror(rc) << ".";
        }
      }
    }
  }
}

std::shared_ptr<ServerRegister> GetServerRegister(const std::string& zk_addr,
                                                  const std::string& zk_path) {
  return GetOrCreate<ZkServerRegister>(zk_addr, zk_path);
}

}  // namespace euler
