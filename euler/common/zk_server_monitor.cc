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

#include "euler/common/zk_server_monitor.h"

#include <vector>
#include <utility>

#include "euler/common/logging.h"
#include "euler/common/str_util.h"
#include "euler/common/zk_util_cache.h"
#include "euler/common/server_meta.pb.h"

namespace euler {

namespace {

bool BytesToMeta(const std::string& bytes, Meta* meta, Meta* shard_meta) {
  if (bytes.empty()) {
    return true;
  }

  ServerMeta metas;
  if (!metas.ParseFromString(bytes)) {
    EULER_LOG(INFO) << "Server meta parsed failed, bytes size: "
                    << bytes.size();
    return false;
  }

  for (auto& item : metas.items()) {
    if (item.type() == 0) {
      meta->emplace(item.key(), item.value());
    } else if (item.type() == 1) {
      shard_meta->emplace(item.key(), item.value());
    } else {
      EULER_LOG(ERROR) << "Invalid MetaItem type: " << item.type();
      return false;
    }
  }

  return true;
}

bool BytesToShard(const std::string &bytes,
                  size_t *shard_index, Server *server) {
  std::vector<std::string> parts = Split(bytes, '#');
  if (parts.size() != 2) {
    return false;
  }

  try {
    *shard_index = std::stoul(parts[0]);
  } catch (std::invalid_argument e) {
    return false;
  }
  *server = std::move(parts[1]);
  return true;
}

template <typename t>
void SetDifference(const std::unordered_set<t> &input1,
                    const std::unordered_set<t> &input2,
                    std::function<void(const t &)> fn) {
  for (const t &element : input1) {
    if (input2.find(element) == input2.end()) {
      fn(element);
    }
  }
}

void ZkLogCallback(const char * /*message*/) { }

}  // namespace

bool ZkServerMonitor::Initialize() {
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

  int rc = zoo_awexists(zk_handle_, zk_path_.c_str(), RootWatcher, this,
                        RootCallback, this);
  if (rc != ZOK) {
    EULER_LOG(ERROR) << "ZK error when checking root node: "
                     << zerror(rc) << ".";
  }

  return true;
}

ZkServerMonitor::~ZkServerMonitor() {
  std::lock_guard<std::mutex> lock(zk_mu_);
  zookeeper_close(zk_handle_);
  zk_handle_ = nullptr;
}

void ZkServerMonitor::Watcher(
    zhandle_t *zh, int /*type*/, int state, const char * /*path*/, void *data) {
  if (state == ZOO_EXPIRED_SESSION_STATE) {
    zookeeper_close(zh);

    ZkServerMonitor *self = static_cast<ZkServerMonitor *>(data);
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

    int rc = zoo_awexists(self->zk_handle_, self->zk_path_.c_str(), RootWatcher,
                          self, RootCallback, self);
    if (rc != ZOK) {
      EULER_LOG(ERROR) << "ZK error when checking root node: "
                       << zerror(rc) << ".";
    }
  }
}

void ZkServerMonitor::RootCallback(
    int rc, const struct Stat * /*stat*/, const void *data) {
  if (rc == ZOK) {
    ZkServerMonitor* self = const_cast<ZkServerMonitor*>(
        reinterpret_cast<const ZkServerMonitor*>(data));
    int rc = zoo_awget_children(self->zk_handle_, self->zk_path_.c_str(),
                                ChildWatcher, self, ChildCallback, self);
    if (rc != ZOK) {
      EULER_LOG(ERROR) << "ZK error when watching child: " << zerror(rc) << ".";
    }
  } else if (rc == ZNONODE) {
  } else {
    EULER_LOG(ERROR) << "ZK error when checking root node: "
                     << zerror(rc) << ".";
  }
}

void ZkServerMonitor::RootWatcher(
    zhandle_t * /*zh*/, int type, int /*state*/, const char * /*path*/,
    void *data) {
  if (type == ZOO_CREATED_EVENT) {
    ZkServerMonitor *self = const_cast<ZkServerMonitor*>(
        reinterpret_cast<const ZkServerMonitor*>(data));
    int rc = zoo_awget_children(self->zk_handle_, self->zk_path_.c_str(),
                                ChildWatcher, self, ChildCallback, self);
    if (rc != ZOK) {
      EULER_LOG(ERROR) << "ZK error when watching child: " << zerror(rc) << ".";
    }
  } else if (type == ZOO_CHANGED_EVENT || ZOO_DELETED_EVENT) {
    // These events may be fired due to meta node already exist when first
    // queried with zoo_wexists, then changed for some reason. Low version of
    // zookeeper doesn't support watch removal.
    EULER_LOG(INFO) << "ZK receive watch event on root with code: "
                    << type << ".";
  } else {
    EULER_LOG(INFO) << "not support";
  }
}

void ZkServerMonitor::ChildCallback(
    int rc, const struct String_vector *strings, const void *data) {
  if (rc == ZOK) {
    ZkServerMonitor *self = const_cast<ZkServerMonitor*>(
        reinterpret_cast<const ZkServerMonitor*>(data));
    std::unordered_set<std::string> new_children(
        strings->data, strings->data + strings->count);

    using std::placeholders::_1;
    SetDifference<std::string>(
        new_children, self->children_,
        std::bind(&ZkServerMonitor::OnAddChild, self, _1));
    SetDifference<std::string>(
        self->children_, new_children,
        std::bind(&ZkServerMonitor::OnRemoveChild, self, _1));

    self->children_ = std::move(new_children);
  } else if (rc == ZNONODE) {
    // ZOO_DELETED_EVENT should be fired to watcher.
    EULER_LOG(WARNING) << "ZK root node get deleted.";
  } else {
    EULER_LOG(ERROR) << "ZK error when watching root node: "
                     << zerror(rc) << ".";
  }
}

void ZkServerMonitor::ChildWatcher(
    zhandle_t * /*zh*/, int type, int /*state*/, const char * /*path*/,
    void *data) {
  ZkServerMonitor *self = static_cast<ZkServerMonitor *>(data);
  if (type == ZOO_CHILD_EVENT) {
    int rc = zoo_awget_children(self->zk_handle_, self->zk_path_.c_str(),
                                ChildWatcher, self, ChildCallback, self);
    if (rc != ZOK) {
      EULER_LOG(ERROR) << "ZK error when watching root node: "
                       << zerror(rc) << ".";
    }
  } else if (type == ZOO_DELETED_EVENT) {
    int rc = zoo_awexists(self->zk_handle_, self->zk_path_.c_str(), RootWatcher,
                          self, RootCallback, self);
    if (rc != ZOK) {
      EULER_LOG(ERROR) << "ZK error when checking root node: "
                       << zerror(rc) << ".";
    }
  }
}

using ZkShardClosure = std::pair<ZkServerMonitor *, size_t>;

void ZkServerMonitor::MetaCallback(
    int rc, const char *value, int value_len, const struct Stat * /*stat*/,
    const void *data) {
  if (rc == ZOK) {
    std::unique_ptr<ZkShardClosure> closure(
        const_cast<ZkShardClosure*>(
            reinterpret_cast<const ZkShardClosure*>(data)));
    ZkServerMonitor *self = closure->first;
    size_t shard_index = closure->second;
    std::string meta_bytes(value, value_len);
    Meta meta, shard_meta;
    BytesToMeta(meta_bytes, &meta, &shard_meta);
    self->UpdateMeta(meta);
    self->UpdateShardMeta(shard_index, shard_meta);
  } else if (rc == ZNONODE) {
  } else {
    EULER_LOG(ERROR) << "ZK error when reading meta: " << zerror(rc) << ".";
  }
}

void ZkServerMonitor::OnAddChild(const std::string &child) {
  EULER_LOG(INFO) << "Online node: " << child << ".";

  size_t shard_index;
  Server server;
  if (BytesToShard(child, &shard_index, &server)) {
    zoo_aget(zk_handle_, JoinString({zk_path_, child}, "/").c_str(), 0,
             MetaCallback, new ZkShardClosure(this, shard_index));

    AddShardServer(shard_index, server);
  } else {
    EULER_LOG(WARNING) << "Invalid ZK child: " << child << ".";
  }
}

void ZkServerMonitor::OnRemoveChild(const std::string &child) {
  EULER_LOG(INFO) << "Offline node: " << child << ".";

  size_t shard_index;
  Server server;
  if (BytesToShard(child, &shard_index, &server)) {
    RemoveShardServer(shard_index, server);
  }
}

std::shared_ptr<ServerMonitor> GetServerMonitor(const std::string& zk_addr,
                                                const std::string& zk_path) {
  return GetOrCreate<ZkServerMonitor>(zk_addr, zk_path);
}

}  // namespace euler
