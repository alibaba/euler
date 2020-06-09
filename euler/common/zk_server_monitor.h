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

#ifndef EULER_COMMON_ZK_SERVER_MONITOR_H_
#define EULER_COMMON_ZK_SERVER_MONITOR_H_

#include <unordered_set>
#include <string>
#include <memory>

#include "zookeeper.h"  // NOLINT

#include "euler/common/server_monitor.h"
#include "euler/common/zk_util_cache.h"

namespace euler {

class ZkServerMonitor : public ServerMonitorBase {
  friend std::shared_ptr<ZkServerMonitor> GetOrCreate<ZkServerMonitor>(
      const std::string &, const std::string &);

 public:
  ~ZkServerMonitor() override;

 private:
  static void Watcher(zhandle_t *zh, int type, int state, const char *path,
                      void *data);
  static void RootCallback(int rc, const struct Stat *stat, const void *data);
  static void RootWatcher(zhandle_t *zk_handle, int type, int state,
                          const char *path, void *data);
  static void ChildCallback(int rc, const struct String_vector *strings,
                            const void *data);
  static void ChildWatcher(zhandle_t *zh, int type, int state,
                           const char *path, void *data);
  static void MetaCallback(int rc, const char *value, int value_len,
                           const struct Stat *stat, const void *data);

  ZkServerMonitor(std::string zk_addr, std::string zk_path)
    : zk_addr_(zk_addr), zk_path_(zk_path), zk_handle_(nullptr) { }

  bool Initialize() override;

  void OnAddChild(const std::string &child);
  void OnRemoveChild(const std::string &child);

  std::string zk_addr_;
  std::string zk_path_;

  std::mutex zk_mu_;
  zhandle_t *zk_handle_;
  std::unordered_set<std::string> children_;
};

}  // namespace euler

#endif  // EULER_COMMON_ZK_SERVER_MONITOR_H_
