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

#include "euler/common/zk_service.h"

#include <unistd.h>

#include <cstdlib>
#include <fstream>

#include "euler/common/net_util.h"

namespace euler {
namespace common {

bool ZkService::Start() {
  int port = common::GetFreePort();

  char zk_dir_template[] = "/tmp/euler_zk_XXXXXX";
  char *zk_dir = mkdtemp(zk_dir_template);
  if (zk_dir == nullptr) {
    return false;
  }

  {
    std::string zk_dir_s(zk_dir);
    std::ofstream os(zk_dir_s + "/zoo.cfg");
    os << "dataDir=" << zk_dir << std::endl;
    os << "clientPort=" << port << std::endl;
  }

  std::string path = __FILE__;
  std::string dir = path.substr(0, path.rfind('/'));
  std::string zk_server = dir + "/../../third_party/zookeeper/bin/zkServer.sh";
  zk_command_ = zk_server + " --config " + zk_dir;
  int ret = std::system((zk_command_ + " start").c_str());

  if (ret == 0) {
    port_ = port;
    sleep(5);
    return true;
  } else {
    return false;
  }
}

std::string ZkService::HostPort() {
  return "127.0.0.1:" + std::to_string(port_);
}

bool ZkService::Shutdown() {
  if (port_) {
    return std::system((zk_command_ + " stop").c_str()) == 0;
  } else {
    return false;
  }
}

}  // namespace common
}  // namespace euler
