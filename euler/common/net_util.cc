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

#include "euler/common/net_util.h"

#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <string>

#include "glog/logging.h"

namespace euler {
namespace common {

std::string GetIP() {
  std::string ip = "";
  char hname[128];
  struct hostent *hent;
  gethostname(hname, sizeof(hname));
  hent = gethostbyname(hname);
  if (hent -> h_addr_list[0]) {
    std::string ip_addr = inet_ntoa(*(struct in_addr*)(hent -> h_addr_list[0]));
    ip = ip_addr;
  }
  return ip;
}

uint16_t GetFreePort() {
  uint16_t port = 0;
  int32_t sock_fd;
  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    LOG(ERROR) << "error opening socket";
  }

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = 0;

  int32_t ret = bind(sock_fd, (struct sockaddr *)&addr, sizeof addr);
  if (ret < 0) {
    LOG(ERROR) << "error on binding";
  }
  struct sockaddr_in conn_addr;
  unsigned int len = sizeof conn_addr;
  ret = getsockname(sock_fd, (struct sockaddr *)&conn_addr, &len);
  port = ntohs(conn_addr.sin_port);
  close(sock_fd);
  return port;
}

}  // namespace common
}  // namespace euler
