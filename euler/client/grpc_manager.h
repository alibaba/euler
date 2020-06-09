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

#ifndef EULER_CLIENT_GRPC_MANAGER_H_
#define EULER_CLIENT_GRPC_MANAGER_H_

#include <string>
#include <memory>

#include "euler/client/rpc_manager.h"

namespace euler {

class GrpcManager : public RpcManager {
 public:
  GrpcManager() : RpcManager() { }

  std::unique_ptr<RpcChannel> CreateChannel(
       const std::string &host_port, int tag) override;

  RpcContext *CreateContext(
      const std::string &method,
      google::protobuf::Message *respone,
      std::function<void(const Status &)> done) override;
};

}  // namespace euler

#endif  // EULER_CLIENT_GRPC_MANAGER_H_
