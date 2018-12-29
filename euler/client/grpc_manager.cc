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

#include "euler/client/grpc_manager.h"

#include <string>

#include "euler/client/grpc_channel.h"

namespace euler {
namespace client {

std::unique_ptr<RpcChannel> GrpcManager::CreateChannel(
    const std::string &host_port, int tag) {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  args.SetInt("tag", tag);
  std::shared_ptr<grpc::Channel> raw_channel =
      grpc::CreateCustomChannel(host_port,
                                grpc::InsecureChannelCredentials(),
                                args);
  return std::unique_ptr<RpcChannel>(new GrpcChannel(host_port, raw_channel));
}

RpcContext *GrpcManager::CreateContext(
    const std::string &method, google::protobuf::Message *respone,
    std::function<void(const Status &)> done) {
  return new GrpcContext(method, respone, done);
}

}  // namespace client
}  // namespace euler
