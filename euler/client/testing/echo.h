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

#ifndef EULER_CLIENT_TESTING_ECHO_H_
#define EULER_CLIENT_TESTING_ECHO_H_

#include <string>

#include "grpcpp/grpcpp.h"

#include "euler/client/testing/echo.grpc.pb.h"

namespace euler {
namespace client {
namespace testing {

class EchoServiceImpl : public EchoService::Service {
 public:
  EchoServiceImpl() : host_port_("127.0.0.1:" + std::to_string(50051)) { }

  grpc::Status Echo(grpc::ServerContext* /*context*/,
                    const EchoRequest* request,
                    EchoResponse* response) override {
    response->set_message(request->message());
    return grpc::Status::OK;
  }

  std::string host_port() { return host_port_; }

  void BuildAndStart() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(host_port_, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();
  }

  void Shutdown() { server_->Shutdown(); }

 private:
  std::unique_ptr<grpc::Server> server_;
  std::string host_port_;
};

}  // namespace testing
}  // namespace client
}  // namespace euler

#endif  // EULER_CLIENT_TESTING_ECHO_H_
