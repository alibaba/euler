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


#ifndef EULER_SERVICE_GRPC_SERVER_H_
#define EULER_SERVICE_GRPC_SERVER_H_

#include <memory>
#include <string>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"

#include "euler/common/status.h"
#include "euler/common/env.h"
#include "euler/common/mutex.h"
#include "euler/service/server_interface.h"
#include "euler/service/async_service_interface.h"
#include "euler/service/grpc_worker.h"
#include "euler/service/grpc_worker_service.h"

namespace euler {

class GrpcWorker;

class GrpcServer: public ServerInterface {
 protected:
  GrpcServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* server);

  virtual ~GrpcServer();

  Status Start() override;
  Status Stop() override;
  Status Join() override;

 protected:
  Status Init();
  virtual std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials(
      const ServerDef& server_def) const;
  int bound_port() const { return bound_port_; }
  const ServerDef& server_def() const { return server_def_; }

  Status LoadGraphAndIndex();

 private:
  const ServerDef server_def_;
  Env* env_;
  int bound_port_ = 0;
  Mutex mu_;

  enum State {NEW, STARTED, STOPPED};
  State state_;

  WorkerEnv worker_env_;
  std::unique_ptr<GrpcWorker> worker_impl_;
  AsyncServiceInterface* worker_service_ = nullptr;
  std::unique_ptr<Thread> worker_thread_;

  std::unique_ptr<::grpc::Server> server_;

  std::string host_port_;
  std::string zk_server_;
  std::string zk_path_;
};

}  // namespace euler

#endif  // EULER_SERVICE_GRPC_SERVER_H_
