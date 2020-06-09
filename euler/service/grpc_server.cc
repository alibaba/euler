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


#include "euler/service/grpc_server.h"

#include <cstdio>
#include <limits>
#include <vector>
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"

#include "euler/common/net_util.h"
#include "euler/common/logging.h"
#include "euler/common/status.h"
#include "euler/common/server_register.h"
#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_builder.h"
#include "euler/core/index/index_manager.h"
#include "euler/common/env.h"

namespace euler {

GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)
    : server_def_(server_def), env_(env), state_(NEW) { }

GrpcServer::~GrpcServer() {
  if (worker_service_ != nullptr) {
    Stop();
    delete worker_service_;
    delete worker_env_.compute_pool;
  }
}

Status GrpcServer::Init() {
  MutexLock l(&mu_);
  worker_env_.env = env_;
  auto& options = server_def_.options;

  int port = 0;
  auto it = options.find("port");
  if (it  == options.end()) {
    EULER_LOG(ERROR) << "Invalid Server config, no port specified";
    return Status(ErrorCode::INTERNAL, "no port specified");
  }
  port = std::atoi(it->second.c_str());

  ::grpc::ServerBuilder builder;
  std::string address("0.0.0.0:");
  address += std::to_string(port);
  EULER_LOG(INFO) << "Bound grpc server to: " << address;
  builder.AddListeningPort(
      address, GetServerCredentials(server_def_), &bound_port_);
  if (bound_port_ == 0) {
    bound_port_ = port;
  }

  builder.SetMaxMessageSize(std::numeric_limits<int32_t>::max());
  worker_impl_ = NewGrpcWorker(&worker_env_);
  worker_service_ =
      NewGrpcWorkerService(worker_impl_.get(), &builder).release();

  server_ = builder.BuildAndStart();

  std::string name = "euler";
  int num_threads = 32;
  it = options.find("thread_pool");
  if (it != options.end()) {
    name = it->second;
  }
  it = options.find("num_threads");
  if (it != options.end()) {
    num_threads = std::atoi(it->second.c_str());
  }
  worker_env_.compute_pool = env_->StartThreadPool(name, num_threads);

  // Load Graph and Index
  RETURN_IF_ERROR(LoadGraphAndIndex());

  return Status::OK();
}

Status GrpcServer::LoadGraphAndIndex() {
  auto& options = server_def_.options;
  auto it = options.find("data_path");
  if (it == options.end()) {
    return Status::InvalidArgument(
        "Server options[data_path] must be specified!");
  }

  auto& data_path = it->second;

  std::unique_ptr<FileIO> data_dir;
  RETURN_IF_ERROR(Env::Default()->NewFileIO(data_path, true, &data_dir));
  if (!data_dir->initialized() || !data_dir->IsDirectory()) {
    EULER_LOG(ERROR) << "No such directory found, path: " << data_path;
    return Status::NotFound("Directory ", data_path, " not found!");
  }

  std::string sampler_type = "node";
  it = options.find("global_sampler_type");
  if (it != options.end()) {
    sampler_type = it->second;
  }

  std::string load_data_type = "node";
  it = options.find("load_data_type");
  if (it != options.end()) {
    load_data_type = it->second;
  }

  auto& graph = Graph::Instance();

  // Load Graph
  RETURN_IF_ERROR(graph.Init(server_def_.shard_index,
                             server_def_.shard_number,
                             sampler_type, data_path, load_data_type));

  it = options.find("zk_server");
  if (it == options.end()) {
    return Status::InvalidArgument(
        "Server options[zk_server] must be specified!");
  }
  zk_server_ = it->second;

  it = options.find("zk_path");
  if (it == options.end()) {
    return Status::InvalidArgument(
        "Server options[zk_path] must be specified!");
  }
  zk_path_ = it->second;

  host_port_ = GetIP();
  it = options.find("server");
  if (it != options.end()) {
    host_port_ = it->second;
  }
  host_port_ += ":" + std::to_string(bound_port_);

  auto& index_manager = IndexManager::Instance();
  index_manager.set_shard_index(server_def_.shard_index);
  index_manager.set_shard_number(server_def_.shard_number);
  if (!data_dir->ListDirectory([](const std::string &filename) {
        return filename == "Index";
      }).empty()) {
    std::string index_dir = JoinPath(data_path, "Index");
    index_manager.Deserialize(index_dir);
  } else {
    EULER_LOG(INFO) << "Missing Index directory, skip loading index.";
  }

  /* register info */
  std::vector<Meta> graph_metas = graph.GetRegisterInfo();
  std::vector<Meta> index_metas = index_manager.GetIndexInfo();
  // merge meta
  for (auto it = index_metas[0].begin(); it != index_metas[0].end(); ++it) {
    graph_metas[0][it->first] = it->second;
  }
  // merge shard meta
  for (auto it = index_metas[1].begin(); it != index_metas[1].end(); ++it) {
    graph_metas[1][it->first] = it->second;
  }
  auto server_register = GetServerRegister(zk_server_, zk_path_);
  if (server_register == nullptr) {
    EULER_LOG(ERROR) << "Get ServerRegister failed, zk_server:"
                     << zk_server_ << ", zk_path: " << zk_path_;
    return Status::Internal("Get ServerRegister failed, zk_server:",
                            zk_server_, ", zk_path: ", zk_path_);
  }
  if (!server_register->RegisterShard(
      server_def_.shard_index, host_port_, graph_metas[0], graph_metas[1])) {
    return Status::Internal("Register graph to zk failed");
  }
  return Status::OK();
}

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(
    const ServerDef&) const {
  return ::grpc::InsecureServerCredentials();
}

Status GrpcServer::Start() {
  MutexLock l(&mu_);
  switch (state_) {
    case NEW: {
      worker_thread_.reset(
          env_->StartThread("worker_service",
                            [this]() { worker_service_->Loop(); }));
      state_ = STARTED;
      EULER_LOG(INFO) << "Server started successfully!";
      return Status::OK();
    }
    case STARTED: {
      EULER_LOG(INFO) << "Server has already started!";
      return Status::OK();
    }
    case STOPPED: {
      return Status::OK();
    }
    default: {
      EULER_LOG(FATAL) << "Invalid State got";
      return Status(ErrorCode::ABORTED, "invalid state");
    }
  }
}

Status GrpcServer::Stop() {
  MutexLock l(&mu_);
  switch (state_) {
    case NEW: {
      state_ = STOPPED;
      return Status::OK();
    }
    case STARTED: {
      server_->Shutdown();
      worker_service_->Shutdown();
      worker_thread_->Join();
      worker_env_.compute_pool->Shutdown();
      state_ = STOPPED;

      Graph::Instance().DeregisterRemote(host_port_, zk_server_, zk_path_);

      EULER_LOG(INFO) << "Server shutdown successfully!";
      return Status::OK();
    }
    case STOPPED: {
      return Status::OK();
    }
    default: {
      EULER_LOG(FATAL) << "Invalid state got";
      return  Status(ErrorCode::ABORTED, "invalid state");
    }
  }
}

Status GrpcServer::Join() {
  MutexLock l(&mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      worker_thread_.reset();
      return Status::OK();
    default: {
      EULER_LOG(FATAL) << "Invalid state got";
      return  Status(ErrorCode::ABORTED, "invalid state");
    }
  }
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<GrpcServer> ret(
      new GrpcServer(server_def, env == nullptr ? Env::Default() : env));
  RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol == "grpc";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return GrpcServer::Create(server_def, Env::Default(), out_server);
  }
};

class GrpcServerRegistrar {
 public:
  GrpcServerRegistrar() {
    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory());
  }
};
static GrpcServerRegistrar grpc_server_register;

}  // namespace

}  // namespace euler
