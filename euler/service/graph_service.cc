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

#include "euler/service/graph_service.h"

#include <stdlib.h>

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <thread>

#include "grpc/support/log.h"
#include "grpcpp/resource_quota.h"

#include "euler/common/timmer.h"
#include "euler/common/server_register.h"
#include "euler/common/net_util.h"
#include "euler/service/call_data.h"

namespace euler {
namespace service {

GraphService::GraphService(const ServiceConfig& conf)
    : GraphService(0, conf) {
}

GraphService::GraphService(unsigned short port, const ServiceConfig& config) {
  auto conf = config;  // Copy for updating
  if (conf.find("server_thread_num") == conf.end() ||
      conf["server_thread_num"] == "") {
    thread_num_ = std::thread::hardware_concurrency() * 2;
  } else {
    thread_num_ = atoi(conf["server_thread_num"].c_str());
    thread_num_ = thread_num_ > 0 && thread_num_ < 1000 ? thread_num_ :
        std::thread::hardware_concurrency() * 2;
  }
  service_port_ = port;
  service_ip_port_ = euler::common::GetIP() +
                     ":" + std::to_string(service_port_);
  euler::core::GraphEngine* ptr = nullptr;
  if (conf["graph_type"] == "compact") {
    ptr = new euler::core::GraphEngine(euler::core::compact);
  } else {
    ptr = new euler::core::GraphEngine(euler::core::fast);
  }
  graph_engine_ = std::shared_ptr<euler::core::GraphEngine>(ptr);
  init_success_ = graph_engine_->Initialize(conf);
}

void GraphService::HandleRpcs(int32_t thread_idx) {
  CallData* call_data = new SampleNodeCallData(
      &async_service_,cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new SampleEdgeCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetNodeTypeCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetNodeFloat32FeatureCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetNodeUInt64FeatureCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetNodeBinaryFeatureCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetEdgeFloat32FeatureCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetEdgeUInt64FeatureCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetEdgeBinaryFeatureCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetFullNeighborCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetSortedNeighborCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new GetTopKNeighborCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  call_data = new SampleNeighborCallData(
      &async_service_, cq_list_[thread_idx].get(), graph_engine_);
  call_data->Proceed();
  void* tag;
  bool ok;
  while (true) {
    GPR_ASSERT(cq_list_[thread_idx]->Next(&tag, &ok));
    GPR_ASSERT(ok);
    static_cast<CallData*>(tag)->Proceed();
  }
}

void GraphService::Start(const ServiceConfig& config) {
  auto conf = config;  // Copy for updating
  if (init_success_) {
    grpc::ServerBuilder builder;

    int bound_port = 0;
    builder.AddListeningPort(service_ip_port_,
                             grpc::InsecureServerCredentials(),
                             &bound_port);
    builder.RegisterService(&async_service_);
    cq_list_.resize(thread_num_);
    for (int32_t i = 0; i < thread_num_; ++i) {
      cq_list_[i] = builder.AddCompletionQueue();
    }
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (service_port_ == 0) {
      service_port_ = bound_port;
      service_ip_port_ = euler::common::GetIP() +
                         ":" + std::to_string(service_port_);
      LOG(INFO) << "bound port: " << service_ip_port_;
    }
    // register ip:port, shard_idx, global sampler info
    auto server_register = euler::common::GetServerRegister(
        conf["zk_addr"], conf["zk_path"]);
    euler::common::Meta meta, shard_meta;
    std::stringstream partition_num_str;
    partition_num_str << graph_engine_->GetPartitionNum();
    meta["num_shards"] = conf["shard_num"];
    meta["num_partitions"] = partition_num_str.str();
    shard_meta["node_sum_weight"] = graph_engine_->GetNodeSumWeight();
    shard_meta["edge_sum_weight"] = graph_engine_->GetEdgeSumWeight();
    int32_t shard_idx = atoi(conf["shard_idx"].c_str());
    server_register->RegisterShard(shard_idx, service_ip_port_,
                                   meta, shard_meta);
    LOG(INFO) << "service start";
    // handle rpcs
    std::vector<std::thread> thread_list(thread_num_);
    for (int32_t i = 0; i < thread_num_; ++i) {
      thread_list[i] = std::thread(&GraphService::HandleRpcs, this, i);
    }

    for (int32_t i = 0; i < thread_num_; ++i) {
      thread_list[i].join();
    }
  } else {
    LOG(ERROR) << "service error";
  }
}

void StartService(
    const GraphService::ServiceConfig& conf) {
  if (conf.find("directory") == conf.end() ||
      conf.find("loader_type") == conf.end() ||
      conf.find("hdfs_addr") == conf.end() ||
      conf.find("hdfs_port") == conf.end() ||
      conf.find("shard_idx") == conf.end() ||
      conf.find("shard_num") == conf.end() ||
      conf.find("zk_addr") == conf.end() ||
      conf.find("zk_path") == conf.end() ||
      conf.find("global_sampler_type") == conf.end() ||
      conf.find("graph_type") == conf.end()) {
    LOG(ERROR) << "config error."
        "config contains:directory,loader_type,hdfs_addr,hdfs_port"
        "shard_idx,shard_num,zk_addr,zk_path,global_sampler_type,"
        "graph_type";
  } else {
    GraphService graph_service(conf);
    LOG(INFO) << "service init finish";
    graph_service.Start(conf);
  }
}

}  // namespace service
}  // namespace euler
