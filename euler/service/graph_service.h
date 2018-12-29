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

#ifndef EULER_SERVICE_GRAPH_SERVICE_H_
#define EULER_SERVICE_GRAPH_SERVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "grpcpp/grpcpp.h"

#include "euler/core/graph_engine.h"
#include "euler/proto/graph_service.grpc.pb.h"

namespace euler {
namespace service {

class GraphService {
 public:
  using ServiceConfig = std::unordered_map<std::string, std::string>;

  explicit GraphService(const ServiceConfig& conf);

  GraphService(unsigned short port, const ServiceConfig& config);

  virtual ~GraphService() {}

  void Start(const ServiceConfig& config);

 private:
  euler::proto::GraphService::AsyncService async_service_;

  std::string service_ip_port_;

  uint16_t service_port_;

  int32_t thread_num_;

  std::shared_ptr<euler::core::GraphEngine> graph_engine_;

  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> cq_list_;

  bool init_success_;

  void HandleRpcs(int32_t thread_idx);
};

void StartService(const GraphService::ServiceConfig& conf);

}  // namespace service
}  // namespace euler

#endif  // EULER_SERVICE_GRAPH_SERVICE_H_
