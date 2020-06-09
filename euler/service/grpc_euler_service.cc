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


#include "euler/service/grpc_euler_service.h"

#include "grpcpp/grpcpp.h"

#include "euler/common/logging.h"

namespace euler {

const char* EulerServiceMethodName(EulerServiceMethod id) {
  switch (id) {
    case EulerServiceMethod::kPing:
      return "euler.EulerService/Ping";
    case EulerServiceMethod::kExecute:
      return "euler.EulerService/Execute";
    case EulerServiceMethod::kSampleNode:
      return "euler.EulerService/SampleNode";
    case EulerServiceMethod::kSampleEdge:
      return "euler.EulerService/SampleEdge";
    case EulerServiceMethod::kGetNodeType:
      return "euler.EulerService/GetNodeType";
    case EulerServiceMethod::kGetNodeFloat32Feature:
      return "euler.EulerService/GetNodeFloat32Feature";
    case EulerServiceMethod::kGetNodeUInt64Feature:
      return "euler.EulerService/GetNodeUInt64Feature";
    case EulerServiceMethod::kGetNodeBinaryFeature:
      return "euler.EulerService/GetNodeBinaryFeature";
    case EulerServiceMethod::kGetEdgeFloat32Feature:
      return "euler.EulerService/GetEdgeFloat32Feature";
    case EulerServiceMethod::kGetEdgeUInt64Feature:
      return "euler.EulerService/GetEdgeUInt64Feature";
    case EulerServiceMethod::kGetEdgeBinaryFeature:
      return "euler.EulerService/GetEdgeBinaryFeature";
    case EulerServiceMethod::kGetFullNeighbor:
      return "euler.EulerService/GetFullNeighbor";
    case EulerServiceMethod::kGetSortedNeighbor:
      return "euler.EulerService/GetSortedNeighbor";
    case EulerServiceMethod::kGetTopKNeighbor:
      return "euler.EulerService/GetTopKNeighbor";
    case EulerServiceMethod::kSampleNeighbor:
      return "euler.EulerService/SampleNeighbor";
  }

  EULER_LOG(FATAL) << "Invalid id: this line shouldn't be reached.";
  return "Invalid Method Id";
}

EulerService::AsyncService::AsyncService() {
  for (int i = 0; i < kMethodNum; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        EulerServiceMethodName(static_cast<EulerServiceMethod>(i)),
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

}  // namespace euler
