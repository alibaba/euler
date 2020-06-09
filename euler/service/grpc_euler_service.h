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


#ifndef EULER_SERVICE_GRPC_EULER_SERVICE_H_
#define EULER_SERVICE_GRPC_EULER_SERVICE_H_

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/support/byte_buffer.h"

namespace euler {

enum EulerServiceMethod {
  kPing,
  kExecute,
  kSampleNode,
  kSampleEdge,
  kGetNodeType,
  kGetNodeFloat32Feature,
  kGetNodeUInt64Feature,
  kGetNodeBinaryFeature,
  kGetEdgeFloat32Feature,
  kGetEdgeUInt64Feature,
  kGetEdgeBinaryFeature,
  kGetFullNeighbor,
  kGetSortedNeighbor,
  kGetTopKNeighbor,
  kSampleNeighbor
};

static const int kMethodNum =
    static_cast<int>(EulerServiceMethod::kSampleNeighbor) + 1;

const char* EulerServiceMethodName(EulerServiceMethod id);

class EulerService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService() { }

    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace euler

#endif  // EULER_SERVICE_GRPC_EULER_SERVICE_H_
