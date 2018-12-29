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

#ifndef EULER_SERVICE_CALL_DATA_H_
#define EULER_SERVICE_CALL_DATA_H_

#include <grpcpp/grpcpp.h>

#include "euler/proto/graph_service.grpc.pb.h"
#include "euler/core/graph_engine.h"

namespace euler {
namespace service {
class CallData {
 public:
  CallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine)
      : async_service_(service), cq_(cq), status_(CREATE),
      graph_engine_(graph_engine) {
  }

  virtual void Proceed() = 0;

  virtual ~CallData() {}

 protected:
  euler::proto::GraphService::AsyncService* async_service_;

  grpc::ServerCompletionQueue* cq_;

  grpc::ServerContext ctx_;

  enum CallStatus {CREATE, PROCESS, FINISH};
  CallStatus status_;

  std::shared_ptr<euler::core::GraphEngine> graph_engine_;
};

class SampleNodeCallData : public CallData {
 public:
  SampleNodeCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::SampleNodeRequest request_;

  euler::proto::SampleNodeReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::SampleNodeReply> responder_;
};

class SampleEdgeCallData : public CallData {
 public:
  SampleEdgeCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::SampleEdgeRequest request_;

  euler::proto::SampleEdgeReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::SampleEdgeReply> responder_;
};

class GetNodeTypeCallData : public CallData {
 public:
  GetNodeTypeCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetNodeTypeRequest request_;

  euler::proto::GetTypeReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::GetTypeReply> responder_;
};

class GetNodeFloat32FeatureCallData : public CallData {
 public:
  GetNodeFloat32FeatureCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetNodeFloat32FeatureRequest request_;

  euler::proto::GetFloat32FeatureReply reply_;

  grpc::ServerAsyncResponseWriter<
      euler::proto::GetFloat32FeatureReply> responder_;
};

class GetNodeUInt64FeatureCallData : public CallData {
 public:
  GetNodeUInt64FeatureCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetNodeUInt64FeatureRequest request_;

  euler::proto::GetUInt64FeatureReply reply_;

  grpc::ServerAsyncResponseWriter<
      euler::proto::GetUInt64FeatureReply> responder_;
};

class GetNodeBinaryFeatureCallData : public CallData {
 public:
  GetNodeBinaryFeatureCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetNodeBinaryFeatureRequest request_;

  euler::proto::GetBinaryFeatureReply reply_;

  grpc::ServerAsyncResponseWriter<
      euler::proto::GetBinaryFeatureReply> responder_;
};

class GetEdgeFloat32FeatureCallData : public CallData {
 public:
  GetEdgeFloat32FeatureCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetEdgeFloat32FeatureRequest request_;

  euler::proto::GetFloat32FeatureReply reply_;

  grpc::ServerAsyncResponseWriter<
      euler::proto::GetFloat32FeatureReply> responder_;
};

class GetEdgeUInt64FeatureCallData : public CallData {
 public:
  GetEdgeUInt64FeatureCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetEdgeUInt64FeatureRequest request_;

  euler::proto::GetUInt64FeatureReply reply_;

  grpc::ServerAsyncResponseWriter<
      euler::proto::GetUInt64FeatureReply> responder_;
};

class GetEdgeBinaryFeatureCallData : public CallData {
 public:
  GetEdgeBinaryFeatureCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetEdgeBinaryFeatureRequest request_;

  euler::proto::GetBinaryFeatureReply reply_;

  grpc::ServerAsyncResponseWriter<
      euler::proto::GetBinaryFeatureReply> responder_;
};

class GetFullNeighborCallData : public CallData {
 public:
  GetFullNeighborCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetFullNeighborRequest request_;

  euler::proto::GetNeighborReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::GetNeighborReply> responder_;
};

class GetSortedNeighborCallData : public CallData {
 public:
  GetSortedNeighborCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetSortedNeighborRequest request_;

  euler::proto::GetNeighborReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::GetNeighborReply> responder_;
};

class GetTopKNeighborCallData : public CallData {
 public:
  GetTopKNeighborCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::GetTopKNeighborRequest request_;

  euler::proto::GetNeighborReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::GetNeighborReply> responder_;
};

class SampleNeighborCallData : public CallData {
 public:
  SampleNeighborCallData(
      euler::proto::GraphService::AsyncService* service,
      grpc::ServerCompletionQueue* cq,
      std::shared_ptr<euler::core::GraphEngine> graph_engine) :
      CallData(service, cq, graph_engine), responder_(&ctx_) {
  }

  void Proceed();

 private:
  euler::proto::SampleNeighborRequest request_;

  euler::proto::GetNeighborReply reply_;

  grpc::ServerAsyncResponseWriter<euler::proto::GetNeighborReply> responder_;
};

}  // namespace service
}  // namespace euler
#endif  // EULER_SERVICE_CALL_DATA_H_
