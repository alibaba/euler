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

#include "euler/service/call_data.h"

#include <vector>
#include <algorithm>

#include "euler/common/data_types.h"

namespace euler {
namespace service {
void SampleNodeCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestSampleNode(&ctx_, &request_,
                                      &responder_, cq_,
                                      cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new SampleNodeCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    int32_t node_type = request_.node_type();
    int32_t count = request_.count();
    std::vector<euler::common::NodeID> results =
        graph_engine_->SampleNode(node_type, count);
    for (size_t i = 0; i < results.size(); ++i) {
      reply_.add_node_ids(results[i]);
    }

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void SampleEdgeCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestSampleEdge(&ctx_, &request_,
                                      &responder_, cq_,
                                      cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new SampleEdgeCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    int32_t edge_type = request_.edge_type();
    int32_t count = request_.count();
    std::vector<euler::common::EdgeID> results =
        graph_engine_->SampleEdge(edge_type, count);
    for (size_t i = 0; i < results.size(); ++i) {
      ::euler::proto::EdgeID* edge_id = reply_.add_edge_ids();
      edge_id->set_src_node(std::get<0>(results[i]));
      edge_id->set_dst_node(std::get<1>(results[i]));
      edge_id->set_type(std::get<2>(results[i]));
    }

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetNodeTypeCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetNodeType(&ctx_, &request_,
                                       &responder_, cq_,
                                       cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetNodeTypeCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> types = graph_engine_->GetNodeType(node_ids);

    reply_.mutable_types()->Resize(types.size(), 0);
    std::copy(types.begin(), types.end(),
        reply_.mutable_types()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetNodeFloat32FeatureCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetNodeFloat32Feature(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetNodeFloat32FeatureCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> feature_ids(request_.feature_ids_size());
    std::copy(request_.feature_ids().begin(), request_.feature_ids().end(),
              feature_ids.begin());

    std::vector<uint32_t> feature_nums;
    std::vector<float> feature_values;
    graph_engine_->GetNodeFloat32Feature(node_ids, feature_ids, &feature_nums,
                                         &feature_values);
    reply_.mutable_value_nums()->Resize(feature_nums.size(), 0);
    std::copy(feature_nums.begin(), feature_nums.end(),
              reply_.mutable_value_nums()->begin());
    reply_.mutable_values()->Resize(feature_values.size(), 0);
    std::copy(feature_values.begin(), feature_values.end(),
              reply_.mutable_values()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetNodeUInt64FeatureCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetNodeUInt64Feature(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
      new GetNodeUInt64FeatureCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> feature_ids(request_.feature_ids_size());
    std::copy(request_.feature_ids().begin(), request_.feature_ids().end(),
              feature_ids.begin());

    std::vector<uint32_t> feature_nums;
    std::vector<uint64_t> feature_values;
    graph_engine_->GetNodeUint64Feature(node_ids, feature_ids, &feature_nums,
                                        &feature_values);
    reply_.mutable_value_nums()->Resize(feature_nums.size(), 0);
    std::copy(feature_nums.begin(), feature_nums.end(),
              reply_.mutable_value_nums()->begin());
    reply_.mutable_values()->Resize(feature_values.size(), 0);
    std::copy(feature_values.begin(), feature_values.end(),
              reply_.mutable_values()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetNodeBinaryFeatureCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetNodeBinaryFeature(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetNodeBinaryFeatureCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> feature_ids(request_.feature_ids_size());
    std::copy(request_.feature_ids().begin(), request_.feature_ids().end(),
              feature_ids.begin());

    std::vector<uint32_t> feature_nums;
    std::vector<char> feature_values;
    graph_engine_->GetNodeBinaryFeature(node_ids, feature_ids, &feature_nums,
                                        &feature_values);
    reply_.mutable_value_nums()->Resize(feature_nums.size(), 0);
    std::copy(feature_nums.begin(), feature_nums.end(),
              reply_.mutable_value_nums()->begin());
    reply_.mutable_values()->resize(feature_values.size());
    std::copy(feature_values.begin(), feature_values.end(),
              reply_.mutable_values()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetEdgeFloat32FeatureCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetEdgeFloat32Feature(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetEdgeFloat32FeatureCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::EdgeID> edge_ids(request_.edge_ids_size());
    for (int i = 0; i < request_.edge_ids_size(); ++i) {
      edge_ids[i] = euler::common::EdgeID(request_.edge_ids(i).src_node(),
                                          request_.edge_ids(i).dst_node(),
                                          request_.edge_ids(i).type());
    }
    std::vector<int32_t> feature_ids(request_.feature_ids_size());
    std::copy(request_.feature_ids().begin(), request_.feature_ids().end(),
            feature_ids.begin());

    std::vector<uint32_t> feature_nums;
    std::vector<float> feature_values;
    graph_engine_->GetEdgeFloat32Feature(edge_ids, feature_ids, &feature_nums,
                                         &feature_values);
    // set response
    reply_.mutable_value_nums()->Resize(feature_nums.size(), 0);
    std::copy(feature_nums.begin(), feature_nums.end(),
              reply_.mutable_value_nums()->begin());
    reply_.mutable_values()->Resize(feature_values.size(), 0);
    std::copy(feature_values.begin(), feature_values.end(),
              reply_.mutable_values()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetEdgeUInt64FeatureCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetEdgeUInt64Feature(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetEdgeUInt64FeatureCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::EdgeID> edge_ids(request_.edge_ids_size());
    for (int i = 0; i < request_.edge_ids_size(); ++i) {
      edge_ids[i] = euler::common::EdgeID(request_.edge_ids(i).src_node(),
                                          request_.edge_ids(i).dst_node(),
                                          request_.edge_ids(i).type());
    }
    std::vector<int32_t> feature_ids(request_.feature_ids_size());
    std::copy(request_.feature_ids().begin(), request_.feature_ids().end(),
              feature_ids.begin());

    std::vector<uint32_t> feature_nums;
    std::vector<uint64_t> feature_values;
    graph_engine_->GetEdgeUint64Feature(edge_ids, feature_ids, &feature_nums,
                                        &feature_values);
    reply_.mutable_value_nums()->Resize(feature_nums.size(), 0);
    std::copy(feature_nums.begin(), feature_nums.end(),
              reply_.mutable_value_nums()->begin());
    reply_.mutable_values()->Resize(feature_values.size(), 0);
    std::copy(feature_values.begin(), feature_values.end(),
              reply_.mutable_values()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetEdgeBinaryFeatureCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetEdgeBinaryFeature(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetEdgeBinaryFeatureCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::EdgeID> edge_ids(request_.edge_ids_size());
    for (int i = 0; i < request_.edge_ids_size(); ++i) {
      edge_ids[i] = euler::common::EdgeID(request_.edge_ids(i).src_node(),
                                          request_.edge_ids(i).dst_node(),
                                          request_.edge_ids(i).type());
    }
    std::vector<int32_t> feature_ids(request_.feature_ids_size());
    std::copy(request_.feature_ids().begin(), request_.feature_ids().end(),
              feature_ids.begin());

    std::vector<uint32_t> feature_nums;
    std::vector<char> feature_values;
    graph_engine_->GetEdgeBinaryFeature(edge_ids, feature_ids, &feature_nums,
                                        &feature_values);
    // set response
    reply_.mutable_value_nums()->Resize(feature_nums.size(), 0);
    std::copy(feature_nums.begin(), feature_nums.end(),
              reply_.mutable_value_nums()->begin());
    reply_.mutable_values()->resize(feature_values.size());
    std::copy(feature_values.begin(), feature_values.end(),
              reply_.mutable_values()->begin());

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

#define ASYNC_REPLY(METHOD, NODE_IDS, EDGE_TYPES, REPLY, ...) {   \
  std::vector<euler::common::IDWeightPair> neighbors;             \
  std::vector<uint32_t> neighbor_nums;                            \
  graph_engine_->METHOD(NODE_IDS, EDGE_TYPES, ##__VA_ARGS__,      \
                        &neighbors, &neighbor_nums);              \
  REPLY.mutable_neighbor_nums()->Resize(neighbor_nums.size(), 0); \
  std::copy(neighbor_nums.begin(), neighbor_nums.end(),           \
            REPLY.mutable_neighbor_nums()->begin());              \
  REPLY.mutable_node_ids()->Reserve(neighbors.size());            \
  REPLY.mutable_weights()->Reserve(neighbors.size());             \
  for (size_t i = 0; i < neighbors.size(); ++i) {                 \
    REPLY.add_node_ids(std::get<0>(neighbors[i]));                \
    REPLY.add_weights(std::get<1>(neighbors[i]));                 \
    REPLY.add_types(std::get<2>(neighbors[i]));                   \
  }                                                               \
}                                                                 \

void GetFullNeighborCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetFullNeighbor(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetFullNeighborCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> edge_types(request_.edge_types_size());
    std::copy(request_.edge_types().begin(), request_.edge_types().end(),
              edge_types.begin());
    ASYNC_REPLY(GetFullNeighbor, node_ids, edge_types, reply_);

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetSortedNeighborCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetSortedNeighbor(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetSortedNeighborCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> edge_types(request_.edge_types_size());
    std::copy(request_.edge_types().begin(), request_.edge_types().end(),
              edge_types.begin());
    ASYNC_REPLY(GetSortedFullNeighbor, node_ids, edge_types, reply_);

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void GetTopKNeighborCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestGetTopKNeighbor(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new GetTopKNeighborCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> edge_types(request_.edge_types_size());
    std::copy(request_.edge_types().begin(), request_.edge_types().end(),
              edge_types.begin());
    int32_t k = request_.k();
    ASYNC_REPLY(GetTopKNeighbor, node_ids, edge_types, reply_, k);

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

void SampleNeighborCallData::Proceed() {
  if (status_ == CREATE) {
    status_ = PROCESS;
    async_service_->RequestSampleNeighbor(
        &ctx_, &request_, &responder_, cq_, cq_, this);
  } else if (status_ == PROCESS) {
    CallData* call_data =
        new SampleNeighborCallData(async_service_, cq_, graph_engine_);
    call_data->Proceed();

    // The actual processing
    std::vector<euler::common::NodeID> node_ids(request_.node_ids_size());
    std::copy(request_.node_ids().begin(), request_.node_ids().end(),
              node_ids.begin());
    std::vector<int32_t> edge_types(request_.edge_types_size());
    std::copy(request_.edge_types().begin(), request_.edge_types().end(),
              edge_types.begin());
    int32_t count = request_.count();
    ASYNC_REPLY(SampleNeighbor, node_ids, edge_types, reply_, count);

    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete this;
  }
}

}  // namespace service
}  // namespace euler
