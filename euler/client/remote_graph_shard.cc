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

#include "euler/client/remote_graph_shard.h"

#include <algorithm>

#include "glog/logging.h"

#include "euler/proto/graph_service.pb.h"
#include "euler/client/rpc_client.h"

namespace euler {
namespace client {

RemoteGraphShard::RemoteGraphShard() :index_(-1) {
}

RemoteGraphShard::RemoteGraphShard(
    std::shared_ptr<ServerMonitor> monitor, int index)
    : monitor_(monitor), index_(index) {
}

bool RemoteGraphShard::Initialize(const GraphConfig& config) {
  client_ = NewRpcClient(monitor_, index_, config);
  if (client_ == nullptr) {
    LOG(ERROR) << "Initialize rpc client failed, shards: " << index_
               << ", config: " << config.DebugString();
    return false;
  }
  return true;
}

#define RPC_METHOD_NAME_PREFIX  "/euler.proto.GraphService/"


///////////////////////////// Sampe Node/Edge /////////////////////////////

#define SHARD_SAMPLE(REQUEST, REPLY, BUILDER, ID_METHOD,        \
                     RESULT_TYPE, ID_BUILDER, RPC_METHOD_NAME)  \
  euler::proto::REQUEST request;                                \
  BUILDER(&request);                                            \
  auto reply = new euler::proto::REPLY();                       \
  auto rpc_callback = [reply, callback, ID_BUILDER] (           \
      const Status& status) {                                   \
    RESULT_TYPE id_vec;                                         \
    if (status.ok()) {                                          \
      auto ids = reply->ID_METHOD();                            \
      id_vec.reserve(ids.size());                               \
      for (auto it = ids.begin(); it != ids.end(); ++it) {      \
        auto id = ID_BUILDER(*it);                              \
        id_vec.push_back(id);                                   \
      }                                                         \
    }                                                           \
    callback(id_vec);                                           \
    delete reply;                                               \
  };                                                            \
  client_->IssueRpcCall(RPC_METHOD_NAME_PREFIX RPC_METHOD_NAME, \
                        request, reply, rpc_callback);          \

void RemoteGraphShard::SampleNode(
    int node_type, int count,
    std::function<void(const NodeIDVec&)> callback) const {
  auto builder = [node_type, count] (euler::proto::SampleNodeRequest* request) {
    request->set_count(count);
    request->set_node_type(node_type);
  };
  auto id_builder = [] (google::protobuf::uint64 id) {
    return static_cast<uint64_t>(id);
  };
  SHARD_SAMPLE(SampleNodeRequest, SampleNodeReply,
               builder, node_ids, NodeIDVec, id_builder, "SampleNode");
}

void RemoteGraphShard::SampleEdge(
    int edge_type, int count,
    std::function<void(const EdgeIDVec&)> callback) const {
  auto builder = [edge_type, count] (euler::proto::SampleEdgeRequest* request) {
    request->set_count(count);
    request->set_edge_type(edge_type);
  };
  auto id_builder = [] (const euler::proto::EdgeID& id) {
    EdgeID eid;
    eid = std::make_tuple(id.src_node(), id.dst_node(), id.type());
    return eid;
  };
  SHARD_SAMPLE(SampleEdgeRequest, SampleEdgeReply,
               builder, edge_ids, EdgeIDVec, id_builder, "SampleEdge");
}

#undef SHARD_SAMPLE  // SHARD_SAMPLE


/////////////////////////// Get Feature ////////////////////////////


namespace {

template<typename Request>
void NodeIdBuilder(const std::vector<NodeID>& node_ids, Request* req) {
  req->mutable_node_ids()->Resize(node_ids.size(), 0);
  memcpy(req->mutable_node_ids()->mutable_data(), node_ids.data(),
         sizeof(node_ids[0]) * node_ids.size());
}

template<typename Request>
void EdgeIdBuilder(const std::vector<EdgeID>& edge_ids, Request* req) {
  for (auto it = edge_ids.begin(); it != edge_ids.end(); ++it) {
    auto eid = req->mutable_edge_ids()->Add();
    eid->set_src_node(std::get<0>(*it));
    eid->set_dst_node(std::get<1>(*it));
    eid->set_type(std::get<2>(*it));
  }
}

template<typename Request>
void FeatureIdBuilder(const std::vector<int>& fids, Request* req) {
  req->mutable_feature_ids()->Resize(fids.size(), 0);
  memcpy(req->mutable_feature_ids()->mutable_data(), fids.data(),
         sizeof(fids[0]) * fids.size());
}

template<typename Reply, typename ResultType>
void ResultBuilder(const Reply* reply, int ids_size,
                   int fids_size, ResultType* results) {
  auto data = reply->values().data();
  for (int i = 0; i < ids_size; ++i) {
    auto& result = results->at(i);
    result.resize(fids_size);
    for (int j = 0; j < fids_size; ++j) {
      auto& item = result[j];
      item.resize(reply->value_nums(i * fids_size + j));
      memcpy(&item[0], data, sizeof(item[0]) * item.size());
      data += item.size();
    }
  }
}

}  // namespace

#define SHARD_GET_FEATURE(REQUEST, REPLY, IDS, ID_BUILDER, FIDS,         \
                          FID_BUILDER, RESULT_TYPE, RPC_METHOD_NAME)     \
  euler::proto::REQUEST request;                                         \
  auto reply = new euler::proto::REPLY;                                  \
  ID_BUILDER(IDS, &request);                                             \
  FID_BUILDER(FIDS, &request);                                           \
  int iz = IDS.size();                                                   \
  int fz = FIDS.size();                                                  \
  auto rpc_callback = [reply, callback, iz, fz] (const Status& status) { \
    RESULT_TYPE features(iz);                                            \
    if (status.ok()) {                                                   \
      ResultBuilder(reply, iz, fz, &features);                           \
    } else {                                                             \
      LOG(ERROR) << "Get features failed: " << status.message();         \
    }                                                                    \
    callback(features);                                                  \
    delete reply;                                                        \
  };                                                                     \
  client_->IssueRpcCall(RPC_METHOD_NAME_PREFIX RPC_METHOD_NAME,          \
                        request, reply, rpc_callback);                   \

void RemoteGraphShard::GetNodeFloat32Feature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const FloatFeatureVec&)> callback) const {
  SHARD_GET_FEATURE(GetNodeFloat32FeatureRequest, GetFloat32FeatureReply,
                    node_ids, NodeIdBuilder, fids, FeatureIdBuilder,
                    FloatFeatureVec, "GetNodeFloat32Feature");
}

void RemoteGraphShard::GetNodeUint64Feature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const UInt64FeatureVec&)> callback) const {
  SHARD_GET_FEATURE(GetNodeUInt64FeatureRequest, GetUInt64FeatureReply,
                    node_ids, NodeIdBuilder, fids, FeatureIdBuilder,
                    UInt64FeatureVec, "GetNodeUInt64Feature");
}

void RemoteGraphShard::GetNodeBinaryFeature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const BinaryFatureVec&)> callback) const {
  SHARD_GET_FEATURE(GetNodeBinaryFeatureRequest, GetBinaryFeatureReply,
                    node_ids, NodeIdBuilder, fids, FeatureIdBuilder,
                    BinaryFatureVec, "GetNodeBinaryFeature");
}

void RemoteGraphShard::GetEdgeFloat32Feature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int>& fids,
    std::function<void(const FloatFeatureVec&)> callback) const {
  SHARD_GET_FEATURE(GetEdgeFloat32FeatureRequest, GetFloat32FeatureReply,
                    edge_ids, EdgeIdBuilder, fids, FeatureIdBuilder,
                    FloatFeatureVec, "GetEdgeFloat32Feature");
}

void RemoteGraphShard::GetEdgeUint64Feature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int32_t>& fids,
    std::function<void(const UInt64FeatureVec&)> callback) const {
  SHARD_GET_FEATURE(GetEdgeUInt64FeatureRequest, GetUInt64FeatureReply,
                    edge_ids, EdgeIdBuilder, fids, FeatureIdBuilder,
                    UInt64FeatureVec, "GetEdgeUInt64Feature");
}

void RemoteGraphShard::GetEdgeBinaryFeature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int>& fids,
    std::function<void(const BinaryFatureVec&)> callback) const {
  SHARD_GET_FEATURE(GetEdgeBinaryFeatureRequest, GetBinaryFeatureReply,
                    edge_ids, EdgeIdBuilder, fids, FeatureIdBuilder,
                    BinaryFatureVec, "GetEdgeBinaryFeature");
}

#undef SHARD_GET_FEATURE  // SHARD_GET_FEATURE

///////////////////////////// Get Type /////////////////////////////

void RemoteGraphShard::GetNodeType(
    const std::vector<NodeID>& node_ids,
    std::function<void(const TypeVec&)> callback) const {
  euler::proto::GetNodeTypeRequest request;
  auto reply = new euler::proto::GetTypeReply();
  NodeIdBuilder(node_ids, &request);
  auto rpc_callback = [node_ids, reply, callback](const Status& status) {
    TypeVec types(node_ids.size());
    if (status.ok()) {
      std::copy(reply->types().begin(),
                reply->types().end(),
                types.begin());
    } else {
      LOG(ERROR) << "get type error";
    }
    callback(types);
    delete reply;
  };
  client_->IssueRpcCall("/euler.proto.GraphService/GetNodeType",
                        request, reply, rpc_callback);
}

///////////////////////////// Get Neighbor /////////////////////////////

namespace {

void NeighborsBuilder(const euler::proto::GetNeighborReply* reply,
                      int node_number, IDWeightPairVec* neighbors) {
  int index = 0;
  for (int i = 0; i < node_number; ++i) {
    auto& neighbor = neighbors->at(i);
    neighbor.resize(reply->neighbor_nums(i));
    for (uint32_t j = 0; j < reply->neighbor_nums(i); ++j) {
      neighbor[j] = std::make_tuple(reply->node_ids(index),
                                    reply->weights(index),
                                    reply->types(index));
      ++index;
    }
  }
}

}  // namespace

#define SHARD_GET_NEIGHBOR(REQUEST, REPLY,                              \
                           BUILD_REQUEST, RPC_METHOD_NAME)              \
  euler::proto::REQUEST request;                                        \
  auto reply = new euler::proto::REPLY;                                 \
  NodeIdBuilder(node_ids, &request);                                    \
  request.mutable_edge_types()->Resize(edge_types.size(), 0);           \
  memcpy(request.mutable_edge_types()->mutable_data(),                  \
         edge_types.data(), sizeof(edge_types[0]) * edge_types.size()); \
  BUILD_REQUEST(&request);                                              \
  int size = node_ids.size();                                           \
  auto rpc_callback = [size, reply, callback] (const Status& status) {  \
    IDWeightPairVec neighbors(size);                                    \
    if (status.ok()) {                                                  \
      NeighborsBuilder(reply, size, &neighbors);                        \
    } else {                                                            \
      LOG(ERROR) << "Get neighbors failed: " << status.message();       \
    }                                                                   \
    callback(neighbors);                                                \
    delete reply;                                                       \
  };                                                                    \
  client_->IssueRpcCall(RPC_METHOD_NAME_PREFIX RPC_METHOD_NAME,         \
                        request, reply, rpc_callback);                  \

void RemoteGraphShard::GetFullNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    std::function<void(const IDWeightPairVec&)> callback) const {
  auto request_builder = [](euler::proto::GetFullNeighborRequest*) {
  };
  SHARD_GET_NEIGHBOR(GetFullNeighborRequest, GetNeighborReply,
                     request_builder, "GetFullNeighbor");
}

void RemoteGraphShard::GetSortedFullNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    std::function<void(const IDWeightPairVec&)> callback) const {
  auto request_builder = [](euler::proto::GetSortedNeighborRequest*) {
  };
  SHARD_GET_NEIGHBOR(GetSortedNeighborRequest, GetNeighborReply,
                     request_builder, "GetSortedNeighbor");
}

void RemoteGraphShard::GetTopKNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    int k,
    std::function<void(const IDWeightPairVec&)> callback) const {
  auto request_builder = [k] (euler::proto::GetTopKNeighborRequest* request) {
    request->set_k(k);
  };
  SHARD_GET_NEIGHBOR(GetTopKNeighborRequest, GetNeighborReply,
                     request_builder, "GetTopKNeighbor");
}

void RemoteGraphShard::SampleNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    int count,
    std::function<void(const IDWeightPairVec&)> callback) const {
  auto builder = [count] (euler::proto::SampleNeighborRequest* request) {
    request->set_count(count);
  };
  SHARD_GET_NEIGHBOR(SampleNeighborRequest, GetNeighborReply,
                     builder, "SampleNeighbor");
}

#undef SHARD_GET_NEIGHBOR  // SHARD_GET_NEIGHBOR

}  // namespace client
}  // namespace euler
