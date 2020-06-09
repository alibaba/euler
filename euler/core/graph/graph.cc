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

#include <unordered_set>

#include "euler/core/graph/graph.h"

#include "euler/core/graph/graph_builder.h"
#include "euler/common/server_register.h"

namespace euler {

Graph::~Graph() {
  auto n_map_it = node_map_.begin();
  while (n_map_it != node_map_.end()) {
    delete n_map_it->second;
    ++n_map_it;
  }

  auto e_map_it = edge_map_.begin();
  while (e_map_it != edge_map_.end()) {
    delete e_map_it->second;
    ++e_map_it;
  }
}

Status StringToSamplerType(const std::string &sampler_type_string,
                           GlobalSamplerType *smt) {
  if (sampler_type_string == "none") {
    *smt = kNone;
  } else if (sampler_type_string == "node") {
    *smt = kNode;
  } else if (sampler_type_string == "edge") {
    *smt = kEdge;
  } else if (sampler_type_string == "all") {
    *smt = kAll;
  } else {
    EULER_LOG(ERROR) << "Invalid sampler type: " << sampler_type_string;
    return Status::Internal("Invalid sampler type:", sampler_type_string);
  }
  return Status::OK();
}

Status StringToDataType(const std::string &data_type_string,
                        GraphDataType *data_type) {
  if (data_type_string == "none") {
    *data_type = kLoadNone;
  } else if (data_type_string == "node") {
    *data_type = kLoadNode;
  } else if (data_type_string == "edge") {
    *data_type = kLoadEdge;
  } else if (data_type_string == "all") {
    *data_type = kLoadAll;
  } else {
    return Status::Internal("Invalid data type: ", data_type_string);
  }
  return Status::OK();
}

Status Graph::Init(int shard_index, int shard_number,
                   const std::string& sampler_type,
                   const std::string& data_path,
                   const std::string& load_data_type) {
  if (!initialized_) {
    if (shard_number <= 0 || shard_index_ >= shard_number) {
      EULER_LOG(ERROR) << "Invalid shard index:" << shard_index
                       << ", shard number: " << shard_number;
      return Status::InvalidArgument("shard_index:", shard_index,
                                     ", shard_number:", shard_number);
    }

    shard_index_ = shard_index;
    shard_number_ = shard_number;

    GlobalSamplerType smt = kNode;
    RETURN_IF_ERROR(StringToSamplerType(sampler_type, &smt));

    auto filter = [shard_index, shard_number] (const std::string& filename) {
      auto vec = Split(filename, "_.");
      if (vec.size() == 3 &&
          atoi(vec[1].c_str()) % shard_number == shard_index &&
          vec[2] == "dat") {
        return true;
      }
      return false;
    };

    GraphDataType data_type = kLoadNode;
    RETURN_IF_ERROR(StringToDataType(load_data_type, &data_type));

    RETURN_IF_ERROR(
        GraphBuilder().Build(this, data_path, smt, filter, data_type));

    if (meta_.partitions_num_ <= 0) {
      EULER_LOG(ERROR) << "Graph partitions_num must > 0";
      return Status::Internal("Graph partitions_num must > 0");
    }

    EULER_LOG(INFO) << "Build graph successfully, shard Index: " << shard_index_
                    << ", shard number: " << shard_number_
                    << ", data path: " << data_path
                    << ", sampler_type: " << sampler_type;

    initialized_ = true;
  }

  return Status::OK();
}

std::vector<Meta> Graph::GetRegisterInfo() {
  Meta meta, shard_meta;
  meta["num_shards"] = std::to_string(shard_number_);
  meta["num_partitions"] = std::to_string(meta_.partitions_num_);
  shard_meta["node_sum_weight"] = Join(GetNodeWeightSums(), ",");
  shard_meta["edge_sum_weight"] = Join(GetEdgeWeightSums(), ",");
  shard_meta["graph_label"] = Join(GetGraphLabel(), ",");

  std::string graph_meta;
  meta_.Serialize(&graph_meta);
  shard_meta["graph_meta"] = graph_meta;

  std::vector<Meta> meta_list = {meta, shard_meta};
  return meta_list;
}

Status Graph::DeregisterRemote(const std::string& host_port,
                               const std::string& zk_server,
                               const std::string& zk_path) {
  auto server_register = GetServerRegister(zk_server, zk_path);
  if (server_register != nullptr) {
    server_register->DeregisterShard(shard_index_, host_port);
  }

  EULER_LOG(INFO) << "Deregister shard " << shard_index_ << " successfully!";
  return Status::OK();
}

Graph::UID Graph::EdgeIdToUID(const euler::common::EdgeID& eid) {
  return eid_hash_(eid);
}

euler::common::EdgeID Graph::UIDToEdgeId(UID uid) {
  euler::common::EdgeID eid;
  if (edge_id_map_.find(uid) != edge_id_map_.end()) {
    eid = edge_id_map_.find(uid)->second;
  }
  return eid;
}

bool Graph::AddNode(Node* n) {
  euler::common::NodeID node_id = n->GetID();
  node_map_[node_id] = n;
  return true;
}

bool Graph::AddNodeFrom(
    const std::unordered_map<euler::common::NodeID, Node*>& map) {
  node_map_.insert(map.begin(), map.end());
  return true;
}

bool Graph::AddNodeFrom(const std::vector<Node*>& vec) {
  for (auto &it : vec) {
     node_map_.insert({it->GetID(), it});
  }
  return true;
}

bool Graph::AddEdge(Edge* e) {
  euler::common::EdgeID edge_id = e->GetID();
  edge_id_map_.insert({eid_hash_(edge_id), edge_id});
  edge_map_[edge_id] = e;
  return true;
}

bool Graph::AddEdgeFrom(const std::unordered_map<euler::common::EdgeID, Edge*,
        euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey>& map) {
  edge_map_.insert(map.begin(), map.end());
  for (auto it = map.begin(); it != map.end(); it++) {
    edge_id_map_.insert({eid_hash_(it->first), it->first});
  }
  return true;
}

bool Graph::AddEdgeFrom(const std::vector<Edge*>& vec) {
  for (auto &it : vec) {
    edge_map_.insert({it->GetID(), it});
    edge_id_map_.insert({eid_hash_(it->GetID()), it->GetID()});
  }
  return true;
}

int64_t Graph::getNodeSize() {
    return node_map_.size();
}

int64_t Graph::getEdgeSize() {
    return edge_map_.size();
}

void Graph::reserveNodeMap(size_t size) {
  node_map_.reserve(size);
}

void Graph::reserveEdgeMap(size_t size) {
  edge_map_.reserve(size);
}

std::vector<euler::common::NodeID> Graph::SampleNode(
    int node_type, int count) const {
  if (!global_sampler_ok_) {
    EULER_LOG(ERROR) << "global sampler is not initialized!";
  }
  std::vector<euler::common::NodeID> vec;
  vec.reserve(count);
  if (node_type == -1) {
    if (node_type_collection_.GetSumWeight() == 0) {
      return vec;
    }
    for (int32_t i = 0; i < count; ++i) {
      node_type = node_type_collection_.Sample().first;
      vec.push_back(node_samplers_[node_type].Sample().first);
    }
  } else {
    if (node_samplers_[node_type].GetSumWeight() == 0) {
      return vec;
    }
    for (int32_t i = 0; i < count; ++i) {
      vec.push_back(node_samplers_[node_type].Sample().first);
    }
  }
  return vec;
}

std::vector<euler::common::NodeID> Graph::SampleNode(
    const std::vector<int>& node_types, int count) const {
  if (!global_sampler_ok_) {
    EULER_LOG(ERROR) << "global sampler is not initialized!";
  }
  std::vector<euler::common::NodeID> vec;
  vec.reserve(count);

  // build type sampler
  std::unordered_set<int> node_types_set(node_types.begin(),
                                         node_types.end());
  std::vector<std::pair<int, float>> sub_type_weight;
  sub_type_weight.reserve(node_types.size());
  for (size_t i = 0; i < node_type_collection_.GetSize(); ++i) {
    if (node_types_set.find(node_type_collection_.Get(i).first) !=
        node_types_set.end()) {
      sub_type_weight.push_back(node_type_collection_.Get(i));
    }
  }
  euler::common::CompactWeightedCollection<int> sub_type_weight_collection;
  sub_type_weight_collection.Init(sub_type_weight);
  if (sub_type_weight_collection.GetSumWeight() > 0) {
    for (int32_t i = 0; i < count; ++i) {
      int node_type = sub_type_weight_collection.Sample().first;
      vec.push_back(node_samplers_[node_type].Sample().first);
    }
  }
  return vec;
}

std::vector<euler::common::EdgeID>
Graph::SampleEdge(int edge_type, int count) const {
  if (!global_edge_sampler_ok_) {
    EULER_LOG(ERROR) << "global edge sampler is not initialized!";
  }
  std::vector<euler::common::EdgeID> vec;
  vec.reserve(count);
  if (edge_type == -1) {
    if (edge_type_collection_.GetSumWeight() == 0) {
      return vec;
    }
    for (int32_t i = 0; i < count; ++i) {
      edge_type = edge_type_collection_.Sample().first;
      vec.push_back(edge_samplers_[edge_type].Sample().first);
    }
  } else {
    if (edge_samplers_[edge_type].GetSumWeight() == 0) {
      return vec;
    }
    for (int32_t i = 0; i < count; ++i) {
      vec.push_back(edge_samplers_[edge_type].Sample().first);
    }
  }
  return vec;
}

std::vector<euler::common::EdgeID> Graph::SampleEdge(
    const std::vector<int>& edge_types, int count) const {
  if (!global_edge_sampler_ok_) {
    EULER_LOG(ERROR) << "global edge sampler is not initialized!";
  }
  std::vector<euler::common::EdgeID> vec;
  vec.reserve(count);

  // build type sampler
  std::unordered_set<int> edge_types_set(edge_types.begin(),
                                         edge_types.end());
  std::vector<std::pair<int, float>> sub_type_weight;
  sub_type_weight.reserve(edge_types.size());
  for (size_t i = 0; i < edge_type_collection_.GetSize(); ++i) {
    if (edge_types_set.find(edge_type_collection_.Get(i).first) !=
        edge_types_set.end()) {
      sub_type_weight.push_back(edge_type_collection_.Get(i));
    }
  }
  euler::common::CompactWeightedCollection<int> sub_type_weight_collection;
  sub_type_weight_collection.Init(sub_type_weight);
  if (sub_type_weight_collection.GetSumWeight() > 0) {
    for (int32_t i = 0; i < count; ++i) {
      int edge_type = sub_type_weight_collection.Sample().first;
      vec.push_back(edge_samplers_[edge_type].Sample().first);
    }
  }
  return vec;
}

bool Graph::BuildGlobalSampler() {
  EULER_LOG(INFO) << "Build Node Global Sampler...";

  std::vector<std::vector<float>> norm_weights;
  std::vector<std::vector<euler::common::NodeID>> node_ids;
  std::vector<int32_t> node_type_ids;

  size_t node_type_num = meta_.node_type_map_.size();
  EULER_LOG(INFO) << "Node type num:" << node_type_num;

  norm_weights.resize(node_type_num);
  node_ids.resize(node_type_num);
  node_type_ids.resize(node_type_num);
  node_weight_sums_.resize(node_type_num);
  node_type_ids.resize(node_type_num);

  for (auto &it : node_map_) {
    int32_t type = it.second->GetType();
    node_ids[type].push_back(it.first);
    norm_weights[type].push_back(it.second->GetWeight());
    node_weight_sums_[type] += it.second->GetWeight();
  }

  for (size_t type = 0; type < node_weight_sums_.size(); type++) {
    for (size_t idx = 0; idx < norm_weights[type].size(); idx++) {
      norm_weights[type][idx] /= node_weight_sums_[type];
    }
    node_type_ids[type] = type;
  }

  node_samplers_.resize(node_ids.size());
  for (size_t type = 0; type < node_ids.size(); type++) {
    node_samplers_[type].Init(node_ids[type], norm_weights[type]);
  }
  node_type_collection_.Init(node_type_ids, node_weight_sums_);
  global_sampler_ok_ = true;
  return true;
}

bool Graph::BuildGlobalEdgeSampler() {
  EULER_LOG(INFO) << "Build Edge Global Sampler...";

  std::vector<std::vector<float>> norm_weights;
  std::vector<std::vector<euler::common::EdgeID>> edge_ids;

  size_t edge_type_num = meta_.edge_type_map_.size();

  norm_weights.resize(edge_type_num);
  edge_ids.resize(edge_type_num);
  edge_weight_sums_.resize(edge_type_num);

  for (auto &it : edge_map_) {
    int type = it.second->GetType();
    edge_ids[type].push_back(it.first);
    norm_weights[type].push_back(it.second->GetWeight());
    edge_weight_sums_[type] += it.second->GetWeight();
  }

  for (size_t type = 0; type < edge_weight_sums_.size(); type++) {
    for (size_t idx = 0; idx < norm_weights[type].size(); idx++) {
      norm_weights[type][idx] /= edge_weight_sums_[type];
    }
  }

  edge_samplers_.resize(edge_ids.size());
  for (size_t type = 0; type < edge_ids.size(); type++) {
    edge_samplers_[type].Init(edge_ids[type], norm_weights[type]);
  }
  global_edge_sampler_ok_ = true;
  return true;
}

size_t Graph::GetNodeTypeNum() {
  return meta_.node_type_map_.size();
}

size_t Graph::GetEdgeTypeNum() {
  return meta_.edge_type_map_.size();
}

bool Graph::GlobalNodeSamplerOk() {
  return global_sampler_ok_;
}

bool Graph::GlobalEdgeSamplerOk() {
  return global_edge_sampler_ok_;
}

std::vector<float> Graph::GetNodeWeightSums() {
  if (global_sampler_ok_) {
    return node_weight_sums_;
  } else {
    EULER_LOG(WARNING) << "global sampler is not ok";
    return std::vector<float>();
  }
}

std::vector<float> Graph::GetEdgeWeightSums() {
  if (global_edge_sampler_ok_) {
    return edge_weight_sums_;
  } else {
    EULER_LOG(WARNING) << "global sampler is not ok";
    return std::vector<float>();
  }
}

std::vector<std::string> Graph::GetGraphLabel() {
  std::unordered_set<std::string> label_set;
  int32_t label_id = GetNodeFeatureId("binary_graph_label");
  if (label_id != -1) {
    for (auto it = node_map_.begin();
         it != node_map_.end(); ++it) {
      std::vector<std::string> graph_label;
      it->second->GetBinaryFeature({label_id}, &graph_label);
      label_set.insert(graph_label[0]);
    }
  }
  std::vector<std::string> label_vec;
  label_vec.reserve(label_set.size());
  for (auto it = label_set.begin();
       it != label_set.end(); ++it) {
    label_vec.push_back(*it);
  }
  return label_vec;
}

void Graph::ShowGraph() {
  int32_t cnt = 0;
  auto it = node_map_.begin();
  while (it != node_map_.end()) {
    if (cnt % 500000 == 0) {
      std::string s;
      if (it->second->Serialize(&s)) {
        std::cout << s << std::endl;
      }
    }
    ++cnt;
    ++it;
  }
}

bool Graph::Dump(euler::FileIO* file_io) const {
  uint32_t size = node_map_.size();
  if (!file_io->Append(size)) {
    return false;
  }
  for (auto& i : node_map_) {
    std::string s;
    if (!i.second->Serialize(&s)) {
      return false;
    }
    file_io->Append(s);
  }

  uint32_t edge_size = edge_map_.size();
  if (!file_io->Append(edge_size)) {
    return false;
  }
  for (auto& i : edge_map_) {
    std::string s;
    if (!i.second->Serialize(&s)) {
      return false;
    }
    file_io->Append(s);
  }
  return true;
}


FeatureType Graph::GetNodeFeatureType(const std::string &feature_name) const {
  return meta_.GetFeatureType(feature_name);
}

int32_t Graph::GetNodeFeatureId(const std::string &feature_name) const {
  return meta_.GetFeatureId(feature_name);
}


FeatureType Graph::GetEdgeFeatureType(const std::string &feature_name) const {
  return meta_.GetEdgeFeatureType(feature_name);
}

int32_t Graph::GetEdgeFeatureId(const std::string &feature_name) const {
  return meta_.GetEdgeFeatureId(feature_name);
}

}  // namespace euler
