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

#include "euler/tools/remote_console/remote_console.h"

#include <iostream>
#include <unordered_map>
#include <algorithm>

#include "euler/core/api/api.h"
#include "euler/common/logging.h"

namespace euler {

void RemoteConsole::GetNodeNeighbor(uint64_t node_id, std::string nb_type,
    std::vector<uint64_t>* nbs, std::vector<float>* edge_weights,
    std::vector<int32_t>* types) {
  std::string gremlin = "v(nodes).outV(e_types).as(nb)";
  Query query(gremlin);

  Tensor* nodes_t = query.AllocInput("nodes", {1}, kUInt64);
  Tensor* e_types_t = query.AllocInput("e_types", {1}, kInt32);
  nodes_t->Raw<uint64_t>()[0] = node_id;
  auto edge_type_map = query_proxy_->graph_meta().edge_type_map();
  e_types_t->Raw<int32_t>()[0] = edge_type_map.at(nb_type);
  std::vector<std::string> result_names = {"nb:0", "nb:1", "nb:2", "nb:3"};
  std::unordered_map<std::string, Tensor*> results =
      query_proxy_->RunGremlin(&query, result_names);
  nbs->resize(results["nb:1"]->NumElements());
  edge_weights->resize(results["nb:2"]->NumElements());
  types->resize(results["nb:3"]->NumElements());
  std::copy(results["nb:1"]->Raw<uint64_t>(),
            results["nb:1"]->Raw<uint64_t>() + results["nb:1"]->NumElements(),
            nbs->begin());
  std::copy(results["nb:2"]->Raw<float>(),
            results["nb:2"]->Raw<float>() + results["nb:2"]->NumElements(),
            edge_weights->begin());
  std::copy(results["nb:3"]->Raw<int32_t>(),
            results["nb:3"]->Raw<int32_t>() + results["nb:3"]->NumElements(),
            types->begin());
}

#define FEATURE(DATA_TYPE) {                                   \
  std::string gremlin = "v(nodes).values(fid).as(nf)";         \
  Query query(gremlin);                                        \
  Tensor* nodes_t = query.AllocInput("nodes", {1}, kUInt64);   \
  Tensor* fid_t = query.AllocInput("fid", {1}, kString);       \
  nodes_t->Raw<uint64_t>()[0] = node_id;                       \
  *(fid_t->Raw<std::string*>()[0]) = feature_name;             \
  std::vector<std::string> result_names = {"nf:0", "nf:1"};    \
  std::unordered_map<std::string, Tensor*> results =           \
      query_proxy_->RunGremlin(&query, result_names);          \
  feature->resize(results["nf:1"]->NumElements());             \
  std::copy(results["nf:1"]->Raw<DATA_TYPE>(),                 \
            results["nf:1"]->Raw<DATA_TYPE>() +                \
            results["nf:1"]->NumElements(), feature->begin()); \
}

void RemoteConsole::GetNodeFeature(uint64_t node_id,
                                   std::string feature_name,
                                   std::vector<uint64_t>* feature) {
  feature_name = "sparse_" + feature_name;
  FEATURE(uint64_t);
}

void RemoteConsole::GetNodeFeature(uint64_t node_id,
                                   std::string feature_name,
                                   std::vector<float>* feature) {
  feature_name = "dense_" + feature_name;
  FEATURE(float);
}

}  // namespace euler

int main() {
  std::string zk_addr, zk_path;
  int32_t shard_num;

  std::cout << "please input zk_addr and zk_path" << std::endl;
  std::cin >> zk_addr >> zk_path;
  std::cout << "please input shard num" << std::endl;
  std::cin >> shard_num;
  std::cout << "zk_addr: " << zk_addr
      << " zk_path: " << zk_path
      << " shard_num: " << shard_num << std::endl;

  euler::RemoteConsole console(zk_addr, zk_path, shard_num);
  while (true) {
    std::string query_type;
    std::cout << "please input: query_type "
        << "query_nb/query_sp_fea/query_dense_fea" << std::endl;
    std::cin >> query_type;
    if (query_type == "query_nb") {
      std::cout << "please input: node_id edge_type" << std::endl;
      uint64_t node_id;
      std::string e_type;
      std::cin >> node_id >> e_type;
      std::vector<uint64_t> nbs;
      std::vector<float> weights;
      std::vector<int32_t> types;
      console.GetNodeNeighbor(node_id, e_type, &nbs, &weights, &types);
      std::cout << "nb: ";
      for (uint64_t nb : nbs) {
        std::cout << nb << " ";
      }
      std::cout << std::endl;
      std::cout << "edge weight: ";
      for (float w : weights) {
        std::cout << w << " ";
      }
      std::cout << std::endl;
      std::cout << "edge type: ";
      for (int32_t t : types) {
        std::cout << t << " ";
      }
      std::cout << std::endl;
    } else if (query_type == "query_sp_fea") {
      std::cout << "please input: node_id fid" << std::endl;
      uint64_t node_id;
      std::string fid;
      std::cin >> node_id >> fid;
      std::vector<uint64_t> sp_feature;
      console.GetNodeFeature(node_id, fid, &sp_feature);
      std::cout << "sp feature: ";
      for (uint64_t sp : sp_feature) {
        std::cout << sp << " ";
      }
      std::cout << std::endl;
    } else if (query_type == "query_dense_fea") {
      std::cout << "please input: node_id fid" << std::endl;
      uint64_t node_id;
      std::string fid;
      std::cin >> node_id >> fid;
      std::vector<float> dense_feature;
      console.GetNodeFeature(node_id, fid, &dense_feature);
      std::cout << "dense feature: ";
      for (float d : dense_feature) {
        std::cout << d << " ";
      }
      std::cout << std::endl;
    } else {
      std::cout << "error query type" << std::endl;
    }
  }
}
