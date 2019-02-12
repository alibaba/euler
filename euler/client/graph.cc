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

#include "euler/client/graph.h"

#include <math.h>

#include <string>
#include <atomic>
#include <algorithm>

#include "glog/logging.h"

#include "euler/client/local_graph.h"
#include "euler/client/remote_graph.h"
#include "euler/common/compact_weighted_collection.h"

namespace euler {
namespace client {

static const char* kLocalGraphMode = "Local";
static const char* kRemoteGraphMode = "Remote";


////////////////////////////  CallBack ////////////////////////////

namespace {

class Callback {
 public:
  enum Hierarchy {
    PARENT,
    CHILD
  };

  Callback(Hierarchy hierachy,
           std::function<void(const IDWeightPairVec&)> callback,
           const std::vector<NodeID>& parents,
           IDWeightPairVec* parent_neighbors,
           IDWeightPairVec* child_neighbors,
           std::atomic<int>* counter, float p, float q, int count)
      : hierarchy_(hierachy), callback_(callback), parents_(parents),
        parent_neighbors_(parent_neighbors), child_neighbors_(child_neighbors),
        counter_(counter), p_(p), q_(q), count_(count) {
  }

  void operator()(const IDWeightPairVec& neighbors);

 private:
  void BuildWeights(
      const std::vector<euler::common::IDWeightPair>& parent_neighbors,
      const std::vector<euler::common::IDWeightPair>& child_neighbors,
      euler::common::NodeID parent_id, std::vector<float>* weights);

 private:
  Hierarchy hierarchy_;
  std::function<void(const IDWeightPairVec&)> callback_;
  std::vector<NodeID> parents_;
  IDWeightPairVec* parent_neighbors_;
  IDWeightPairVec* child_neighbors_;
  std::atomic<int>* counter_;
  float p_;
  float q_;
  int count_;
};

void Callback::operator()(const IDWeightPairVec& neighbors) {
  IDWeightPairVec* dst = child_neighbors_;
  if (hierarchy_ == PARENT) {
    dst = parent_neighbors_;
  }

  std::swap(*dst, const_cast<IDWeightPairVec&>(neighbors));
  auto counter = ++*counter_;
  if (counter == 2) {
    IDWeightPairVec result(child_neighbors_->size());
    for (auto it = result.begin(); it != result.end(); ++it) {
      it->reserve(count_);
    }

    for (size_t i = 0; i < child_neighbors_->size(); ++i) {
      auto& pn = parent_neighbors_->at(i);
      auto& cn = child_neighbors_->at(i);
      if (!cn.empty()) {
        auto parent_id = parents_.at(i);
        std::vector<float> weights(cn.size());
        BuildWeights(pn, cn, parent_id, &weights);

        // Sample neighbors
        using CWC = euler::common::CompactWeightedCollection<IDWeightPair>;
        CWC sampler;
        sampler.Init(cn, weights);
        auto& neighbors = result[i];
        auto j = 0;
        while (j < count_) {
          neighbors.push_back(sampler.Sample().first);
          ++j;
        }
      }
    }
    delete parent_neighbors_;
    delete child_neighbors_;
    delete counter_;
    callback_(result);
  }
}

void Callback::BuildWeights(
    const std::vector<euler::common::IDWeightPair>& parent_neighbors,
    const std::vector<euler::common::IDWeightPair>& child_neighbors,
    euler::common::NodeID parent_id, std::vector<float>* weights) {
  size_t j = 0;
  size_t k = 0;
  while (j < child_neighbors.size() && k < parent_neighbors.size()) {
    if (std::get<0>(child_neighbors[j]) < std::get<0>(parent_neighbors[k])) {
      if (std::get<0>(child_neighbors[j]) != parent_id) {
        weights->at(j) = std::get<1>(child_neighbors[j]) / q_;  // q range
      } else {
        weights->at(j) = std::get<1>(child_neighbors[j]);  // backward to parent
      }
      ++j;
    } else if (std::get<0>(child_neighbors[j]) ==
               std::get<0>(parent_neighbors[k])) {
      weights->at(j) = std::get<1>(child_neighbors[j]) / p_;  // p range
      ++k;
      ++j;
    } else {
      ++k;
    }
  }
  while (j < child_neighbors.size()) {
    if (std::get<0>(child_neighbors[j]) != parent_id) {
      weights->at(j) = std::get<1>(child_neighbors[j]) / q_;
    } else {
      weights->at(j) = std::get<1>(child_neighbors[j]);
    }
    ++j;
  }
}

}  // namespace

////////////////////////////  Graph  ////////////////////////////

std::unique_ptr<Graph> Graph::NewGraph(const std::string& config_file) {
  GraphConfig config;
  config.Load(config_file);
  return NewGraph(config);
}

std::unique_ptr<Graph> Graph::NewGraph(const GraphConfig& config) {
  Graph* graph = nullptr;

  std::string mode;
  config.Get("mode", &mode);
  if (kLocalGraphMode == mode) {
    graph = new LocalGraph();
  } else if (kRemoteGraphMode == mode) {
    graph = new RemoteGraph();
  } else {
    LOG(ERROR) << "Invlaid mode got: " << mode;
    return nullptr;
  }

  std::string init = "instant";
  config.Get("init", &init);
  if (init != "lazy" && !graph->Initialize(config)) {
    LOG(ERROR) << "Initialize graph failed, config: " << config.DebugString();
    graph = nullptr;
  }

  return std::unique_ptr<Graph>(graph);
}

void Graph::BiasedSampleNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<NodeID>& parent_node_ids,
    const std::vector<int>& edge_types,
    const std::vector<int>& parent_edge_types,
    int count,
    float p,
    float q,
    std::function<void(const IDWeightPairVec&)> callback) const {
  // if p = q = 1, biased sample degenerates to normal random walk
  const float kEps = 1.0e-6;
  if (fabs(p - 1.0) <= kEps && fabs(q - 1.0) <= kEps) {
    SampleNeighbor(node_ids, edge_types, count, callback);
  } else {
    auto parent_neighbors = new IDWeightPairVec();
    auto child_neighbors = new IDWeightPairVec();
    auto counter = new std::atomic<int>(0);
    Callback pcb(Callback::PARENT, callback, parent_node_ids,
                 parent_neighbors, child_neighbors, counter, p, q, count);
    Callback ccb(Callback::CHILD, callback, parent_node_ids,
                 parent_neighbors, child_neighbors, counter, p, q, count);
    GetSortedFullNeighbor(node_ids, edge_types, ccb);
    GetSortedFullNeighbor(parent_node_ids, parent_edge_types, pcb);
  }
}

}  // namespace client
}  // namespace euler
