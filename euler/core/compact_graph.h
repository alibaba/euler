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

#ifndef EULER_CORE_COMPACT_GRAPH_H_
#define EULER_CORE_COMPACT_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>

#include "euler/common/data_types.h"
#include "euler/common/fast_weighted_collection.h"
#include "euler/core/node.h"
#include "euler/core/edge.h"
#include "euler/core/graph.h"

namespace euler {
namespace core {

class CompactGraph : public Graph {
 public:
  CompactGraph();

  ~CompactGraph();

  std::vector<euler::common::NodeID>
  SampleNode(int node_type, int count) const override;

  std::vector<euler::common::EdgeID>
  SampleEdge(int edge_type, int count) const override;

  bool BuildGlobalSampler() override;

  bool BuildGlobalEdgeSampler() override;

 private:
  euler::common::FastWeightedCollection<int32_t> node_type_collection_;

  euler::common::FastWeightedCollection<int32_t> edge_type_collection_;

  std::vector<euler::common::FastWeightedCollection<euler::common::NodeID>>
      node_samplers_;

  std::vector<euler::common::FastWeightedCollection<euler::common::EdgeID>>
      edge_samplers_;
};

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_COMPACT_GRAPH_H_
