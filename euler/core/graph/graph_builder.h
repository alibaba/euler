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

#ifndef EULER_CORE_GRAPH_GRAPH_BUILDER_H_
#define EULER_CORE_GRAPH_GRAPH_BUILDER_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <unordered_map>

#include "euler/common/file_io.h"
#include "euler/common/data_types.h"
#include "euler/common/str_util.h"
#include "euler/core/graph/node.h"
#include "euler/core/graph/edge.h"
#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_meta.h"

namespace euler {

enum GlobalSamplerType {
  kEdge,
  kNode,
  kAll,
  kNone
};

enum GraphDataType {
  kLoadNone = 0,
  kLoadNode = 1,
  kLoadEdge = 2,
  kLoadAll = 3
};

typedef std::unordered_map<euler::common::NodeID, Node*> NodeMap;
typedef std::vector<Node*> NodeVec;

typedef std::unordered_map<
  euler::common::EdgeID,
  Edge*,
  euler::common::EdgeIDHashFunc,
  euler::common::EdgeIDEqualKey> EdgeMap;

typedef std::vector<Edge*> EdgeVec;

struct TaskInfo;

class GraphBuilder {
 public:
  Status Build(Graph* graph, Slice path, GlobalSamplerType sampler_type,
               FileIO::FilterFunc filter, GraphDataType data_type);

 private:
  friend class Graph;

  Status BuildSampler(Graph* graph, GlobalSamplerType sampler_type);

  bool ParseNodes(FileIO* reader, NodeVec* node_vec);

  bool ParseEdges(FileIO* reader, EdgeVec* edge_vec);

  bool LoadData(TaskInfo* task_info);

  Status LoadMeta(Slice meta_path, GraphMeta* meta);

  Status AddToGraph(Graph* graph, const TaskInfo* start, const TaskInfo* end);

  GraphBuilder() { }
};

}  // namespace euler

#endif  // EULER_CORE_GRAPH_GRAPH_BUILDER_H_
