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

#ifndef EULER_CORE_GRAPH_BUILDER_H_
#define EULER_CORE_GRAPH_BUILDER_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "euler/common/data_types.h"
#include "euler/common/file_io_factory.h"
#include "euler/core/node.h"
#include "euler/core/edge.h"
#include "euler/core/graph.h"
#include "euler/core/graph_factory.h"

namespace euler {
namespace core {

enum LoaderType {local, hdfs};
enum GlobalSamplerType {edge, node, all, none};

typedef std::unordered_map<euler::common::NodeID, Node*> NODEMAP;
typedef std::vector<Node*> NODEVEC;

typedef std::unordered_map<euler::common::EdgeID, Edge*,
        euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey> EDGEMAP;
typedef std::vector<Edge*> EDGEVEC;
class GraphBuilder {
 public:
  explicit GraphBuilder(GraphFactory* factory) : factory_(factory) {}

  virtual ~GraphBuilder() {}

  Graph* BuildGraph(
      const std::vector<std::string>& file_names,
      LoaderType loader_type, std::string addr, int32_t port,
      GlobalSamplerType global_sampler_type);

 private:
  bool ParseBlock(euler::common::FileIO* file_io, Graph* graph, int32_t checksum,
                  NODEVEC &np, EDGEVEC &ep, int32_t& n_type_num, int32_t& e_type_num);

  bool LoadData(
      LoaderType loader_type,
      const std::vector<std::string>& file_list,
      Graph* graph, std::string addr, int32_t port,NODEVEC &np, EDGEVEC &ep,
      int32_t& n_type_num, int32_t& e_type_num);

 private:
  std::unique_ptr<GraphFactory> factory_;
};

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_GRAPH_BUILDER_H_
