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

#include "euler/core/compact_graph_factory.h"

#include "euler/core/compact_node.h"
#include "euler/core/compact_edge.h"
#include "euler/core/compact_graph.h"

namespace euler {
namespace core {
Node* CompactGraphFactory::CreateNode() {
  return new CompactNode();
}

Edge* CompactGraphFactory::CreateEdge() {
  return new CompactEdge();
}

Graph* CompactGraphFactory::CreateGraph() {
  return new CompactGraph();
}

}  // namespace core
}  // namespace euler
