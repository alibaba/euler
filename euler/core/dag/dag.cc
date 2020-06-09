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

#include "euler/core/dag/dag.h"

#include <utility>
#include <unordered_map>

#include "euler/core/dag/node.h"
#include "euler/core/dag/edge.h"
#include "euler/core/framework/dag.pb.h"
#include "euler/common/str_util.h"
#include "euler/core/framework/op_kernel.h"

namespace euler {

DAG::DAG(const std::string& name): name_(name) { }

DAG::~DAG() {
  for (auto& n : nodes_) { delete n; }
  for (auto& e : edges_) { delete e; }
}

DAGNode* DAG::AddNode(const DAGNodeProto& node_def, Status* s) {
  *s = LookupOpKernel(node_def.op());
  if (!s->ok()) {
    return nullptr;
  }

  auto node = new DAGNode;
  node->id_ = nodes_.size();
  node->def_ = node_def;
  nodes_.emplace_back(node);
  *s = Status::OK();
  return node;
}

DAGEdge* DAG::AddEdge(DAGNode* src, int x, DAGNode* dst, int y) {
  auto edge = new DAGEdge;
  edge->id_ = edges_.size();
  edge->src_ = src;
  edge->src_output_ = x;
  edge->dst_ = dst;
  edge->dst_input_ = y;
  src->out_edges_.insert(edge);
  dst->in_edges_.insert(edge);
  edges_.emplace_back(edge);
  return edge;
}

std::unique_ptr<DAG> DAG::NewFromProto(const DAGProto& dag_def) {
  std::unique_ptr<DAG> dag(new DAG(dag_def.name()));
  std::unordered_map<std::string, DAGNode*> node_map;

  // Add Nodes
  for (auto& node_def : dag_def.nodes()) {
    Status s;
    auto node = dag->AddNode(node_def, &s);
    if (!s.ok()) {
      EULER_LOG(ERROR) << "Add node failed, status:" << s;
      dag.reset();
      return dag;
    }
    node_map.insert({node_def.name(), node});
  }

  // Add Edges
  for (auto& node_def : dag_def.nodes()) {
    int y = 0;
    auto dst = node_map[node_def.name()];
    for (auto& input : node_def.inputs()) {
      auto vec = Split(input, ":");
      if (vec.size() != 2) {
        continue;
      }
      auto it = node_map.find(vec[0]);
      if (it != node_map.end()) {
        auto src = it->second;
        int x = atoi(vec[1].c_str());
        (void) dag->AddEdge(src, x, dst, y);
      } else {  // Node is source node
        // Do nothing
      }
      y++;
    }
  }

  return dag;
}

}  //  namespace euler
