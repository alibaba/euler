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


#ifndef EULER_CORE_DAG_NODE_H_
#define EULER_CORE_DAG_NODE_H_

#include <functional>
#include <string>
#include <vector>
#include <unordered_set>

#include "euler/common/status.h"
#include "euler/core/framework/dag_node.pb.h"

namespace euler {

class DAGEdge;

typedef std::unordered_set<DAGEdge*> EdgeSet;

class DAGNode {
 public:
  int id() const { return id_; }

  std::string name() const { return def_.name(); }

  std::string op() const { return def_.op(); }

  const DAGNodeProto& def() const { return def_; }

  int num_inputs() const { return in_edges_.size(); }
  int num_outputs() const { return out_edges_.size(); }

  const EdgeSet& input_edges() const { return in_edges_; }
  const EdgeSet& output_edges() const { return out_edges_; }

  Status input_edge(int idx, DAGEdge** e) const;
  Status input_edges(std::vector<const DAGEdge*>* edges) const;

  Status input_node(int idx, const DAGNode** n) const;
  Status input_node(int idx, DAGNode** n) const;

 private:
  friend class DAG;
  DAGNode() { }

  int id_;
  DAGNodeProto def_;
  EdgeSet in_edges_;
  EdgeSet out_edges_;

  DISALLOW_COPY_AND_ASSIGN(DAGNode);
};

}  //  namespace euler

#endif   // EULER_CORE_DAG_NODE_H_
