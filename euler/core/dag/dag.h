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


#ifndef EULER_CORE_DAG_DAG_H_
#define EULER_CORE_DAG_DAG_H_

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>
#include <memory>

#include "euler/common/macros.h"
#include "euler/common/status.h"

namespace euler {

class DAGNode;
class DAGEdge;
class DAGNodeProto;
class DAGProto;

class DAG {
 public:
  explicit DAG(const std::string& name);
  ~DAG();

  std::string name() { return name_; }

  int num_nodes() const { return nodes_.size(); }
  int num_edges() const { return edges_.size(); }

  std::vector<DAGNode*> nodes() { return nodes_; }
  std::vector<DAGEdge*> edges() { return edges_; }

  DAGNode* AddNode(const DAGNodeProto& node_def, Status* s);
  DAGEdge* AddEdge(DAGNode* src, int x, DAGNode* dst, int y);

  static std::unique_ptr<DAG> NewFromProto(const DAGProto& dag_def);

 private:
  std::string name_;
  std::vector<DAGNode*> nodes_;
  std::vector<DAGEdge*> edges_;

  DISALLOW_COPY_AND_ASSIGN(DAG);
};

}  //  namespace euler

#endif  // EULER_CORE_DAG_DAG_H_
