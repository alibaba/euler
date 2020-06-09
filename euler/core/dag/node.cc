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


#include "euler/core/dag/node.h"
#include "euler/core/dag/edge.h"

namespace euler {

Status DAGNode::input_edge(int idx, DAGEdge** e) const {
  if (idx < 0 || idx > num_inputs()) {
    return Status::InvalidArgument("Invalid idx: ", idx);
  }

  for (const auto edge : in_edges_) {
    if (edge->dst_input() == idx) {
      *e = edge;
      return Status::OK();
    }
  }

  return Status::NotFound("Noinput edge: ", idx, " for node: ", name());
}

Status DAGNode::input_edges(std::vector<const DAGEdge*>* edges) const {
  if (edges == nullptr) {
    return Status::InvalidArgument("Invalid nullptr argument");
  }

  edges->clear();
  edges->resize(num_inputs(), nullptr);

  for (const auto edge : in_edges_) {
    if (edge->dst_input() < 0 || edge->dst_input() > num_inputs()) {
      return Status::Internal("Invaid edge input number: ", edge->dst_input());
    }

    if (edges->at(edge->dst_input()) != nullptr) {
      return Status::Internal("Duplicate edge, input number: ",
                              edge->dst_input());
    }

    edges->at(edge->dst_input()) = edge;
  }

  for (int i = 0; i < num_inputs(); ++i) {
    if (edges->at(i) == nullptr) {
      return Status::Internal("Missing input edge: ", i);
    }
  }

  return Status::OK();
}

Status DAGNode::input_node(int idx, const DAGNode** n) const {
  DAGNode* node;
  RETURN_IF_ERROR(input_node(idx, &node));
  *n = node;
  return Status::OK();
}

Status DAGNode::input_node(int idx, DAGNode** n) const {
  DAGEdge* edge = nullptr;
  RETURN_IF_ERROR(input_edge(idx, &edge));
  if (edge == nullptr) {
    *n = nullptr;
  } else {
    *n = edge->src();
  }
  return Status::OK();
}

}  // namespace euler
