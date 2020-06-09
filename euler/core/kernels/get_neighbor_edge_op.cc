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

#include <stdlib.h>

#include <string>

#include "euler/core/kernels/common.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"
#include "euler/core/api/api.h"
#include "euler/core/graph/graph.h"

namespace euler {

class GetNeighborEdgeOp: public OpKernel {
 public:
  explicit GetNeighborEdgeOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void GetNeighborEdgeOp::Compute(
    const DAGNodeProto& node_def, OpKernelContext* ctx) {
  Graph& graph = Graph::Instance();
  if (node_def.inputs_size() < 2) {
    EULER_LOG(ERROR) << "Argments 'node_ids' and 'edge_type'"
                     << " must be specified!";
    return;
  }

  NodeIdVec node_ids;
  auto s = GetNodeIds(node_def, 0, ctx, &node_ids);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Invalid argment 'node_ids'";
    return;
  }

  std::vector<int> edge_types;
  s = GetArg(node_def, 1, ctx, &edge_types);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Invalid argment 'edge_types'";
    return;
  }

  // get nb node
  IdWeightPairVec res = GetFullNeighbor(node_ids, edge_types);
  // filter
  if (node_def.dnf_size() > 0) {
    auto filter_ids = QueryIndexIds(node_def, ctx);
    size_t idx = 0;
    for (auto& item : res) {
      auto cur = item.data();
      for (auto& iw : item) {
        EdgeId eid(node_ids[idx], std::get<0>(iw), std::get<2>(iw));
        Graph::UID uid = graph.EdgeIdToUID(eid);
        if (filter_ids.find(uid) != filter_ids.end()) {
          *cur++ = iw;
        }
      }
      item.resize(cur - item.data());
      ++idx;
    }
  }
  // Post process
  for (auto& post : node_def.post_process()) {
    auto vec = Split(post, " ");
    if (vec[0] == "order_by") {
      if (vec.size() < 2 || vec.size() > 3) {
        EULER_LOG(ERROR) << "Invalid post process: " << post;
        continue;
      }

      int sign = 1;
      if (vec.size() == 3 && vec[2] == "desc") {
        sign = -1;
      }

      if (vec[1] == "id") {
        auto cmp = [sign] (IdWeightPair a, IdWeightPair b) {
          return (std::get<0>(a) < std::get<0>(b) ? 1 : -1) * sign > 0;
        };

        for (auto& item : res) {
          std::sort(item.begin(), item.end(), cmp);
        }
      } else if (vec[1] == "weight") {
        auto cmp = [sign] (IdWeightPair a, IdWeightPair b) {
          return (std::get<1>(a) < std::get<1>(b) ? 1 : -1) * sign > 0;
        };

        for (auto& item : res) {
          std::sort(item.begin(), item.end(), cmp);
        }
      } else {
        EULER_LOG(ERROR) << "Invalid order_by field: " << vec[1];
        continue;
      }
    } else if (vec[0] == "limit") {
      if (vec.size() != 2) {
        EULER_LOG(ERROR) << "Invalid post process: " << post;
        continue;
      }

      size_t k = atoi(vec[1].c_str());
      for (auto& item : res) {
        if (item.size() > k) {
          item.resize(k);
        }
      }
    }
  }
  // output
  FillNeighborEdge(node_def, ctx, res, node_ids);
}

REGISTER_OP_KERNEL("API_GET_NB_EDGE", GetNeighborEdgeOp);
}  // namespace euler
