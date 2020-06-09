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

class GetNeighborOp: public OpKernel {
 public:
  explicit GetNeighborOp(const std::string& name): OpKernel(name) {
    IndexManager& index_manager = IndexManager::Instance();
    Meta meta = index_manager.GetIndexInfo()[0];
    std::string index_info = meta["index_info"];
    std::vector<std::string> index_names = Split(index_info, ",");
    for (const std::string& index_name : index_names) {
      std::vector<std::string> name_type = Split(index_name, ":");
      if (name_type[1] == "hash_range_index") {
        nb_index_set_.insert(name_type[0]);
      }
    }
  }
  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);

 private:
  std::unordered_set<std::string> nb_index_set_;
  bool IsNeighborIndex(const DAGNodeProto& node_def);
};

bool GetNeighborOp::IsNeighborIndex(const DAGNodeProto& node_def) {
  if (node_def.dnf_size() > 0) {
    auto& dnf = node_def.dnf(0);
    auto tokens = Split(dnf, ",");
    auto& token = tokens[0];
    auto v = Split(token, " ");
    if (v.size() != 3) {
      EULER_LOG(FATAL) << "DNF must be triple";
    }
    std::string fn = v[0];
    if (nb_index_set_.find(fn) == nb_index_set_.end()) return false;
    return true;
  } else {
    return false;
  }
}

void GetNeighborOp::Compute(const DAGNodeProto& node_def,
                            OpKernelContext* ctx) {
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

  IdWeightPairVec res = GetFullNeighbor(node_ids, edge_types);
  if (!IsNeighborIndex(node_def)) {
    // Filter
    if (node_def.dnf_size() > 0) {
      auto filter_ids = QueryIndexIds(node_def, ctx);
      for (auto& item : res) {
        auto cur = item.data();
        for (auto& iw : item) {
          if (filter_ids.find(std::get<0>(iw)) != filter_ids.end()) {
            *cur++ = iw;
          }
        }
        item.resize(cur - item.data());
      }
    }
  } else {
    if (node_def.dnf_size() > 0) {
      auto filters =
          QueryNeighborIndexIds(node_def, node_ids, ctx);
      for (size_t i = 0; i < res.size(); ++i) {
        auto cur = res[i].data();
        for (auto& iw : res[i]) {
          if (filters[i].find(std::get<0>(iw)) != filters[i].end()) {
            *cur++ = iw;
          }
        }
        res[i].resize(cur - res[i].data());
      }
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
          return (std::get<0>(a) <= std::get<0>(b) ? 1 : -1) * sign > 0;
        };

        for (auto& item : res) {
          std::sort(item.begin(), item.end(), cmp);
        }
      } else if (vec[1] == "weight") {
        auto cmp = [sign] (IdWeightPair a, IdWeightPair b) {
          return (std::get<1>(a) <= std::get<1>(b) ? 1 : -1) * sign > 0;
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

  FillNeighbor(node_def, ctx, res);
}

REGISTER_OP_KERNEL("API_GET_NB_NODE", GetNeighborOp);

}  // namespace euler
