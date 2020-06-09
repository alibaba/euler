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
#include "euler/common/data_types.h"

namespace euler {

class SampleNeighborOp: public OpKernel {
 public:
  explicit SampleNeighborOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void SampleNeighborOp::Compute(const DAGNodeProto& node_def,
                               OpKernelContext* ctx) {
  if (node_def.inputs_size() < 4) {
    EULER_LOG(ERROR) << "Argment 'node_ids', 'edge_types', 'count',"
                     << "'default_node' must be specified!";
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

  std::vector<int> arg;
  s = GetArg(node_def, 2, ctx, &arg);
  if (!s.ok() || arg.empty()) {
    EULER_LOG(ERROR) << "Invalid argment 'count'";
    return;
  }

  IdWeightPairVec res;
  if (node_def.dnf_size() > 0) {
    res = GetFullNeighbor(node_ids, edge_types);
    auto filters =
        SampleNeighborIndexIds(node_def, node_ids, arg[0], ctx);
    for (size_t i = 0; i < res.size(); ++i) {
      auto cur = res[i].data();
      for (auto& iw : res[i]) {
        if (filters[i].find(std::get<0>(iw)) != filters[i].end()) {
          int32_t cnt = filters[i][std::get<0>(iw)];
          for (int32_t j = 0; j < cnt; ++j) *cur++ = iw;
        }
      }
      res[i].resize(cur - res[i].data());
    }
  } else {
    res = SampleNeighbor(node_ids, edge_types, arg[0]);
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

  // uint64_t default_node = atol(node_def.inputs(3).c_str());
  uint64_t default_node = euler::common::DEFAULT_UINT64;
  for (auto& item : res) {
    if (item.empty()) {
      item.reserve(arg[0]);
      for (int32_t i = 0; i < arg[0]; ++i) {
        item.push_back(IdWeightPair(default_node, 0, 0));
      }
    }
  }
  FillNeighbor(node_def, ctx, res);
}

REGISTER_OP_KERNEL("API_SAMPLE_NB", SampleNeighborOp);

}  // namespace euler
