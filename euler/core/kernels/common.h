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

#ifndef EULER_CORE_KERNELS_COMMON_H_
#define EULER_CORE_KERNELS_COMMON_H_

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <memory>
#include <string>
#include <unordered_map>

#include "euler/core/api/api.h"
#include "euler/common/status.h"
#include "euler/core/graph/graph.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/index/index_manager.h"

namespace euler {

struct IdWeightType {
  uint64_t id_;
  float w_;
  int32_t t_;
};

Graph* EulerGraph();

template<typename T>
Status GetArg(const DAGNodeProto& node_def, int index,
              OpKernelContext* ctx, std::vector<T>* arg) {
  if (node_def.inputs_size() <= index) {
    return Status::InvalidArgument("Argment ", index , " not found!");
  }

  Tensor* arg_t = nullptr;
  RETURN_IF_ERROR(ctx->tensor(node_def.inputs(index), &arg_t));
  auto data = arg_t->Raw<T>();
  arg->resize(arg_t->NumElements());
  std::copy(data, data + arg_t->NumElements(), arg->begin());
  return Status::OK();
}

Status GetNodeIds(const DAGNodeProto& node_def, int index,
                  OpKernelContext* ctx, NodeIdVec* node_ids);

Status GetEdgeIds(const DAGNodeProto& node_def, int index,
                  OpKernelContext* ctx, EdgeIdVec* edge_ids);

std::vector<std::shared_ptr<IndexResult>> QueryNeighborIndex(
    const DAGNodeProto& node_def, const NodeIdVec& root_ids,
    OpKernelContext* ctx);

std::vector<std::unordered_set<Graph::UID>> QueryNeighborIndexIds(
    const DAGNodeProto& node_def,
    const NodeIdVec& root_ids,
    OpKernelContext* ctx);

std::vector<std::unordered_map<Graph::UID, int32_t>>
SampleNeighborIndexIds(
    const DAGNodeProto& node_def,
    const NodeIdVec& root_ids,
    size_t count,
    OpKernelContext* ctx);

std::shared_ptr<IndexResult> QueryIndex(const DAGNodeProto& node_def,
                                        OpKernelContext* ctx);

std::vector<std::shared_ptr<IndexResult>> QueryIndex(
    const std::string& index_name,
    const std::vector<std::string>& value);

std::unordered_set<Graph::UID> QueryIndexIds(const DAGNodeProto& node_def,
                                             OpKernelContext* ctx);

std::vector<Graph::UID> SampleByIndex(
    const DAGNodeProto& node_def, int count,
    OpKernelContext* ctx);

void Filter(const std::unordered_set<Graph::UID>& range,
            std::vector<Graph::UID>* target);

void FilerByIndex(const DAGNodeProto& node_def,
                  OpKernelContext* ctx,
                  std::vector<Graph::UID>* uids);

void FillNeighbor(const DAGNodeProto& node_def,
                  OpKernelContext* ctx, const IdWeightPairVec& result);

void FillNeighborEdge(const DAGNodeProto& node_def,
                      OpKernelContext* ctx, const IdWeightPairVec& result,
                      const NodeIdVec& root_nodes);
}  // namespace euler

#endif  // EULER_CORE_KERNELS_COMMON_H_
