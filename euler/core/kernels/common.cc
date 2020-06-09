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

#include "euler/core/kernels/common.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <string>
#include <unordered_map>

#include "euler/common/str_util.h"

namespace euler {

std::string GetParams(OpKernelContext* ctx, std::string p_name) {
  Tensor* t = nullptr;
  if (ctx->tensor(p_name, &t).ok()) {
    if (t->Type() == kString) {
      return *(t->Raw<std::string*>()[0]);
    } else {
      EULER_LOG(FATAL) << "parmas type error";
      return p_name;
    }
  } else {
    return p_name;
  }
}

Graph* EulerGraph() {
  return &Graph::Instance();
}

Status GetNodeIds(const DAGNodeProto& node_def, int index,
                  OpKernelContext* ctx, NodeIdVec* node_ids) {
  Tensor* node_ids_t = nullptr;
  RETURN_IF_ERROR(ctx->tensor(node_def.inputs(index), &node_ids_t));
  node_ids->resize(node_ids_t->NumElements(), 0);
  auto data = node_ids_t->Raw<int64_t>();
  std::copy(data, data + node_ids->size(), node_ids->begin());
  return Status::OK();
}

Status GetEdgeIds(const DAGNodeProto& node_def, int index,
                  OpKernelContext* ctx, EdgeIdVec* edge_ids) {
  Tensor* edge_ids_t = nullptr;
  RETURN_IF_ERROR(ctx->tensor(node_def.inputs(index), &edge_ids_t));
  if (edge_ids_t->NumElements() % 3 != 0) {
    return Status::InvalidArgument("Invalid edge_ids shape, with ",
                                   edge_ids_t->NumElements(), " elements!");
  }

  int count = edge_ids_t->NumElements() / 3;
  edge_ids->reserve(count);

  auto data = edge_ids_t->Raw<int64_t>();
  for (int i = 0; i < count; ++i) {
    EdgeId eid{data[0], data[1], data[2]};
    edge_ids->push_back(eid);
    data += 3;
  }

  return Status::OK();
}

std::shared_ptr<IndexResult> QueryIndex(
    const DAGNodeProto& node_def, OpKernelContext* ctx) {
  IndexManager& index_manager = IndexManager::Instance();
  std::shared_ptr<IndexResult> result(nullptr);

  for (const auto& dnf : node_def.dnf()) {
    auto tokens = Split(dnf, ",");
    std::shared_ptr<IndexResult> row_result(nullptr);
    for (auto& token : tokens) {
      auto v = Split(token, " ");
      if (v.size() != 3) {
        EULER_LOG(ERROR) << "DNF must be triple";
        return nullptr;
      }
      auto& fn = v[0];
      auto& op = v[1];
      auto value = GetParams(ctx, v[2]);
      auto index = index_manager.GetIndex(fn);
      if (index == nullptr) {
        EULER_LOG(ERROR) << "No index on field: " << fn;
        return nullptr;
      }

      auto tmp = index->Search(op, value);
      if (row_result == nullptr) {
        row_result = tmp;
      } else {
        row_result->Intersection(tmp);
      }
    }

    if (result == nullptr) {
      result = row_result;
    } else {
      result->Union(row_result);
    }
  }
  return result;
}

std::vector<std::shared_ptr<IndexResult>> QueryIndex(
    const std::string& index_name,
    const std::vector<std::string>& value) {
  std::vector<std::shared_ptr<IndexResult>> results;
  results.reserve(value.size());
  IndexManager& index_manager = IndexManager::Instance();
  auto index = index_manager.GetIndex(index_name);
  if (index == nullptr) {
    EULER_LOG(FATAL) << "no such index: " << index_name;
  }
  for (const std::string& v : value) {
    results.push_back(index->Search("eq", v));
  }
  return results;
}

std::vector<std::shared_ptr<IndexResult>> QueryNeighborIndex(
    const DAGNodeProto& node_def, const NodeIdVec& root_ids,
    OpKernelContext* ctx) {
  IndexManager& index_manager = IndexManager::Instance();
  std::vector<std::shared_ptr<IndexResult>> results(root_ids.size());
  for (size_t i = 0; i < root_ids.size(); ++i) {
    std::shared_ptr<IndexResult> result(nullptr);
    for (const auto& dnf : node_def.dnf()) {
      auto tokens = Split(dnf, ",");
      std::shared_ptr<IndexResult> row_result(nullptr);
      for (auto& token : tokens) {
        auto v = Split(token, " ");
        if (v.size() != 3) {
          EULER_LOG(ERROR) << "DNF must be triple";
          return results;
        }
        auto& fn = v[0];
        auto& op = v[1];
        auto value = ToString(root_ids[i], "::", GetParams(ctx, v[2]));
        auto index = index_manager.GetIndex(fn);
        if (index == nullptr) {
          EULER_LOG(ERROR) << "No index on field: " << fn;
          return results;
        }

        auto tmp = index->Search(op, value);
        if (row_result == nullptr) {
          row_result = tmp;
        } else {
          row_result->Intersection(tmp);
        }
      }
      if (result == nullptr) {
        result = row_result;
      } else {
        result->Union(row_result);
      }
    }
    results[i] = result;
  }
  return results;
}

std::unordered_set<Graph::UID> QueryIndexIds(
    const DAGNodeProto& node_def, OpKernelContext* ctx) {
  auto result = QueryIndex(node_def, ctx);
  if (result == nullptr) {
    return std::unordered_set<Graph::UID>();
  }

  auto filtered_ids = result->GetIds();
  return std::unordered_set<Graph::UID>(
      filtered_ids.begin(), filtered_ids.end());
}

std::vector<std::unordered_set<Graph::UID>>
QueryNeighborIndexIds(const DAGNodeProto& node_def,
                      const NodeIdVec& root_ids,
                      OpKernelContext* ctx) {
  std::vector<std::unordered_set<Graph::UID>> results;
  results.reserve(root_ids.size());
  std::vector<std::shared_ptr<IndexResult>> index_results =
      QueryNeighborIndex(node_def, root_ids, ctx);
  for (auto index_result : index_results) {
    if (index_result == nullptr) {
      results.push_back(std::unordered_set<Graph::UID>());
    } else {
      auto filtered_ids = index_result->GetIds();
      std::unordered_set<Graph::UID> result(
        filtered_ids.begin(), filtered_ids.end());
      results.emplace_back(result);
    }
  }
  return results;
}

std::vector<std::unordered_map<Graph::UID, int32_t>>
SampleNeighborIndexIds(const DAGNodeProto& node_def,
                       const NodeIdVec& root_ids,
                       size_t count,
                       OpKernelContext* ctx) {
  std::vector<std::unordered_map<Graph::UID, int32_t>> results;
  results.reserve(root_ids.size());
  std::vector<std::shared_ptr<IndexResult>> index_results =
      QueryNeighborIndex(node_def, root_ids, ctx);
  for (auto index_result : index_results) {
    if (index_result == nullptr) {
      results.push_back(std::unordered_map<Graph::UID, int32_t>());
    } else {
      std::vector<std::pair<uint64_t, float>> filtered_ids_weight =
          index_result->Sample(count);
      std::unordered_map<Graph::UID, int32_t> result(count * 2);
      for (const auto& p : filtered_ids_weight) {
        result[std::get<0>(p)] += 1;
      }
      results.emplace_back(result);
    }
  }
  return results;
}

std::vector<Graph::UID> SampleByIndex(
    const DAGNodeProto& node_def, int count,
    OpKernelContext* ctx) {
  auto result = QueryIndex(node_def, ctx);
  if (result == nullptr) {
    return std::vector<Graph::UID>();
  }

  auto pairs = result->Sample(count);
  std::vector<Graph::UID> ids(count);
  for (size_t i = 0; i < pairs.size(); ++i) {
    ids[i] = pairs[i].first;
  }
  return ids;
}

void Filter(const std::unordered_set<Graph::UID>& range,
            std::vector<Graph::UID>* target) {
  auto cur = target->data();
  for (auto it = target->begin(); it != target->end(); ++it) {
    if (range.find(*it) != range.end()) {  // contains
      *cur++ = *it;
    }
  }
  target->resize(cur - target->data());
}

void FilerByIndex(const DAGNodeProto& node_def,
                  OpKernelContext* ctx,
                  std::vector<Graph::UID>* uids) {
  if (node_def.dnf_size() > 0) {
    auto filter_ids = QueryIndexIds(node_def, ctx);
    if (uids->empty()) {
      std::copy(filter_ids.begin(), filter_ids.end(), uids->begin());
    } else {
      Filter(filter_ids, uids);
    }
  }
}

void FillNeighbor(const DAGNodeProto& node_def,
                  OpKernelContext* ctx, const IdWeightPairVec& result) {
  // Fill Index tensor
  TensorShape index_shape({result.size(), 2});
  auto index_name = OutputName(node_def, 0);
  Tensor* index_tensor = nullptr;
  size_t offset = 0;
  auto s = ctx->Allocate(index_name, index_shape,
                         DataType::kInt32, &index_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '" << index_name << "' failed!";
    return;
  }
  auto index_data = index_tensor->Raw<int32_t>();
  for (auto& item : result) {
    index_data[0] = offset;
    index_data[1] = offset + item.size();
    offset += item.size();
    index_data += 2;
  }

  TensorShape value_shape({offset});
  auto id_name = OutputName(node_def, 1);
  auto weight_name = OutputName(node_def, 2);
  auto type_name = OutputName(node_def, 3);
  Tensor* id_tensor = nullptr;
  Tensor* weight_tensor = nullptr;
  Tensor* type_tensor = nullptr;

  s = ctx->Allocate(id_name, value_shape, DataType::kUInt64, &id_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '" << id_name << "' failed!";
    return;
  }

  s = ctx->Allocate(weight_name, value_shape, DataType::kFloat, &weight_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '"
                     << weight_name << "' failed!";
    return;
  }

  s = ctx->Allocate(type_name, value_shape, DataType::kInt32, &type_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '"
                     << type_name << "' failed!";
    return;
  }

  auto id_data = id_tensor->Raw<uint64_t>();
  auto weight_data = weight_tensor->Raw<float>();
  auto type_data = type_tensor->Raw<int32_t>();
  for (auto& item : result) {
    for (auto& iw : item) {
      *id_data++ = std::get<0>(iw);
      *weight_data++ = std::get<1>(iw);
      *type_data++ = std::get<2>(iw);
    }
  }
}

void FillNeighborEdge(const DAGNodeProto& node_def,
                      OpKernelContext* ctx, const IdWeightPairVec& result,
                      const NodeIdVec& root_nodes) {
  // fill index tensor
  TensorShape index_shape({result.size(), 2});
  auto index_name = OutputName(node_def, 0);
  Tensor* index_tensor = nullptr;
  size_t offset = 0;
  auto s = ctx->Allocate(index_name, index_shape,
                         DataType::kInt32, &index_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '" << index_name << "' failed!";
    return;
  }
  auto index_data = index_tensor->Raw<int32_t>();
  for (auto& item : result) {
    index_data[0] = offset;
    index_data[1] = offset + item.size();
    offset += item.size();
    index_data += 2;
  }

  // fill data
  TensorShape eid_shape({offset, 3});
  TensorShape weight_shape({offset});
  auto eid_name = OutputName(node_def, 1);
  auto weight_name = OutputName(node_def, 2);
  Tensor* eid_tensor = nullptr;
  Tensor* weight_tensor = nullptr;

  s = ctx->Allocate(eid_name, eid_shape, DataType::kUInt64, &eid_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '" << eid_name << "' failed!";
    return;
  }

  s = ctx->Allocate(weight_name, weight_shape,
                    DataType::kFloat, &weight_tensor);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '"
                     << weight_name << "' failed!";
    return;
  }

  auto eid_data = eid_tensor->Raw<uint64_t>();
  auto weight_data = weight_tensor->Raw<float>();
  size_t idx = 0;
  for (auto& item : result) {
    uint64_t node_id = root_nodes[idx];
    for (auto& iw : item) {
      *eid_data++ = node_id;
      *eid_data++ = std::get<0>(iw);
      *eid_data++ = static_cast<uint64_t>(std::get<2>(iw));
      *weight_data++ = std::get<1>(iw);
    }
    ++idx;
  }
}

}  // namespace euler
