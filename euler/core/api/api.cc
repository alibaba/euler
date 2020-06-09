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

#include "euler/core/api/api.h"

#include <omp.h>

#include "euler/common/data_types.h"

namespace euler {

namespace {

Graph* EulerGraph() {
  return &Graph::Instance();
}

}  // namespace

NodeIdVec SampleNode(const std::vector<int>& node_types, int count) {
  if (node_types.size() == 1) {
    return EulerGraph()->SampleNode(node_types[0], count);
  }
  return EulerGraph()->SampleNode(node_types, count);
}

EdgeIdVec SampleEdge(const std::vector<int>& edge_types, int count) {
  if (edge_types.size() == 1) {
    return EulerGraph()->SampleEdge(edge_types[0], count);
  }
  return EulerGraph()->SampleEdge(edge_types, count);
}

bool EdgeExist(const EdgeId& eid) {
  return EulerGraph()->GetEdgeByID(eid) != nullptr;
}

TypeVec GetNodeType(const std::vector<NodeId>& node_ids) {
  TypeVec vec(node_ids.size());
  for (size_t i = 0 ; i < node_ids.size(); ++i) {
    auto node = EulerGraph()->GetNodeByID(node_ids[i]);
    if (node != nullptr) {
      vec[i] = node->GetType();
    } else {
      vec[i] = euler::common::DEFAULT_INT32;
    }
  }
  return vec;
}

FloatFeatureVec GetNodeFloat32Feature(const NodeIdVec& node_ids,
                                      const std::vector<int>& fids) {
  FloatFeatureVec features(node_ids.size());
  #ifdef OPENMP
  #pragma omp parallel for
  #endif
  for (int32_t i = 0 ; i < static_cast<int32_t>(node_ids.size()); ++i) {
    auto node = EulerGraph()->GetNodeByID(node_ids[i]);
    if (node != nullptr) {
      node->GetFloat32Feature(fids, &features[i]);
    } else {
      features[i].resize(fids.size());
    }
  }
  return features;
}

UInt64FeatureVec GetNodeUint64Feature(const NodeIdVec& node_ids,
                                      const std::vector<int>& fids) {
  UInt64FeatureVec features(node_ids.size());
  #ifdef OPENMP
  #pragma omp parallel for
  #endif
  for (int32_t i = 0 ; i < static_cast<int32_t>(node_ids.size()); ++i) {
    auto node = EulerGraph()->GetNodeByID(node_ids[i]);
    if (node != nullptr) {
      node->GetUint64Feature(fids, &features[i]);
    } else {
      features[i].resize(fids.size());
    }
  }
  return features;
}

BinaryFatureVec GetNodeBinaryFeature(const NodeIdVec& node_ids,
                                     const std::vector<int>& fids) {
  BinaryFatureVec features(node_ids.size());
  #ifdef OPENMP
  #pragma omp parallel for
  #endif
  for (int32_t i = 0 ; i < static_cast<int32_t>(node_ids.size()); ++i) {
    auto node = EulerGraph()->GetNodeByID(node_ids[i]);
    if (node != nullptr) {
      node->GetBinaryFeature(fids, &features[i]);
    } else {
      features[i].resize(fids.size());
    }
  }
  return features;
}

std::vector<int> GetNodeFeatureIds(const std::vector<std::string*>& ft_names) {
  std::vector<int> feature_ids(ft_names.size());
  for (size_t i = 0; i < ft_names.size(); ++i) {
    feature_ids[i] = EulerGraph()->GetNodeFeatureId(*ft_names[i]);
  }
  return feature_ids;
}

std::vector<int> GetEdgeFeatureIds(const std::vector<std::string*>& ft_names) {
  std::vector<int> feature_ids(ft_names.size());
  for (size_t i = 0; i < ft_names.size(); ++i) {
    feature_ids[i] = EulerGraph()->GetEdgeFeatureId(*ft_names[i]);
  }
  return feature_ids;
}

FloatFeatureVec GetNodeFloat32Feature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names) {
  auto fids = GetNodeFeatureIds(ft_names);
  return GetNodeFloat32Feature(node_ids, fids);
}

UInt64FeatureVec GetNodeUint64Feature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names) {
  auto fids = GetNodeFeatureIds(ft_names);
  return GetNodeUint64Feature(node_ids, fids);
}

BinaryFatureVec GetNodeBinaryFeature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names) {
  auto fids = GetNodeFeatureIds(ft_names);
  return GetNodeBinaryFeature(node_ids, fids);
}

FloatFeatureVec GetEdgeFloat32Feature(const EdgeIdVec& edge_ids,
                                      const std::vector<int>& fids) {
  FloatFeatureVec features(edge_ids.size());
  for (size_t i = 0 ; i < edge_ids.size(); ++i) {
    auto edge = EulerGraph()->GetEdgeByID(edge_ids[i]);
    if (edge != nullptr) {
      edge->GetFloat32Feature(fids, &features[i]);
    } else {
      features[i].resize(fids.size());
    }
  }
  return features;
}

UInt64FeatureVec GetEdgeUint64Feature(const EdgeIdVec& edge_ids,
                                      const std::vector<int32_t>& fids) {
  UInt64FeatureVec features(edge_ids.size());
  for (size_t i = 0 ; i < edge_ids.size(); ++i) {
    auto edge = EulerGraph()->GetEdgeByID(edge_ids[i]);
    if (edge != nullptr) {
      edge->GetUint64Feature(fids, &features[i]);
    } else {
      features[i].resize(fids.size());
    }
  }
  return features;
}

BinaryFatureVec GetEdgeBinaryFeature(const EdgeIdVec& edge_ids,
                                     const std::vector<int>& fids) {
  BinaryFatureVec features(edge_ids.size());
  for (size_t i = 0 ; i < edge_ids.size(); ++i) {
    auto edge = EulerGraph()->GetEdgeByID(edge_ids[i]);
    if (edge != nullptr) {
      edge->GetBinaryFeature(fids, &features[i]);
    } else {
      features[i].resize(fids.size());
    }
  }
  return features;
}

FloatFeatureVec GetEdgeFloat32Feature(
    const EdgeIdVec& edge_ids, const std::vector<std::string*>& ft_names) {
  auto fids = GetEdgeFeatureIds(ft_names);
  return GetEdgeFloat32Feature(edge_ids, fids);
}

UInt64FeatureVec GetEdgeUint64Feature(
    const EdgeIdVec& edge_ids, const std::vector<std::string*>& ft_names) {
  auto fids =  GetEdgeFeatureIds(ft_names);
  return GetEdgeUint64Feature(edge_ids, fids);
}

BinaryFatureVec GetEdgeBinaryFeature(
    const EdgeIdVec& edge_ids, const std::vector<std::string*>& ft_names) {
  auto fids = GetEdgeFeatureIds(ft_names);
  return GetEdgeBinaryFeature(edge_ids, fids);
}

IdWeightPairVec GetFullNeighbor(const NodeIdVec& node_ids,
                                const std::vector<int>& edge_types) {
  IdWeightPairVec neighbor(node_ids.size());
  #ifdef OPENMP
  #pragma omp parallel for
  #endif
  for (int32_t i = 0 ; i < static_cast<int32_t>(node_ids.size()); ++i) {
    auto node = EulerGraph()->GetNodeByID(node_ids[i]);
    if (node != nullptr) {
      neighbor[i] = node->GetFullNeighbor(edge_types);
    }
  }
  return neighbor;
}

IdWeightPairVec SampleNeighbor(const NodeIdVec& node_ids,
                               const std::vector<int>& edge_types, int count) {
  IdWeightPairVec neighbor(node_ids.size());
  #ifdef OPENMP
  #pragma omp parallel for
  #endif
  for (int32_t i = 0 ; i < static_cast<int32_t>(node_ids.size()); ++i) {
    auto node = EulerGraph()->GetNodeByID(node_ids[i]);
    if (node != nullptr) {
      neighbor[i] = node->SampleNeighbor(edge_types, count);
    }
  }
  return neighbor;
}

bool GetNodeType(const std::vector<std::string*> node_types,
                 std::vector<int>* type_ids) {
  type_ids->resize(node_types.size());
  for (size_t i = 0; i < node_types.size(); ++i) {
    if (!EulerGraph()->GetNodeTypeByName(*node_types[i], &(type_ids->at(i)))) {
      type_ids->clear();
      return false;
    }
  }

  return true;
}

bool GetEdgeType(const std::vector<std::string*> edge_types,
                 std::vector<int>* type_ids) {
  type_ids->resize(edge_types.size());
  for (size_t i = 0; i < edge_types.size(); ++i) {
    if (!EulerGraph()->GetEdgeTypeByName(*edge_types[i], &(type_ids->at(i)))) {
      type_ids->clear();
      return false;
    }
  }

  return true;
}

bool GetNodeType(const std::string& node_type, int* type_id) {
  if (node_type.empty()) {
    *type_id = -1;
    return true;
  }

  return EulerGraph()->GetNodeTypeByName(node_type, type_id);
}

bool GetEdgeType(const std::string& edge_type, int* type_id) {
  if (edge_type.empty()) {
    *type_id = -1;
    return true;
  }

  return EulerGraph()->GetEdgeTypeByName(edge_type, type_id);
}

}  // namespace euler
