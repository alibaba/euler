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

#ifndef EULER_CORE_API_API_H_
#define EULER_CORE_API_API_H_

#include <string>
#include <vector>

#include "euler/common/data_types.h"
#include "euler/core/graph/graph.h"

namespace euler {

Graph** GetGraph();

using NodeId = euler::common::NodeID;
using EdgeId = euler::common::EdgeID;
using IdWeightPair = euler::common::IDWeightPair;

typedef std::vector<std::vector<IdWeightPair>> IdWeightPairVec;
typedef std::vector<NodeId> NodeIdVec;
typedef std::vector<EdgeId> EdgeIdVec;
typedef std::vector<int32_t> TypeVec;
typedef std::vector<std::vector<std::vector<float>>> FloatFeatureVec;
typedef std::vector<std::vector<std::vector<uint64_t>>>  UInt64FeatureVec;
typedef std::vector<std::vector<std::string>> BinaryFatureVec;

bool EdgeExist(const EdgeId& eid);

// Sample
NodeIdVec SampleNode(const std::vector<int>& node_types, int count);
EdgeIdVec SampleEdge(const std::vector<int>& edge_types, int count);

// Get ndoe type
TypeVec GetNodeType(const NodeIdVec& node_ids);

// Get node feature
FloatFeatureVec GetNodeFloat32Feature(const NodeIdVec& node_ids,
                                      const std::vector<int>& fids);
UInt64FeatureVec GetNodeUint64Feature(const NodeIdVec& node_ids,
                                      const std::vector<int>& fids);
BinaryFatureVec GetNodeBinaryFeature(const NodeIdVec& node_ids,
                                     const std::vector<int>& fids);

FloatFeatureVec GetNodeFloat32Feature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names);
UInt64FeatureVec GetNodeUint64Feature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names);
BinaryFatureVec GetNodeBinaryFeature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names);

// Get edge feature
FloatFeatureVec GetEdgeFloat32Feature(const EdgeIdVec& edge_ids,
                                      const std::vector<int>& fids);
UInt64FeatureVec GetEdgeUint64Feature(const EdgeIdVec& edge_ids,
                                      const std::vector<int>& fids);
BinaryFatureVec GetEdgeBinaryFeature(const EdgeIdVec& edge_ids,
                                     const std::vector<int>& fids);

FloatFeatureVec GetEdgeFloat32Feature(
    const EdgeIdVec& edge_ids, const std::vector<std::string*>& ft_names);
UInt64FeatureVec GetEdgeUint64Feature(
    const EdgeIdVec& edge_ids, const std::vector<std::string*>& ft_names);
BinaryFatureVec GetEdgeBinaryFeature(
    const EdgeIdVec& edge_ids, const std::vector<std::string*>& ft_names);

// Get neighbor
IdWeightPairVec GetFullNeighbor(const NodeIdVec& node_ids,
                                const std::vector<int>& edge_types);
IdWeightPairVec SampleNeighbor(const NodeIdVec& node_ids,
                               const std::vector<int>& edge_types, int count);

bool GetNodeType(const std::vector<std::string*> node_types,
                 std::vector<int>* type_ids);
bool GetEdgeType(const std::vector<std::string*> edge_types,
                 std::vector<int>* type_ids);

bool GetNodeType(const std::string& node_type, int* type_id);
bool GetEdgeType(const std::string& edge_type, int* type_id);

}  // namespace euler

#endif  // EULER_CORE_API_API_H_
