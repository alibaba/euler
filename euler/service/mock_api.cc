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
#include "euler/common/logging.h"

namespace euler {

////////////////// Mock Euler Api //////////////////

NodeIdVec SampleNode(int node_type, int count) {
  EULER_LOG(INFO) << "Here: mock euler api";
  NodeIdVec node_ids(count, node_type);
  return node_ids;
}

// Get node feature
FloatFeatureVec GetNodeFloat32Feature(const NodeIdVec& node_ids,
                                      const std::vector<int>& fids) {
  FloatFeatureVec features(node_ids.size());
  for (auto& feature : features) {
    feature.resize(fids.size());
    int i = 0;
    for (auto& ff : feature) {
      ff.resize(10);
      for (auto& fff : ff) {
        fff = fids[i] + 0.5;
      }
      i++;
    }
  }
  return features;
}

FloatFeatureVec GetNodeFloat32Feature(
    const NodeIdVec& node_ids, const std::vector<std::string*>& ft_names) {
  std::vector<int> fids;
  for (auto& ft_name : ft_names) {
    fids.push_back(atoi(ft_name->c_str()));
  }
  return GetNodeFloat32Feature(node_ids, fids);
}
}  // namespace euler
