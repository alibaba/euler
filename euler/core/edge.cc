/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "euler/core/edge.h"

#include "euler/common/bytes_reader.h"

namespace euler {
namespace core {

#define GET_EDGE_FEATURE(F_NUMS_PTR, F_VALUES_PTR, FEATURES,           \
                         FEATURES_IDX, FIDS) {                         \
  for (size_t i = 0; i < FIDS.size(); ++i) {                           \
    int32_t fid = FIDS[i];                                             \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) { \
      int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];              \
      F_NUMS_PTR->push_back(FEATURES_IDX[fid] - pre);                  \
    } else {                                                           \
      F_NUMS_PTR->push_back(0);                                        \
    }                                                                  \
  }                                                                    \
  for (size_t i = 0; i < FIDS.size(); ++i) {                           \
    int32_t fid = FIDS[i];                                             \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) { \
      int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];              \
      int32_t now = FEATURES_IDX[fid];                                 \
      F_VALUES_PTR->insert(F_VALUES_PTR->end(),                        \
                           FEATURES.begin() + pre,                     \
                           FEATURES.begin() + now);                    \
    }                                                                  \
  }                                                                    \
}

void Edge::GetUint64Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_EDGE_FEATURE(feature_nums, feature_values, uint64_features_,
                   uint64_features_idx_, fids);
}

void Edge::GetFloat32Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_EDGE_FEATURE(feature_nums, feature_values, float_features_,
                   float_features_idx_, fids);
}

void Edge::GetBinaryFeature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_EDGE_FEATURE(feature_nums, feature_values, binary_features_,
                   binary_features_idx_, fids);
}

}  // namespace core
}  // namespace euler
