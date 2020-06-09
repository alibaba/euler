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

#ifndef EULER_COMMON_DATA_TYPES_H_
#define EULER_COMMON_DATA_TYPES_H_

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <utility>
#include <tuple>
#include <limits>

#include "euler/common/hash.h"

namespace euler {
namespace common {

extern const float DEFAULT_FLOAT;

extern const int32_t DEFAULT_INT32;

extern const int64_t DEFAULT_INT64;

extern const uint64_t DEFAULT_UINT64;

extern const char DEFAULT_CHAR;

typedef uint64_t NodeID;

typedef std::tuple<NodeID, NodeID, int32_t> EdgeID;

typedef std::tuple<NodeID, float, int32_t> IDWeightPair;

struct EdgeIDHashFunc {
  std::size_t operator()(const EdgeID& key) const {
    char tmp[20];
    memcpy(tmp, &std::get<0>(key), 8);
    memcpy(tmp + 8, &std::get<1>(key), 8);
    memcpy(tmp + 16, &std::get<2>(key), 4);
    return euler::hash64(tmp, 20);
  }
};

struct EdgeIDEqualKey {
  bool operator()(const EdgeID& key1, const EdgeID& key2) const {
    return std::get<0>(key1) == std::get<0>(key2) &&
           std::get<1>(key1) == std::get<1>(key2) &&
           std::get<2>(key1) == std::get<2>(key2);
  }
};

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_DATA_TYPES_H_
