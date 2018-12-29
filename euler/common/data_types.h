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

#ifndef EULER_COMMON_DATA_TYPES_H_
#define EULER_COMMON_DATA_TYPES_H_

#include <stdint.h>

#include <algorithm>
#include <functional>
#include <utility>
#include <tuple>

namespace euler {
namespace common {

typedef uint64_t NodeID;

typedef std::tuple<NodeID, NodeID, int32_t> EdgeID;

typedef std::tuple<NodeID, float, int32_t> IDWeightPair;

struct EdgeIDHashFunc {
  std::size_t operator()(const EdgeID& key) const {
    std::hash<uint64_t> hasher;
    const std::size_t k_mul = 0x9ddfea08eb382d69ULL;
    std::size_t a = (hasher(std::get<0>(key)) ^
                     hasher(std::get<1>(key))) * k_mul;
    a ^= (a >> 47);
    std::size_t b = (hasher(std::get<0>(key)) ^ a) * k_mul;
    b ^= (b >> 47);
    std::size_t temp = b * k_mul;
    std::size_t c = (hasher(temp) ^ hasher(std::get<2>(key))) * k_mul;
    c^= (c >> 47);
    std::size_t d = (hasher(temp) ^ c) * k_mul;
    d ^= (d >> 47);
    return d * k_mul;
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
