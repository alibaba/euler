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

#ifndef EULER_CORE_INDEX_INDEX_UTIL_H_
#define EULER_CORE_INDEX_INDEX_UTIL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <utility>

#include "euler/core/index/index_types.h"

namespace euler {

template<typename T>
inline void VecToPairVec(const std::vector<T>& ids,
                         const std::vector<float>& weights,
                         std::vector<std::pair<T, float>>* v) {
  for (size_t i = 0; i < ids.size(); ++i) {
    v->push_back(std::make_pair(ids[i], weights[i]));
  }
}

template<typename PairIter, typename Iter, typename OutputIter>
inline OutputIter intersection(PairIter first1, PairIter last1,
                    Iter first2, Iter last2,
                    OutputIter result) {
  while (first1 != last1 && first2 != last2) {
    if ((*first1).first < uint64_t(*first2)) {
      ++first1;
    } else if (uint64_t(*first2) < (*first1).first) {
      ++first2;
    } else {
      *result = *first1;
      ++result; ++first1; ++first2;
    }
  }
  return result;
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_INDEX_UTIL_H_
