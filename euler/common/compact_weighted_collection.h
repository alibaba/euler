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

#ifndef EULER_COMMON_COMPACT_WEIGHTED_COLLECTION_H_
#define EULER_COMMON_COMPACT_WEIGHTED_COLLECTION_H_

#include <iostream>
#include <vector>
#include <utility>

#include "glog/logging.h"

#include "euler/common/weighted_collection.h"
#include "euler/common/random.h"

namespace euler {
namespace common {

template<class T>
size_t RandomSelect(const std::vector<float>& sum_weights,
                    size_t begin_pos, size_t end_pos) {
  float limit_begin = begin_pos == 0 ? 0 : sum_weights[begin_pos - 1];
  float limit_end = sum_weights[end_pos];
  float r = ThreadLocalRandom() * (limit_end - limit_begin) +
            limit_begin;
  size_t low = begin_pos, high = end_pos, mid = 0;
  bool finish = false;
  while(low <= high && !finish) {
    mid = (low + high) / 2;
    float interval_begin = mid == 0 ? 0 : sum_weights[mid - 1];
    float interval_end = sum_weights[mid];
    if (interval_begin <= r && r < interval_end) {
      finish = true;
    } else if (interval_begin > r) {
      high = mid - 1;
    } else if (interval_end <= r) {
      low = mid + 1;
    }
  }
  return mid;
}

template<class T>
class CompactWeightedCollection : public WeightedCollection<T> {
 public:
  CompactWeightedCollection() {
  }

  void Init(const std::vector<T>& ids,
            const std::vector<float>& weights) override;

  void Init(const std::vector<std::pair<T, float>>& id_weight_pairs) override;

  std::pair<T, float> Sample() const override;

  size_t GetSize() const override;

  std::pair<T, float> Get(size_t idx) const override;

  float GetSumWeight() const override;

 private:
  std::vector<T> ids_;
  std::vector<float> sum_weights_;
  float sum_weight_;
};

template<class T>
void CompactWeightedCollection<T>::Init(const std::vector<T>& ids,
                                        const std::vector<float>& weights) {
  if (ids.size() == weights.size()) {
    sum_weight_ = 0.0;
    ids_.resize(ids.size());
    sum_weights_.resize(weights.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      ids_[i] = ids[i];
      sum_weight_ += weights[i];
      sum_weights_[i] = sum_weight_;
    }
  } else {
    LOG(ERROR) << "ids size != weights size, init error";
  }
}

template<class T>
void CompactWeightedCollection<T>::Init(
    const std::vector<std::pair<T, float>>& id_weight_pairs) {
  sum_weight_ = 0.0;
  ids_.resize(id_weight_pairs.size());
  sum_weights_.resize(id_weight_pairs.size());
  for (size_t i = 0; i < id_weight_pairs.size(); ++i) {
    ids_[i] = id_weight_pairs[i].first;
    sum_weight_ += id_weight_pairs[i].second;
    sum_weights_[i] = sum_weight_;
  }
}

template<class T>
std::pair<T, float> CompactWeightedCollection<T>::Sample() const {
  size_t mid = RandomSelect<T>(sum_weights_, 0, ids_.size() - 1);
  float pre_sum_weight = 0;
  if (mid > 0) {
    pre_sum_weight = sum_weights_[mid - 1];
  }
  std::pair<T, float> id_weight_pair(
      ids_[mid], sum_weights_[mid] - pre_sum_weight);
  return id_weight_pair;
}

template<class T>
size_t CompactWeightedCollection<T>::GetSize() const {
  return ids_.size();
}

template<class T>
std::pair<T, float> CompactWeightedCollection<T>::Get(size_t idx) const {
  if (idx <= ids_.size()) {
    float pre_sum_weight = 0;
    if (idx > 0) {
      pre_sum_weight = sum_weights_[idx - 1];
    }
    std::pair<T, float> id_weight_pair(
        ids_[idx], sum_weights_[idx] - pre_sum_weight);
    return id_weight_pair;
  } else {
    LOG(ERROR) << "idx out of boundary";
    return std::pair<T, float>();
  }
}

template<class T>
float CompactWeightedCollection<T>::GetSumWeight() const {
  return sum_weight_;
}

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_COMPACT_WEIGHTED_COLLECTION_H_
