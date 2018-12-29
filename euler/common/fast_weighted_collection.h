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

#ifndef EULER_COMMON_FAST_WEIGHTED_COLLECTION_H_
#define EULER_COMMON_FAST_WEIGHTED_COLLECTION_H_

#include <vector>
#include <utility>

#include "euler/common/weighted_collection.h"
#include "euler/common/alias_method.h"
namespace euler {
namespace common {

template<class T>
class FastWeightedCollection : public WeightedCollection<T> {
 public:
  void Init(const std::vector<T>& ids,
            const std::vector<float>& weights) override;

  void Init(const std::vector<std::pair<T, float>>& id_weight_pairs) override;

  std::pair<T, float> Sample() const override;

  size_t GetSize() const override;

  std::pair<T, float> Get(size_t idx) const override;

  float GetSumWeight() const override;

 private:
  std::vector<T> ids_;
  std::vector<float> weights_;
  AliasMethod alias_;
  float sum_weight_;
};

template<class T>
void FastWeightedCollection<T>::Init(const std::vector<T>& ids,
                                     const std::vector<float>& weights) {
  if (ids.size() != weights.size()) {
    return;
  }
  ids_.resize(ids.size());
  weights_.resize(weights.size());
  sum_weight_ = 0.0;
  for (size_t i = 0; i < weights.size(); i++) {
    sum_weight_ += weights[i];
    ids_[i] = ids[i];
    weights_[i] = weights[i];
  }
  std::vector<float> norm_weights(weights);
  for (size_t i = 0; i < norm_weights.size(); i++) {
    norm_weights[i] /= sum_weight_;
  }
  alias_.Init(norm_weights);
  return;
}

template<class T>
void FastWeightedCollection<T>::Init(
    const std::vector<std::pair<T, float>>& id_weight_pairs) {
  ids_.resize(id_weight_pairs.size());
  weights_.resize(id_weight_pairs.size());
  sum_weight_ = 0.0;
  for (size_t i = 0; i < id_weight_pairs.size(); i++) {
    sum_weight_ += id_weight_pairs[i].second;
    ids_[i] = id_weight_pairs[i].first;
    weights_[i] = id_weight_pairs[i].second;
  }
  std::vector<float> norm_weights(weights_);
  for (size_t i = 0; i < norm_weights.size(); i++) {
    norm_weights[i] /= sum_weight_;
  }
  alias_.Init(norm_weights);
  return;
}

template<class T>
std::pair<T, float> FastWeightedCollection<T>::Sample() const {
  int64_t column = alias_.Next();
  std::pair<T, float> id_weight_pair(ids_[column], weights_[column]);
  return id_weight_pair;
}

template<class T>
size_t FastWeightedCollection<T>::GetSize() const {
  return ids_.size();
}

template<class T>
std::pair<T, float> FastWeightedCollection<T>::Get(size_t idx) const {
  if (idx <= ids_.size()) {
    std::pair<T, float> id_weight_pair(ids_[idx], weights_[idx]);
    return id_weight_pair;
  } else {
    // LOG(ERROR) << "idx out of boundary";
    return std::pair<T, float>();
  }
}

template<class T>
float FastWeightedCollection<T>::GetSumWeight() const {
  return sum_weight_;
}
}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_FAST_WEIGHTED_COLLECTION_H_
