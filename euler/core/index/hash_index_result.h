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

#ifndef EULER_CORE_INDEX_HASH_INDEX_RESULT_H_
#define EULER_CORE_INDEX_HASH_INDEX_RESULT_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>

#include "euler/common/logging.h"
#include "euler/core/index/index_result.h"
#include "euler/core/index/common_index_result.h"
#include "euler/common/fast_weighted_collection.h"

namespace euler {

template<typename IdType, typename ValueType>
class HashIndexResult : public IndexResult {
 public:
  typedef euler::common::FastWeightedCollection<IdType> FWC;

  explicit HashIndexResult(const std::string& name):
      IndexResult(HASHINDEXRESULT, name) {}

  HashIndexResult(const std::string& name,
                  const std::unordered_map<ValueType,
                  std::shared_ptr<FWC>>& data):
      IndexResult(HASHINDEXRESULT, name), data_(data) {}

  std::vector<uint64_t> GetIds() const override;

  std::vector<float> GetWeights() const override;

  std::vector<std::pair<uint64_t, float>> Sample(size_t count) const override;

  std::shared_ptr<IndexResult>
  Intersection(std::shared_ptr<IndexResult> indexResult) override;

  std::shared_ptr<IndexResult>
  Union(std::shared_ptr<IndexResult> indexResult) override;

  std::shared_ptr<IndexResult> ToCommonIndexResult() override;

  float SumWeight() const override;

  size_t size() const override { return data_.size(); }

  ~HashIndexResult() {}

 private:
  std::unordered_map<ValueType, std::shared_ptr<FWC> > data_;
};

template<typename T1, typename T2>
std::vector<uint64_t> HashIndexResult<T1, T2>::GetIds() const {
  std::vector<uint64_t> result;
  for (auto& it : data_) {
    std::copy(it.second->GetIds().begin(),
              it.second->GetIds().end(), back_inserter(result));
  }
  return result;
}

template<typename T1, typename T2>
std::vector<float> HashIndexResult<T1, T2>::GetWeights() const {
  std::vector<float> result;
  for (auto& it : data_) {
    size_t start = result.size() + 1;
    std::copy(it.second->GetWeights().begin(),
              it.second->GetWeights().end(), back_inserter(result));
    size_t end = result.size();
    for (size_t j = end; j > start; --j) {
      result[j - 1] -= result[j -2];
    }
  }
  return result;
}

template<typename IdType, typename ValueType>
std::vector<std::pair<uint64_t, float>>
HashIndexResult<IdType, ValueType>::Sample(size_t count) const {
  std::vector<std::pair<uint64_t, float>> result;
  if (data_.size() == 0) {
    return result;
  } else if (data_.size() == 1) {
    auto it = data_.begin();
    result.resize(count);
    for (auto& i : result) {
      auto s = it->second->Sample();
      i = std::make_pair(static_cast<uint64_t>(s.first), s.second);
    }
  } else {
    std::vector<ValueType> ids;
    std::vector<float> weights;

    for (auto& it : data_) {
      ids.push_back(it.first);
      weights.push_back(it.second->GetSumWeight());
    }
    euler::common::FastWeightedCollection<ValueType> fwc;
    fwc.Init(ids, weights);
    result.resize(count);
    for (auto& i : result) {
      auto id = fwc.Sample();
      auto iit = data_.find(id.first);
      auto s = iit->second->Sample();
      i = std::make_pair(static_cast<uint64_t>(s.first), s.second);
    }
  }
  return result;
}

template<typename IdType, typename ValueType>
std::shared_ptr<IndexResult> HashIndexResult<IdType, ValueType>::
Intersection(std::shared_ptr<IndexResult> indexResult) {
  if (this->GetName() != indexResult->GetName()) {
    auto cIndexResult = ToCommonIndexResult();
    return cIndexResult->Intersection(indexResult);
  }

  HashIndexResult<IdType, ValueType>* hIndexResult =
      dynamic_cast<HashIndexResult<IdType, ValueType>*>(indexResult.get());
  if (hIndexResult == NULL) {
    EULER_LOG(FATAL) << "HashIndexResult convert to HashIndexResult ptr error ";
  }
  HashIndexResult<IdType, ValueType>* result =
      new HashIndexResult<IdType, ValueType>(this->GetName());

  for (auto& it : hIndexResult->data_) {
    auto iit = data_.find(it.first);
    if (iit != data_.end()) {
      auto r = result->data_.insert(*iit);
      if (!r.second) {
        return std::shared_ptr<IndexResult>(nullptr);
      }
    }
  }

  return std::shared_ptr<IndexResult>(result);
}

template<typename IdType, typename ValueType>
std::shared_ptr<IndexResult> HashIndexResult<IdType, ValueType>::
Union(std::shared_ptr<IndexResult> indexResult) {
  if (this->GetName() != indexResult->GetName()) {
    auto cIndexResult = ToCommonIndexResult();
    return cIndexResult->Union(indexResult);
  }

  HashIndexResult<IdType, ValueType>* hIndexResult =
      dynamic_cast<HashIndexResult<IdType, ValueType>*>(indexResult.get());
  if (hIndexResult == nullptr) {
    EULER_LOG(FATAL) << "HashIndexResult convert to HashIndexResult ptr error ";
  }

  HashIndexResult<IdType, ValueType>* result =
      new HashIndexResult<IdType, ValueType>(this->GetName());
  result->data_ = data_;

  for (auto& it : hIndexResult->data_) {
    auto iit = data_.find(it.first);
    if (iit == data_.end()) {
      auto r = result->data_.insert(it);
      if (!r.second) {
        return std::shared_ptr<IndexResult>(nullptr);
      }
    }
  }
  return std::shared_ptr<IndexResult>(result);
}

template<typename IdType, typename ValueType>
std::shared_ptr<IndexResult>
HashIndexResult<IdType, ValueType>::ToCommonIndexResult() {
  std::vector<std::pair<uint64_t, float>> data;
  for (auto& i : data_) {
    auto ids = i.second->GetIds();
    auto weights = i.second->GetWeights();
    for (size_t j = 0; j < ids.size(); ++j) {
      data.push_back(std::make_pair(ids[j], weights[j]));
    }
  }
  auto f = [](const std::pair<uint64_t, float>& a,
              const std::pair<uint64_t, float>& b) {return a.first < b.first;};
  std::sort(data.begin(), data.end(), f);
  CommonIndexResult* commonResult = new CommonIndexResult("common", data);
  return std::shared_ptr<IndexResult>(commonResult);
}

template<typename IdType, typename ValueType>
float HashIndexResult<IdType, ValueType>::SumWeight() const {
  float sum = 0;
  for (auto& it : data_) {
    sum += it.second->GetSumWeight();
  }
  return sum;
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_HASH_INDEX_RESULT_H_
