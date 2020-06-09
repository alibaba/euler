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

#ifndef EULER_CORE_INDEX_RANGE_SAMPLE_INDEX_H_
#define EULER_CORE_INDEX_RANGE_SAMPLE_INDEX_H_

#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <string>

#include "euler/common/logging.h"
#include "euler/core/index/sample_index.h"
#include "euler/core/index/index_util.h"
#include "euler/core/index/range_index_result.h"
#include "euler/common/file_io.h"
#include "euler/common/bytes_io.h"
#include "euler/common/bytes_compute.h"

namespace euler {

template<typename T1, typename T2>
class RangeSampleIndex : public SampleIndex {
 public:
  typedef T1 IdType;

  typedef T2 ValueType;

  typedef typename std::vector<T1>::const_iterator ResultPos;

  typedef std::pair<ResultPos, ResultPos> ResultRange;

 public:
  explicit RangeSampleIndex(const std::string& name) : SampleIndex(name) {}

  uint32_t SerializeSize() const;

  bool Serialize(FileIO* file_io) const override;

  bool Deserialize(FileIO* file_io) override;

  bool Deserialize_ori(FileIO* file_io);

  // values must sorted
  bool Init(const std::vector<IdType>& ids,
            const std::vector<ValueType>& values,
            const std::vector<float>& weights);

  ~RangeSampleIndex() {}

  std::shared_ptr<IndexResult> SearchAll() const override {
    std::vector<ResultRange> r = {std::make_pair(ids_.begin(), ids_.end())};
    RangeIndexResult<T1, T2>* h = new RangeIndexResult<T1, T2>(this->GetName());
    h->Init(ids_.begin(), ids_.end(), values_.begin(), sum_weights_.begin(), r);
    return std::shared_ptr<IndexResult> (h);
  }

  std::shared_ptr<IndexResult>
  Search(IndexSearchType op, const std::string& value) const override {
    std::vector<ResultRange> r;
    if (op == IN) {
      return SearchIN(value);
    } else if (op == NOT_IN) {
      return SearchNOTIN(value);
    } else {
      ValueType v = euler::StringTo<ValueType>(value);
      if (op == LESS) {
        r = SearchLess(v);
      } else if (op == LESS_EQ) {
        r = SearchLessEqual(v);
      } else if (op == GREATER) {
        r = SearchMore(v);
      } else if (op == GREATER_EQ) {
        r = SearchMoreEqual(v);
      } else if (op == EQ) {
        r = SearchEqual(v);
      } else if (op == NOT_EQ) {
        r = SearchNotEqual(v);
      } else {
        return std::shared_ptr<IndexResult>();
      }
      RangeIndexResult<T1, T2>* h =
          new RangeIndexResult<T1, T2>(this->GetName());
      h->Init(ids_.begin(), ids_.end(),
              values_.begin(), sum_weights_.begin(), r);
      return std::shared_ptr<IndexResult> (h);
    }
    return std::shared_ptr<IndexResult>();
  }

 private:
  std::vector<ResultRange> SearchLess(const ValueType& value) const {
    auto p = std::lower_bound(values_.begin(), values_.end(), value);
    auto diff = p - values_.begin();
    if (diff > 0) {
        std::vector<ResultRange> v =
            {std::make_pair(ids_.begin(), ids_.begin() + diff)};
        return v;
    }
    return std::vector<ResultRange>();
  }

  std::vector<ResultRange> SearchLessEqual(const ValueType& value) const {
    auto p = std::upper_bound(values_.begin(), values_.end(), value);
    auto diff = p - values_.begin();
    if (diff > 0) {
      std::vector<ResultRange> v =
          {std::make_pair(ids_.begin(), ids_.begin() + diff)};
      return v;
    }
    return std::vector<ResultRange>();
  }

  std::vector<ResultRange> SearchMore(const ValueType& value) const {
    auto p = std::upper_bound(values_.begin(), values_.end(), value);
    if (p != values_.end()) {
      auto diff = p - values_.begin();
      std::vector<ResultRange> v =
          {std::make_pair(ids_.begin() + diff, ids_.end())};
      return v;
    }
    return std::vector<ResultRange>();
  }

  std::vector<ResultRange> SearchMoreEqual(const ValueType& value) const {
    auto p = std::lower_bound(values_.begin(), values_.end(), value);
    if (p != values_.end()) {
      auto diff = p - values_.begin();
      std::vector<ResultRange> v =
          {std::make_pair(ids_.begin() + diff, ids_.end())};
      return v;
    }
    return std::vector<ResultRange>();
  }

  std::vector<ResultRange> SearchEqual(const ValueType& value) const {
    auto p = std::equal_range(values_.begin(), values_.end(), value);
    if (p.first != p.second) {
      auto begin = p.first - values_.begin();
      auto end = p.second - values_.begin();
      std::vector<ResultRange> v =
          {std::make_pair(ids_.begin() + begin, ids_.begin() + end)};
      return v;
    }
    return std::vector<ResultRange>();
  }

  std::vector<ResultRange> SearchNotEqual(const ValueType& value) const {
    auto p = std::equal_range(values_.begin(), values_.end(), value);
    if (p.first + values_.size() != p.second) {
      auto begin = p.first - values_.begin();
      auto end = p.second - values_.begin();
      std::vector<ResultRange> result;
      result.push_back(std::make_pair(ids_.begin(), ids_.begin() + begin));
      result.push_back(std::make_pair(ids_.begin() + end, ids_.end()));
      return result;
    }
    return std::vector<ResultRange>();
  }

  std::shared_ptr<IndexResult> SearchIN(const std::string& value) const {
    const std::string DELIM = "::";
    auto vec = euler::Split(value, DELIM);
    if (vec.size() >= 1) {
      auto result = Search(EQ, vec[0]);
      for (size_t i = 1; i < vec.size(); ++i) {
        result = result->Union(Search(EQ, vec[i]));
      }
      return result;
    }
    return std::shared_ptr<IndexResult>();
  }

  std::shared_ptr<IndexResult> SearchNOTIN(const std::string& value) const {
    const std::string DELIM = "::";
    auto vec = euler::Split(value, DELIM);
    if (vec.size() >= 1) {
      auto result = Search(NOT_EQ, vec[0]);
      for (size_t i = 1; i < vec.size(); ++i) {
        result = result->Intersection(Search(NOT_EQ, vec[i]));
      }
      return result;
    }
    return std::shared_ptr<IndexResult>();
  }

 public:
  bool Merge(std::shared_ptr<SampleIndex> hIndex) override {
    auto index = dynamic_cast<RangeSampleIndex*>(hIndex.get());
    if (index == nullptr) {
      EULER_LOG(FATAL) << "convert to HashSampleIndex ptr error ";
      return false;
    }
    return Merge(*index);
  }

  bool Merge(const RangeSampleIndex& r) {
    std::vector<Pair> p;
    VecToPairVec(ids_, values_, sum_weights_, &p);
    VecToPairVec(r.ids_, r.values_, r.sum_weights_, &p);
    std::sort(p.begin(), p.end());
    ids_.resize(p.size());
    values_.resize(p.size());
    sum_weights_.resize(p.size());
    float total_weight = 0;
    for (size_t i = 0; i < p.size(); ++i) {
      ids_[i] = p[i].id_;
      values_[i] = p[i].value_;
      total_weight += p[i].weight_;
      sum_weights_[i] = total_weight;
    }
    return true;
  }

  bool Merge(const std::vector<std::shared_ptr<SampleIndex>>& r) override {
    std::vector<Pair> p;
    VecToPairVec(ids_, values_, sum_weights_, &p);
    for (auto it : r) {
      auto iit = dynamic_cast<RangeSampleIndex*>(it.get());
      VecToPairVec(iit->ids_, iit->values_, iit->sum_weights_, &p);
    }
    std::sort(p.begin(), p.end());
    ids_.resize(p.size());
    values_.resize(p.size());
    sum_weights_.resize(p.size());
    float total_weight = 0;
    for (size_t i = 0; i < p.size(); ++i) {
      ids_[i] = p[i].id_;
      values_[i] = p[i].value_;
      total_weight += p[i].weight_;
      sum_weights_[i] = total_weight;
    }
    return true;
  }


 private:
  struct Pair {
    Pair(IdType id, ValueType value, float weight):
        id_(id), value_(value), weight_(weight) {}

    IdType id_;
    ValueType value_;
    float weight_;

    bool operator<(const Pair& r) {
      return value_ < r.value_;
    }
  };

  void VecToPairVec(const std::vector<IdType>& ids,
                    const std::vector<ValueType>& values,
                    const std::vector<float>& sum_weights,
                    std::vector<Pair>* pair) const {
    for (size_t i = 0; i < ids.size(); ++i) {
      float weight = 0;
      if (i == 0) {
        weight = sum_weights[0];
      } else {
        weight = sum_weights[i] - sum_weights[i-1];
      }
      Pair p(ids[i], values[i], weight);
      pair->push_back(p);
    }
  }

 private:
  std::vector<IdType> ids_;
  std::vector<ValueType> values_;
  std::vector<float> sum_weights_;
};

template<typename T1, typename T2>
uint32_t RangeSampleIndex<T1, T2>::SerializeSize() const {
  uint32_t total = BytesSize(ids_);
  total += BytesSize(values_);
  total += BytesSize(sum_weights_);
  return total;
}

template<typename T1, typename T2>
bool RangeSampleIndex<T1, T2>::Serialize(FileIO* file_io) const {
  if (!file_io->Append(ids_)) {
    EULER_LOG(ERROR) << "write ids error";
    return false;
  }
  if (!file_io->Append(values_)) {
    EULER_LOG(ERROR) << "write values error";
    return false;
  }
  std::vector<float> weights;
  weights.reserve(sum_weights_.size());
  std::adjacent_difference(sum_weights_.begin(),
                           sum_weights_.end(), back_inserter(weights));
  if (!file_io->Append(weights)) {
    EULER_LOG(ERROR) << "write sum weights error";
    return false;
  }
  return true;
}

template<typename T1, typename T2>
bool RangeSampleIndex<T1, T2>::Deserialize(FileIO* file_io) {
  ids_.clear();
  values_.clear();
  sum_weights_.clear();

  std::vector<Pair> pair;
  while (!file_io->FileEnd()) {
    std::vector<IdType> id;
    std::vector<ValueType> value;
    std::vector<float> weight;

    if (!file_io->Read(&id)) {
      EULER_LOG(ERROR) << "read ids error";
      return false;
    }
    if (!file_io->Read(&value)) {
      EULER_LOG(ERROR) << "read values error";
      return false;
    }
    if (!file_io->Read(&weight)) {
      EULER_LOG(ERROR) << "read sum weights error";
      return false;
    }
    if (id.size() != value.size() || id.size() != weight.size()) {
      EULER_LOG(ERROR) << "id, value, weight size not equal";
      return false;
    }
    for (size_t i = 0; i < id.size(); ++i) {
      Pair p(id[i], value[i], weight[i]);
      pair.push_back(p);
    }
  }

  std::sort(pair.begin(), pair.end());
  float cum_weight = 0;
  ids_.resize(pair.size());
  values_.resize(pair.size());
  sum_weights_.resize(pair.size());
  for (size_t i = 0; i < pair.size(); ++i) {
    ids_[i] = pair[i].id_;
    values_[i] = pair[i].value_;
    cum_weight += pair[i].weight_;
    sum_weights_[i] = cum_weight;
  }
  return true;
}

template<typename T1, typename T2>
bool RangeSampleIndex<T1, T2>::Deserialize_ori(FileIO* file_io) {
  ids_.clear();
  values_.clear();
  sum_weights_.clear();

  if (!file_io->Read(&ids_)) {
    EULER_LOG(ERROR) << "read ids error";
    return false;
  }
  if (!file_io->Read(&values_)) {
    EULER_LOG(ERROR) << "read values error";
    return false;
  }
  if (!file_io->Read(&sum_weights_)) {
    EULER_LOG(ERROR) << "read weights error";
    return false;
  }
  if (ids_.size() != values_.size() || ids_.size() != sum_weights_.size()) {
    EULER_LOG(ERROR) << "id, value, weight size not equal";
    return false;
  }

  float cum_weight = 0;
  for (size_t i = 0; i < sum_weights_.size(); ++i) {
    cum_weight += sum_weights_[i];
    sum_weights_[i] = cum_weight;
  }
  return true;
}

// values must sorted
template<typename T1, typename T2>
bool RangeSampleIndex<T1, T2>::Init(const std::vector<IdType>& ids,
                                    const std::vector<ValueType>& values,
                                    const std::vector<float>& weights) {
  if (ids.size() == values.size() && values.size() == weights.size()) {
    float sum_weight = 0.0;
    ids_.resize(ids.size());
    values_.resize(values.size());
    sum_weights_.resize(weights.size());
    std::copy(ids.begin(), ids.end(), ids_.begin());
    std::copy(values.begin(), values.end(), values_.begin());
    auto f = [&sum_weight](float w){sum_weight += w; return sum_weight; };
    std::transform(weights.begin(), weights.end(), sum_weights_.begin(), f);
    return true;
  } else {
    EULER_LOG(ERROR) << "ids values weights size not equal, init error ";
    return false;
  }
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_RANGE_SAMPLE_INDEX_H_
