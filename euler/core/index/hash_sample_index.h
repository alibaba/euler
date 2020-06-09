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

#ifndef EULER_CORE_INDEX_HASH_SAMPLE_INDEX_H_
#define EULER_CORE_INDEX_HASH_SAMPLE_INDEX_H_

#include <unordered_map>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <memory>

#include "euler/common/logging.h"
#include "euler/core/index/sample_index.h"
#include "euler/core/index/hash_index_result.h"
#include "euler/core/index/index_util.h"
#include "euler/common/fast_weighted_collection.h"
#include "euler/common/file_io.h"
#include "euler/common/bytes_io.h"
#include "euler/common/str_util.h"
#include "euler/common/bytes_compute.h"

namespace euler {

template<typename T1, typename T2>
class HashSampleIndex : public SampleIndex {
 public:
  typedef T1 IdType;

  typedef T2 ValueType;

  typedef euler::common::FastWeightedCollection<IdType> FWC;

 public:
  explicit HashSampleIndex(const std::string& name): SampleIndex(name) {}

  bool Serialize(FileIO* file_io) const;

  bool Deserialize(FileIO* file_io);

  uint32_t SerializeSize() const;

  std::shared_ptr<IndexResult>
  Search(IndexSearchType op, const std::string& values) const override;

  std::shared_ptr<IndexResult>
  SearchAll() const override;

  bool AddItem(ValueType value,
               const std::vector<IdType>& ids,
               const std::vector<float>& weights);

  bool AddItem(ValueType value,
               const std::vector<std::pair<IdType, float>>& idWeights);

  bool Merge(std::shared_ptr<SampleIndex> hIndex) override;

  bool Merge(const HashSampleIndex<T1, T2>& hIndex);

  bool Merge(const std::vector<std::shared_ptr<SampleIndex>>& hIndexs) override;

  std::vector<ValueType> GetKeys() const;

  ~HashSampleIndex() {}

 private:
  static bool PairCmp(std::pair<IdType, float> p1,
                      std::pair<IdType, float> p2)
  {return p1.first < p2.first;}

  static bool PairEQ(std::pair<IdType, float> p1,
                     std::pair<IdType, float> p2)
  {return p1.first == p2.first;}

  bool Check(IndexSearchType op) const;

 private:
  std::unordered_map<ValueType, std::shared_ptr<FWC> > data_;
};

template<typename T1, typename T2>
uint32_t HashSampleIndex<T1, T2>::SerializeSize() const {
  uint32_t total = sizeof(uint32_t);
  for (auto& it : data_) {
    total += BytesSize(it.first);
    total += BytesSize(it.second->GetIds());
    total += BytesSize(it.second->GetWeights());
  }
  return total;
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::Serialize(FileIO* file_io) const {
  for (auto& it : data_) {
    if (!file_io->Append(it.first)) {
      EULER_LOG(ERROR) << "write value error";
      return false;
    }
    const std::vector<IdType>& ids = it.second->GetIds();
    const std::vector<float>& weights = it.second->GetWeights();
    if (!file_io->Append(ids) || !file_io->Append(weights)) {
      EULER_LOG(ERROR) << "write ids weights error";
      return false;
    }
  }
  return true;
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::Deserialize(FileIO* file_io) {
  data_.clear();
  while (!file_io->FileEnd()) {
    ValueType value;
    if (!file_io->Read(&value)) {
      EULER_LOG(ERROR) << "read hash sample index value error";
      return false;
    }
    std::vector<IdType> ids;
    std::vector<float> weights;
    if (!file_io->Read(&ids)) {
      EULER_LOG(ERROR) << "read ids error";
      return false;
    }
    if (!file_io->Read(&weights)) {
      EULER_LOG(ERROR) << "read weights error";
      return false;
    }
    if (weights.size() != ids.size()) {
      EULER_LOG(ERROR) << "ids size not equal weights size";
      return false;
    }

    std::shared_ptr<FWC> fwcp(new FWC());
    fwcp->Init(ids, weights);
    data_[value] = fwcp;
  }
  return true;
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::AddItem(ValueType value,
                                      const std::vector<IdType>& ids,
                                      const std::vector<float>& weights) {
  auto it = data_.find(value);
  if (it != data_.end()) {
    EULER_LOG(ERROR) << value << " already in HashIndex";
    return false;
  }
  std::shared_ptr<FWC> fwcp(new FWC());
  fwcp->Init(ids, weights);
  data_[value] = fwcp;
  return true;
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::AddItem(
    ValueType value, const std::vector<std::pair<IdType, float>>& idWeights) {
  auto it = data_.find(value);
  if (it != data_.end()) {
    EULER_LOG(ERROR) << value << " already in HashIndex";
    return false;
  }
  std::shared_ptr<FWC> fwcp(new FWC());
  fwcp->Init(idWeights);
  data_[value] = fwcp;
  return true;
}

template<typename T1, typename T2>
std::shared_ptr<IndexResult> HashSampleIndex<T1, T2>::Search(
    IndexSearchType op,
    const std::string& value) const {
  if (!Check(op)) {
    return std::shared_ptr<IndexResult>();
  }
  std::unordered_map<ValueType, std::shared_ptr<FWC>> data;
  if (op == EQ) {
    auto v = euler::StringTo<ValueType>(value);
    auto it = data_.find(v);
    if (it != data_.end()) {
      data.insert(*it);
    }
  } else if (op == NOT_EQ) {
    auto v = euler::StringTo<ValueType>(value);
    for (auto& it : data_) {
      if (it.first != v) {
        data.insert(it);
      }
    }
  } else {
    const std::string DELIM = "::";
    auto vec = euler::Split(value, DELIM);
    std::vector<ValueType> values;
    for (auto& str : vec) {
      values.push_back(euler::StringTo<ValueType>(str));
    }
    std::sort(values.begin(), values.end());
    auto keys = GetKeys();
    std::sort(keys.begin(), keys.end());
    std::vector<ValueType> set_result;

    if (op == IN) {
      std::set_intersection(keys.begin(), keys.end(), values.begin(),
                            values.end(), std::back_inserter(set_result));
    } else {  // NOT_IN
      std::set_difference(keys.begin(), keys.end(), values.begin(),
                          values.end(), std::back_inserter(set_result));
    }

    for (auto& k : set_result) {
      auto it = data_.find(k);
      if (it != data_.end()) {
        data.insert(*it);
      }
    }
  }
  HashIndexResult<T1, T2>* h =
      new HashIndexResult<T1, T2>(this->GetName(), data);
  return std::shared_ptr<IndexResult>(h);
}

template<typename T1, typename T2>
std::shared_ptr<IndexResult> HashSampleIndex<T1, T2>::SearchAll() const {
  HashIndexResult<T1, T2>* h =
      new HashIndexResult<T1, T2>(this->GetName(), data_);
  return std::shared_ptr<IndexResult>(h);
}


template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::Merge(const HashSampleIndex<T1, T2>& hIndex) {
  for (auto& it : hIndex.data_) {
    auto iit = data_.find(it.first);
    if (iit != data_.end()) {
      std::vector<std::pair<IdType, float>> result;
      VecToPairVec(iit->second->GetIds(), iit->second->GetWeights(), &result);
      VecToPairVec(it.second->GetIds(), it.second->GetWeights(), &result);

      std::sort(result.begin(), result.end(), PairCmp);
      auto p = std::unique(result.begin(), result.end(), PairEQ);
      result.resize(std::distance(result.begin(), p));

      iit->second->Init(result);
    } else {
      data_.insert(it);
    }
  }
  return true;
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::Merge(std::shared_ptr<SampleIndex> hIndex) {
  auto index = dynamic_cast<HashSampleIndex<T1, T2>*>(hIndex.get());
  if (index == nullptr) {
    EULER_LOG(FATAL) << "convert to HashSampleIndex ptr error ";
    return false;
  }
  return Merge(*index);
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::
Merge(const std::vector<std::shared_ptr<SampleIndex>>& hIndexs) {
  std::unordered_map<ValueType, std::vector<std::shared_ptr<FWC>> > map;
  for (auto& it : data_) {
    std::vector<std::shared_ptr<FWC>> v;
    v.push_back(it.second);
    map[it.first] = v;
  }
  for (auto& sIndex : hIndexs) {
    auto index = dynamic_cast<HashSampleIndex<T1, T2>*>(sIndex.get());
    for (auto& it : index->data_) {
      auto iit = map.find(it.first);
      if (iit == map.end()) {
        std::vector<std::shared_ptr<FWC>> v;
        v.push_back(it.second);
        map[it.first] = v;
      } else {
        iit->second.push_back(it.second);
      }
    }
  }
  data_.clear();
  for (auto& it : map) {
    if (it.second.size() == 1) {
      data_[it.first] = (it.second)[0];
    } else {
      std::vector<std::pair<IdType, float>> result;
      for (auto& f : it.second) {
        VecToPairVec(f->GetIds(), f->GetWeights(), &result);
      }
      std::sort(result.begin(), result.end(), PairCmp);
      auto p = std::unique(result.begin(), result.end(), PairEQ);
      result.resize(std::distance(result.begin(), p));
      std::shared_ptr<FWC> fwcp(new FWC());
      fwcp->Init(result);
      data_[it.first] = fwcp;
    }
  }
  return true;
}

template<typename T1, typename T2>
std::vector<T2> HashSampleIndex<T1, T2>::GetKeys() const {
  std::vector<T2> result;
  result.reserve(data_.size());
  for (auto& it : data_) {
    result.push_back(it.first);
  }
  return result;
}

template<typename T1, typename T2>
bool HashSampleIndex<T1, T2>::Check(IndexSearchType op) const {
  if (op != EQ && op != NOT_EQ && op != IN && op != NOT_IN) {
    EULER_LOG(ERROR)
        << " HashSampleIndex only support EQ, NOT_EQ, IN, NOT_IN, operator";
    return false;
  }
  return true;
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_HASH_SAMPLE_INDEX_H_
