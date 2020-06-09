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

#ifndef EULER_CORE_INDEX_HASH_RANGE_SAMPLE_INDEX_H_
#define EULER_CORE_INDEX_HASH_RANGE_SAMPLE_INDEX_H_

#include <unordered_map>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <memory>

#include "euler/common/logging.h"
#include "euler/core/index/range_sample_index.h"
#include "euler/core/index/hash_index_result.h"
#include "euler/core/index/index_util.h"
#include "euler/common/fast_weighted_collection.h"
#include "euler/common/file_io.h"
#include "euler/common/bytes_io.h"
#include "euler/common/str_util.h"
#include "euler/common/bytes_compute.h"

namespace euler {

template<typename T1, typename T2>
class HashRangeSampleIndex : public SampleIndex {
 public:
  typedef T1 IdType;

  typedef T2 ValueType;

  typedef std::shared_ptr<RangeSampleIndex<IdType, ValueType>> RangeIndex;

 public:
  explicit HashRangeSampleIndex(const std::string& name) : SampleIndex(name) {}

  uint32_t SerializeSize() const;

  bool Serialize(FileIO* file_io) const;

  bool Deserialize(FileIO* file_io);

  bool Merge(std::shared_ptr<SampleIndex> hrIndex) override;

  bool Merge(const HashRangeSampleIndex<T1, T2>& hrIndex);

  bool Merge(const std::vector<std::shared_ptr<SampleIndex>>& hrIndex) override;

  std::shared_ptr<IndexResult> Search(
                      IndexSearchType op,
                      const std::string& value) const;

  std::shared_ptr<IndexResult>
  SearchAll() const override;

  bool AddItem(IdType id, RangeIndex index);


 private:
  std::unordered_map<IdType, RangeIndex> data_;
};

template<typename T1, typename T2>
uint32_t HashRangeSampleIndex<T1, T2>::SerializeSize() const {
  uint32_t total = sizeof(uint32_t);
  for (auto& it : data_) {
    total += BytesSize(it.first);
    total += it.second->SerializeSize();
  }
  return total;
}

template<typename T1, typename T2>
bool HashRangeSampleIndex<T1, T2>::Deserialize(FileIO* file_io) {
  data_.clear();
  while (!file_io->FileEnd()) {
    IdType id;
    if (!file_io->Read(&id)) {
      EULER_LOG(ERROR) << "read hash range sample index id error";
      return false;
    }
    RangeIndex rindex =
      RangeIndex(new RangeSampleIndex<T1, T2>(GetName() + "_" + ToString(id)));
    if (!rindex->Deserialize_ori(file_io)) {
      EULER_LOG(ERROR) << "read hash range sample index range value error";
      return false;
    }
    if (!data_.insert(std::make_pair(id, rindex)).second) {
      EULER_LOG(ERROR) << "insert range value error";
      return false;
    }
  }
  return true;
}

template<typename T1, typename T2>
bool HashRangeSampleIndex<T1, T2>::Serialize(FileIO* file_io) const {
  for (auto& it : data_) {
    if (!file_io->Append(it.first)) {
      EULER_LOG(ERROR) << "write id error";
      return false;
    }
    if (!it.second->Serialize(file_io)) {
      EULER_LOG(ERROR) << "write range index error";
      return false;
    }
  }
  return true;
}

template<typename T1, typename T2>
std::shared_ptr<IndexResult> HashRangeSampleIndex<T1, T2>::Search(
    IndexSearchType op,
    const std::string& value) const {
  const std::string DELIM = "::";
  auto p = value.find("::");
  if (p == std::string::npos) {
    EULER_LOG(ERROR) << "value format error";
    return std::shared_ptr<IndexResult>();
  }
  auto id = StringTo<IdType>(value.substr(0, p));
  auto it = data_.find(id);
  if (it == data_.end()) {
    return std::shared_ptr<IndexResult>();
  }
  return it->second->Search(op, value.substr(p + DELIM.size()));
}

template<typename T1, typename T2>
std::shared_ptr<IndexResult> HashRangeSampleIndex<T1, T2>::SearchAll() const {
  EULER_LOG(FATAL) << "search all not support";
  return std::shared_ptr<IndexResult>();
}

template<typename T1, typename T2>
bool HashRangeSampleIndex<T1, T2>::AddItem(IdType id, RangeIndex index) {
  if (!data_.insert(std::make_pair(id, index)).second) {
    EULER_LOG(ERROR) << "insert range index error";
    return false;
  }
  return true;
}

template<typename T1, typename T2>
bool HashRangeSampleIndex<T1, T2>::Merge(std::shared_ptr<SampleIndex> hrIndex) {
  auto index = dynamic_cast<HashRangeSampleIndex<T1, T2>*>(hrIndex.get());
  if (index == nullptr) {
    EULER_LOG(FATAL) << "convert to HashSampleIndex ptr error ";
    return false;
  }
  return Merge(*index);
}

template<typename T1, typename T2>
bool HashRangeSampleIndex<T1, T2>::Merge(
       const HashRangeSampleIndex<T1, T2>& hrIndex) {
  for (auto& it : hrIndex.data_) {
    auto iit = data_.find(it.first);
    if (iit != data_.end()) {
      iit->second->Merge(it.second);
    } else {
      data_.insert(it);
    }
  }
  return true;
}

template<typename T1, typename T2>
bool HashRangeSampleIndex<T1, T2>::Merge(
const std::vector<std::shared_ptr<SampleIndex>>& hrIndex) {
  for (auto& hr : hrIndex) {
    Merge(hr);
  }
  return true;
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_HASH_RANGE_SAMPLE_INDEX_H_
