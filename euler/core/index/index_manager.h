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

#ifndef EULER_CORE_INDEX_INDEX_MANAGER_H_
#define EULER_CORE_INDEX_INDEX_MANAGER_H_

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>

#include "euler/core/index/index_meta.h"
#include "euler/core/index/hash_sample_index.h"
#include "euler/core/index/range_sample_index.h"
#include "euler/core/index/hash_range_sample_index.h"
#include "euler/common/file_io.h"
#include "euler/common/slice.h"
#include "euler/common/server_monitor.h"

namespace euler {

class IndexManager {
 public:
  IndexManager(): shard_index_(0), shard_number_(1) { }

  std::shared_ptr<SampleIndex> ReadIndex(Slice path, const std::string& name);

  bool DeserializeMeta(FileIO* file_io, const std::string& index_name);

  Status Deserialize(Slice path);

  std::vector<Meta> GetIndexInfo();

  std::shared_ptr<SampleIndex> GetIndex(const std::string& key) const;

  std::vector<std::string> GetKeys() const;

  bool HasIndex(const std::string& key) const;

  const IndexMeta& GetMeta() const;

  static IndexManager& Instance() {
    static IndexManager instance;
    return instance;
  }

  void set_shard_index(int shard_index) {
    shard_index_ = shard_index;
  }

  void set_shard_number(int shard_number) {
    shard_number_ = shard_number;
  }

 private:
  template<typename IdType>
  std::shared_ptr<SampleIndex>
       GetHashIndex(const std::string& key, IndexDataType valueType) const;

  template<typename IdType>
  std::shared_ptr<SampleIndex>
       GetRangeIndex(const std::string& key, IndexDataType valueType) const;

  template<typename IdType>
  std::shared_ptr<SampleIndex>
       GetHashRangeIndex(const std::string& key, IndexDataType valueType) const;

  std::shared_ptr<SampleIndex> GetIndexByType(const std::string& key,
      IndexType t, IndexDataType idType, IndexDataType valueType) const;

 private:
  int shard_index_;
  int shard_number_;

  IndexMeta meta_;
  std::unordered_map<std::string, std::shared_ptr<SampleIndex> > index_;
};

template<typename IdType>
std::shared_ptr<SampleIndex>
IndexManager::GetHashIndex(const std::string& key,
                           IndexDataType valueType) const {
  SampleIndex* h = NULL;
  if (valueType == kFloat) {
    h = new HashSampleIndex<IdType, float>(key);
  } else if (valueType == kUInt32) {
    h = new HashSampleIndex<IdType, uint32_t>(key);
  } else if (valueType == kInt32) {
    h = new HashSampleIndex<IdType, int32_t>(key);
  } else if (valueType == kUInt64) {
    h = new HashSampleIndex<IdType, uint64_t>(key);
  } else if (valueType == kInt64) {
    h = new HashSampleIndex<IdType, int64_t>(key);
  } else if (valueType == kString) {
    h = new HashSampleIndex<IdType, std::string>(key);
  } else {
    EULER_LOG(FATAL) << "hash index not only support this value type "
                     << valueType;
  }
  return std::shared_ptr<SampleIndex>(h);
}

template<typename IdType>
std::shared_ptr<SampleIndex>
IndexManager::GetRangeIndex(const std::string& key,
                            IndexDataType valueType) const {
  SampleIndex* h = NULL;
  if (valueType == kFloat) {
    h = new RangeSampleIndex<IdType, float>(key);
  } else if (valueType == kUInt32) {
    h = new RangeSampleIndex<IdType, uint32_t>(key);
  } else if (valueType == kInt32) {
    h = new RangeSampleIndex<IdType, int32_t>(key);
  } else if (valueType == kUInt64) {
    h = new RangeSampleIndex<IdType, uint64_t>(key);
  } else if (valueType == kInt64) {
    h = new RangeSampleIndex<IdType, int64_t>(key);
  } else if (valueType == kString) {
    h = new RangeSampleIndex<IdType, std::string>(key);
  } else {
    EULER_LOG(FATAL) << "range index not only support this value type "
                     << valueType;
  }
  return std::shared_ptr<SampleIndex>(h);
}

template<typename IdType>
std::shared_ptr<SampleIndex>
IndexManager::GetHashRangeIndex(const std::string& key,
                            IndexDataType valueType) const {
  SampleIndex* h = NULL;
  if (valueType == kFloat) {
    h = new HashRangeSampleIndex<IdType, float>(key);
  } else if (valueType == kUInt32) {
    h = new HashRangeSampleIndex<IdType, uint32_t>(key);
  } else if (valueType == kInt32) {
    h = new HashRangeSampleIndex<IdType, int32_t>(key);
  } else if (valueType == kUInt64) {
    h = new HashRangeSampleIndex<IdType, uint64_t>(key);
  } else if (valueType == kInt64) {
    h = new HashRangeSampleIndex<IdType, int64_t>(key);
  } else if (valueType == kString) {
    h = new HashRangeSampleIndex<IdType, std::string>(key);
  } else {
    EULER_LOG(FATAL) << "hashrange index not only support this value type "
                     << valueType;
  }
  return std::shared_ptr<SampleIndex>(h);
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_INDEX_MANAGER_H_
