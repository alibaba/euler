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

#ifndef EULER_CORE_INDEX_INDEX_META_H_
#define EULER_CORE_INDEX_INDEX_META_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "euler/core/index/index_types.h"
#include "euler/common/file_io.h"

namespace euler {

struct IndexMetaRecord {
  IndexType type;
  IndexDataType idType;
  IndexDataType valueType;
};

class IndexMeta {
 public:
  IndexMeta() {}

  uint32_t SerializeSize() const;

  bool Serialize(std::string* s) const;

  bool Deserialize(const char* buffer, size_t size);

  bool Serialize(FileIO* file_io) const;

  bool Deserialize(FileIO* file_io);

  bool AddMeta(const std::string& columnName, IndexMetaRecord record);

  bool HasIndex(const std::string& columnName) const;

  IndexMetaRecord GetMetaRecord(const std::string& columnName) const;

  uint32_t GetIndexNum() const;

  ~IndexMeta() {}

  std::vector<std::string> IndexFields() const {
    std::vector<std::string> fields;
    fields.reserve(meta_.size());
    for (auto& it : meta_) {
      fields.emplace_back(it.first);
    }

    return fields;
  }

 private:
  std::unordered_map<std::string, IndexMetaRecord> meta_;
};

}  // namespace euler

#endif  // EULER_CORE_INDEX_INDEX_META_H_
