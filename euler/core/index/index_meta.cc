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

#include "euler/core/index/index_meta.h"

#include<iostream>
#include<utility>

#include "euler/core/index/index_util.h"
#include "euler/common/bytes_compute.h"
#include "euler/common/bytes_io.h"

namespace euler {

uint32_t IndexMeta::SerializeSize() const {
  uint32_t len = meta_.size();
  uint32_t total = sizeof(len);
  for (auto& it : meta_) {
    total += BytesSize(it.first);
    total += BytesSize(it.second);
  }
  return total;
}

bool IndexMeta::Serialize(std::string* s) const {
  uint32_t size = SerializeSize();
  s->resize(size);
  BytesWriter bytes_writer;

  uint32_t len = meta_.size();
  if (!bytes_writer.Write(len)) {
    EULER_LOG(ERROR) << "write meta size error";
    return false;
  }

  for (auto& it : meta_) {
    if (!bytes_writer.Write(it.first) || !bytes_writer.Write(it.second)) {
      EULER_LOG(ERROR) << "write meta info error";
      return false;
    }
  }

  *s = bytes_writer.data();
  return true;
}

bool IndexMeta::Deserialize(const char* buffer, size_t size) {
  BytesReader bytes_reader(buffer, size);
  uint32_t len = 0;
  if (!bytes_reader.Read(&len)) {
    EULER_LOG(ERROR) << "read meta size error";
    return false;
  }
  meta_.clear();
  for (uint32_t i = 0; i < len; ++i) {
    std::string s;
    IndexMetaRecord record;
    if (!bytes_reader.Read(&s) || !bytes_reader.Read(&record)) {
      EULER_LOG(ERROR) << "read meta info error";
      return false;
    }

    if (!AddMeta(s, record)) {
      return false;
    }
  }
  return true;
}

bool IndexMeta::Serialize(FileIO* file_io) const {
  uint32_t len = meta_.size();
  if (!file_io->Append(len)) {
    EULER_LOG(ERROR) << "write meta size " << len << " error";
    return false;
  }

  for (auto& it : meta_) {
    if (!file_io->Append(it.first) || !file_io->Append(it.second)) {
      EULER_LOG(ERROR) << "write meta key "<< it.first << " error";
      return false;
    }
  }
  return true;
}

bool IndexMeta::Deserialize(FileIO* file_io) {
  uint32_t size = 0;
  if (!file_io->Read(&size)) {
    EULER_LOG(ERROR) << "read meta size error";
    return false;
  }
  meta_.clear();
  for (uint32_t i = 0; i < size; ++i) {
    std::string s;
    if (!file_io->Read(&s)) {
      EULER_LOG(ERROR) << "read meta key error";
      return false;
    }
    IndexMetaRecord record;
    if (!file_io->Read(&record)) {
      EULER_LOG(ERROR) << "read meta record error";
      return false;
    }
    if (!AddMeta(s, record)) {
      return false;
    }
  }
  return true;
}

bool IndexMeta::AddMeta(const std::string& columnName, IndexMetaRecord record) {
  auto result = meta_.insert(std::make_pair(columnName, record));
  return result.second;
}

bool IndexMeta::HasIndex(const std::string& columnName) const {
  auto it = meta_.find(columnName);
  return it != meta_.end();
}

IndexMetaRecord IndexMeta::GetMetaRecord(const std::string& columnName) const {
  auto it = meta_.find(columnName);
  if (it != meta_.end()) {
    return it->second;
  }
  return IndexMetaRecord();
}

uint32_t IndexMeta::GetIndexNum() const {
  return meta_.size();
}

}  // namespace euler
