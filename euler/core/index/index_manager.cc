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

#include "euler/core/index/index_manager.h"

#include <string>

#include "euler/common/bytes_compute.h"
#include "euler/common/env.h"
#include "euler/common/logging.h"
#include "euler/common/server_register.h"

namespace euler {

bool IndexManager::DeserializeMeta(FileIO* file_io,
                                   const std::string& name) {
  IndexMetaRecord record;
  if (!file_io->Read(&record.type)) {
    EULER_LOG(ERROR) << "read type error";
    return false;
  }
  if (!file_io->Read(&record.idType)) {
    EULER_LOG(ERROR) << "read idtype error";
    return false;
  }
  if (!file_io->Read(&record.valueType)) {
    EULER_LOG(ERROR) << "read valuetype error";
    return false;
  }
  meta_.AddMeta(name, record);
  return true;
}

std::vector<Meta> IndexManager::GetIndexInfo() {
  std::vector<std::string> index_info;
  for (auto it = index_.begin(); it != index_.end(); ++it) {
    std::string name = it->first;
    auto record = meta_.GetMetaRecord(name);
    if (record.type == HASHINDEX) {
      index_info.push_back(ToString(name, ":", "hash_index"));
    } else if (record.type == RANGEINDEX) {
      index_info.push_back(ToString(name, ":", "range_index"));
    } else if (record.type == HASHRANGEINDEX) {
      index_info.push_back(ToString(name, ":", "hash_range_index"));
    }
  }

  std::vector<Meta> meta_list(2);
  meta_list[0]["index_info"] = Join(index_info, ",");
  return meta_list;
}

std::shared_ptr<SampleIndex>
IndexManager::ReadIndex(Slice path, const std::string& index_name) {
  std::unique_ptr<FileIO> dir_reader;
  std::string new_path = path.ToString() + "/" + index_name;
  Env::Default()->NewFileIO(new_path, true, &dir_reader);
  if (!dir_reader->IsDirectory()) {
    EULER_LOG(ERROR) << "Invalid data path: " << path;
    return std::shared_ptr<SampleIndex>();;
  }

  auto meta_reader = dir_reader->Open("meta", true);
  if (!DeserializeMeta(meta_reader.get(), index_name)) {
    EULER_LOG(ERROR) << "deserialize " << index_name << " meta error";
    return std::shared_ptr<SampleIndex>();;
  }

  auto filter = [this] (const std::string& filename) {
    auto vec = Split(filename, "_.");
    if (vec.size() == 3 &&
        atoi(vec[1].c_str()) % shard_number_ == shard_index_ &&
        vec[2] == "dat") {
      return true;
    }
    return false;
  };

  auto record = meta_.GetMetaRecord(index_name);
  std::vector<std::shared_ptr<SampleIndex>> result;
  auto files = dir_reader->ListDirectory(filter);
  for (auto& file : files) {
    EULER_LOG(INFO) << " load file " << file;
    auto reader = dir_reader->Open(file, true);
    auto i = GetIndexByType(index_name, record.type,
                     record.idType, record.valueType);
    if (!i->Deserialize(reader.get())) {
      EULER_LOG(ERROR) << index_name <<" index deserialize fail";
      return std::shared_ptr<SampleIndex>();
    }
    result.push_back(i);
  }

  if (result.size() == 1) {
    return result[0];
  }

  auto index = GetIndexByType(index_name, record.type,
                   record.idType, record.valueType);
  if (!index->Merge(result)) {
    EULER_LOG(ERROR) << "merge " << index_name << " index failed!";
    return std::shared_ptr<SampleIndex>();
  }
  EULER_LOG(INFO) << "Deserialize IndexManager "
                  << path << " successfully!";
  return index;
}

Status IndexManager::Deserialize(Slice path) {
  std::unique_ptr<FileIO> dir_reader;
  RETURN_IF_ERROR(Env::Default()->NewFileIO(path, true, &dir_reader));
  if (!dir_reader->IsDirectory()) {
    EULER_LOG(ERROR) << "Invalid data path: " << path;
    return Status::Internal("Invalid path: ", path);
  }

  auto default_filter = [](const std::string& ){return true;};
  auto index_names = dir_reader->ListDirectory(default_filter);

  for (auto& name : index_names) {
    auto index = ReadIndex(path, name);
    if (index == nullptr) {
      return Status::Internal("invalid index: ", name);
    }
    index_[name] = index;
  }
  EULER_LOG(INFO) << "load index success ";
  EULER_LOG(INFO) <<"Index names are as follows:";
  for (auto key : GetKeys()) {
    EULER_LOG(INFO) << "name:" << key;
  }
  return Status::OK();
}

bool IndexManager::HasIndex(const std::string& key) const {
  auto it = index_.find(key);
  if (it == index_.end()) {
    return false;
  }
  return true;
}

std::shared_ptr<SampleIndex>
IndexManager::GetIndex(const std::string& key) const {
  auto it = index_.find(key);
  if (it == index_.end()) {
    EULER_LOG(ERROR) << "not find  " << key << " index";
    return nullptr;
  }
  return it->second;
}

std::vector<std::string> IndexManager::GetKeys() const {
  std::vector<std::string> v;
  for (auto& it : index_) {
    v.push_back(it.first);
  }
  return v;
}

const IndexMeta& IndexManager::GetMeta() const {
  return meta_;
}

std::shared_ptr<SampleIndex>
IndexManager::GetIndexByType(const std::string& key, IndexType t,
                 IndexDataType idType, IndexDataType valueType) const {
  if (t == HASHINDEX) {
    if (idType == kUInt32) {
      return GetHashIndex<uint32_t>(key, valueType);
    } else {
      return GetHashIndex<uint64_t>(key, valueType);
    }
  } else if (t == RANGEINDEX) {
    if (idType == kUInt32) {
      return GetRangeIndex<uint32_t>(key, valueType);
    } else {
      return GetRangeIndex<uint64_t>(key, valueType);
    }
  } else if (t == HASHRANGEINDEX) {
    if (idType == kUInt32) {
      return GetHashRangeIndex<uint32_t>(key, valueType);
    } else {
      return GetHashRangeIndex<uint64_t>(key, valueType);
    }
  } else {
    EULER_LOG(FATAL) << "not only support this index type " << t;
  }
  return nullptr;
}

}  // namespace euler
