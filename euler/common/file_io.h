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

#ifndef EULER_COMMON_FILE_IO_H_
#define EULER_COMMON_FILE_IO_H_

#include <stdint.h>

#include <string>
#include <vector>
#include <unordered_map>

#include "euler/common/string_util.h"

namespace euler {
namespace common {

class FileIO {
 public:
  using ConfigMap = std::unordered_map<std::string, std::string>;

  FileIO() : initialized_(false) { }

  virtual ~FileIO() {}

  bool Initialize(const std::string& config) {
    auto conf = ParseConfig(config);
    return Initialize(conf);
  }

  virtual bool Initialize(const ConfigMap& config) = 0;

  virtual bool ReadData(void* data, size_t size) = 0;
  virtual bool WriteData(const void* data, size_t size) = 0;

  template<typename T>
  inline bool Read(T* data);
  template<typename T>
  inline bool  Read(size_t count, std::vector<T>* list);
  inline bool Read(size_t len, std::string* str);

  template<typename T>
  inline bool Append(T data);
  template <typename T>
  inline bool Append(const std::vector<T>& list);
  inline bool Append(const std::string& str);

  virtual bool FileEnd() = 0;

  bool initialized() const { return initialized_; }

 private:
  ConfigMap ParseConfig(const std::string& config) {
    std::unordered_map<std::string, std::string> conf;
    std::vector<std::string> vec;
    split_string(config, ';', &vec);
    for (auto it = vec.begin(); it != vec.end(); ++it) {
      std::vector<std::string> kv;
      split_string(*it, '=', &kv);
      if (kv.size() != 2) {
        vec.clear();
        break;
      }
      conf.insert(std::make_pair(kv[0], kv[1]));
    }
    return conf;
  }

 protected:
  bool initialized_;
};

template<typename T>
bool FileIO::Read(T* data) {
  return ReadData(data, sizeof(T));
}

template<typename T>
bool  FileIO::Read(size_t count, std::vector<T>* list) {
  list->resize(count);
  return ReadData(list->data(), count * sizeof(T));
}

bool FileIO::Read(size_t len, std::string* str) {
  str->resize(len);
  return ReadData(&(str->front()), len);
}

template<typename T>
bool FileIO::Append(T data) {
  return WriteData(&data, sizeof(data));
}

template <typename T>
bool FileIO::Append(const std::vector<T>& list) {
  return WriteData(list.data(), list.size() * sizeof(T));
}

bool FileIO::Append(const std::string& str) {
  return WriteData(str.data(), str.size());
}

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_FILE_IO_H_
