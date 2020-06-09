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

#ifndef EULER_COMMON_FILE_IO_H_
#define EULER_COMMON_FILE_IO_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <utility>

#include "euler/common/str_util.h"
#include "euler/common/status.h"
#include "euler/common/slice.h"

namespace euler {

class FileIO {
 public:
  using ConfigMap = std::unordered_map<std::string, std::string>;

  typedef std::function<bool (const std::string&)> FilterFunc;

  FileIO() : initialized_(false), read_index_(0), file_size_(0) { }

  virtual ~FileIO() {}

  bool Initialize(const std::string& config) {
    auto conf = ParseConfig(config);
    return Initialize(conf);
  }

  virtual bool Initialize(const ConfigMap& config) = 0;

  virtual bool IsDirectory() = 0;
  virtual std::vector<std::string> ListDirectory(FilterFunc func) = 0;

  virtual bool ReadData(void* data, size_t size) = 0;
  virtual bool WriteData(const void* data, size_t size) = 0;

  template<typename T>
  inline bool Read(T* data);

  inline bool Read(std::string* str);

  template<typename T>
  inline bool  Read(std::vector<T>* list);

  inline bool Read(std::vector<std::string>* list);

  template<typename T>
  inline bool Append(const T& data);

  inline bool Append(const std::string& str);

  template <typename T>
  inline bool Append(const std::vector<T>& list);

  inline bool Append(const std::vector<std::string>& list);

  bool FileEnd() const { return read_index_ >= file_size_; }

  size_t read_index() const { return read_index_; }

  size_t file_size() const {return file_size_; }

  virtual std::unique_ptr<FileIO> Open(Slice path, bool readonly) = 0;

  bool initialized() const { return initialized_; }

 private:
  ConfigMap ParseConfig(const std::string& config) {
    std::unordered_map<std::string, std::string> conf;
    auto vec = Split(config, ";");
    for (auto it = vec.begin(); it != vec.end(); ++it) {
      auto kv = Split(*it, "=");
      if (kv.size() != 2) {
        vec.clear();
        break;
      }
      conf.insert({kv[0], kv[1]});
    }
    return conf;
  }

 protected:
  bool initialized_;
  size_t read_index_;
  size_t file_size_;
};

template<typename T>
bool FileIO::Read(T* data) {
  return ReadData(data, sizeof(T));
}

bool FileIO::Read(std::string* data) {
  uint32_t len = 0;
  bool succ = Read(&len);
  if (succ) {
    data->resize(len);
    succ = ReadData(&(data->front()), len);
  }
  return succ;
}

template<typename T>
bool FileIO::Read(std::vector<T>* list) {
  uint32_t len = 0;
  bool succ = Read(&len);
  if (succ) {
    list->resize(len);
    succ = ReadData(list->data(), len* sizeof(T));
  }
  return succ;
}

bool FileIO::Read(std::vector<std::string>* list) {
  uint32_t len = 0;
  bool succ = Read(&len);
  if (succ) {
    list->resize(len);
    for (size_t i = 0; succ && i < len; ++i) {
      succ = Read(&(*list)[i]);
    }
  }
  return succ;
}

template<typename T>
bool FileIO::Append(const T& data) {
  return WriteData(&data, sizeof(data));
}

bool FileIO::Append(const std::string& str) {
  uint32_t len = str.size();
  return Append(len) && WriteData(str.data(), len);
}

template <typename T>
bool FileIO::Append(const std::vector<T>& list) {
  uint32_t len = list.size();
  return Append(len) && WriteData(list.data(), len * sizeof(T));
}

bool FileIO::Append(const std::vector<std::string>& list) {
  uint32_t len = list.size();
  bool succ = Append(len);
  for (uint32_t i = 0; succ && i < len; ++i) {
    succ = Append(list[i]);
  }
  return succ;
}

#define REGISTER_FILE_IO(scheme, cls) \
  REGISTER_FILE_IO_UNIQ_HELPER(__COUNTER__, scheme, cls)

#define REGISTER_FILE_IO_UNIQ_HELPER(counter, scheme, cls) \
  REGISTER_FILE_IO_UNIQ(counter, scheme, cls)

#define REGISTER_FILE_IO_UNIQ(counter, scheme, cls)   \
  static ::euler::FileIORegistrar                     \
  io_registrar__##counter##__obj(                     \
      scheme,                                         \
      [] () -> euler::FileIO* {                       \
        return new cls;                               \
      });

class FileIORegistrar {
 public:
  typedef FileIO* (*Factory) ();

  FileIORegistrar(const std::string& scheme, Factory&& factory) {
    Register(scheme, std::move(factory));
  }

 private:
  void Register(const std::string& scheme, Factory&& factory);
};

Status CreateFileIO(const std::string& scheme, std::unique_ptr<FileIO>* out);

}  // namespace euler

#endif  // EULER_COMMON_FILE_IO_H_
