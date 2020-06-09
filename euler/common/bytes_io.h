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

#ifndef EULER_COMMON_BYTES_IO_H_
#define EULER_COMMON_BYTES_IO_H_

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <string>
#include <vector>

namespace euler {

class BytesReader {
 public:
  explicit BytesReader(const char* bytes) : BytesReader(bytes, UINT32_MAX) {
  }

  BytesReader(const char* bytes, uint32_t size) {
    bytes_ = bytes;
    total_size_ = size;
    begin_idx_ = 0;
  }

  template<typename T>
  bool Read(T* result) {
    if (total_size_ < begin_idx_ + sizeof(T)) {
      return false;
    }

    *result = *reinterpret_cast<const T*>(bytes_ + begin_idx_);
    begin_idx_ += sizeof(T);
    return true;
  }

  template<typename T>
  bool Read(std::vector<T>* result) {
    uint32_t num = 0;
    if (!Read(&num)) {return false;}
    if (total_size_ < begin_idx_ + num * sizeof(T)) {
      return false;
    }
    result->resize(num);
    auto begin = reinterpret_cast<const T*>(bytes_ + begin_idx_);
    std::copy(begin, begin + num, result->begin());
    begin_idx_ += sizeof(T) * num;
    return true;
  }

  template<typename T>
  bool Read(std::vector<std::vector<T>>* result) {
    uint32_t num = 0;
    if (!Read(&num)) {return false;}
    result->resize(num);
    for (size_t i = 0; i < num; ++i) {
      if (!Read(&((*result)[i]))) { return false; }
    }
    return true;
  }

  bool Read(std::string* result);

 private:
  const char* bytes_;
  uint32_t total_size_;
  uint32_t begin_idx_;
};

class BytesWriter {
 public:
  BytesWriter() {
    buffer_.reserve(512);
  }

  template<typename T>
  bool Write(T result) {
    return Write(&result, sizeof(T));
  }

  template<typename T>
  bool Write(const std::vector<T>& result) {
    uint32_t num = result.size();
    Write(num);
    Write(result.data(), result.size() * sizeof(T));
    return true;
  }

  template<typename T>
  bool Write(const std::vector<std::vector<T>>& result) {
    uint32_t num = result.size();
    Write(num);

    for (auto r : result) {
      Write(r);
    }

    return true;
  }

  bool Write(const std::string& result) {
    uint32_t num = result.size();
    Write(num);
    return Write(result.data(), num);
  }

  bool Write(const void* data, size_t size);

  const std::string& data() const {
    return buffer_;
  }

 private:
  std::string buffer_;
};

}  // namespace euler

#endif  // EULER_COMMON_BYTES_IO_H_
