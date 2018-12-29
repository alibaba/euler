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

#ifndef EULER_COMMON_BYTES_READER_H_
#define EULER_COMMON_BYTES_READER_H_

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

namespace euler {
namespace common {

class BytesReader {
 public:
  BytesReader(const char* bytes, size_t size) {
    bytes_ = bytes;
    total_size_ = size;
    begin_idx_ = 0;
  }

  bool GetInt32(int32_t* result);

  bool GetInt32List(size_t num, std::vector<int32_t>* result);

  bool GetInt64(int64_t* result);

  bool GetInt64List(size_t num, std::vector<int64_t>* result);

  bool GetUInt64(uint64_t* result);

  bool GetUInt64List(size_t num, std::vector<uint64_t>* result);

  bool GetFloat(float* result);

  bool GetFloatList(size_t num, std::vector<float>* result);

  bool GetDouble(double* result);

  bool GetString(size_t length, std::string* result);

 private:
  const char* bytes_;

  size_t total_size_;

  size_t begin_idx_;
};

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_BYTES_READER_H_
