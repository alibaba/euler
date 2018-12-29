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

#include "euler/common/bytes_reader.h"

#include <string>
#include <algorithm>
#include <iostream>

namespace euler {
namespace common {

bool BytesReader::GetInt32(int32_t* result) {
  if (total_size_ < begin_idx_ + sizeof(int32_t)) {
    return false;
  }
  const char* bytes_begin = bytes_ + begin_idx_;
  *result = *(int32_t*)bytes_begin;
  begin_idx_ += sizeof(int32_t);
  return true;
}

bool BytesReader::GetInt32List(size_t num, std::vector<int32_t>* result) {
  if (total_size_ < begin_idx_ + num * sizeof(int32_t)) {
    return false;
  }
  result->resize(num);
  std::copy(bytes_ + begin_idx_,
            bytes_ + begin_idx_ + sizeof(int32_t) * num,
            (char*)(result->data()));
  begin_idx_ += sizeof(int32_t) * num;
  return true;
}

bool BytesReader::GetInt64(int64_t* result) {
  if (total_size_ < begin_idx_ + sizeof(int64_t)) {
    return false;
  }
  const char* bytes_begin = bytes_ + begin_idx_;
  *result = *(int64_t*)bytes_begin;
  begin_idx_ += sizeof(int64_t);
  return true;
}

bool BytesReader::GetInt64List(size_t num, std::vector<int64_t>* result) {
  if (total_size_ < begin_idx_ + num * sizeof(int64_t)) {
    return false;
  }
  result->resize(num);
  std::copy(bytes_ + begin_idx_,
            bytes_ + begin_idx_ + sizeof(int64_t) * num,
            (char*)(result->data()));
  begin_idx_ += sizeof(int64_t) * num;
  return true;
}

bool BytesReader::GetUInt64(uint64_t* result) {
  if (total_size_ < begin_idx_ + sizeof(uint64_t)) {
    return false;
  }
  const char* bytes_begin = bytes_ + begin_idx_;
  *result = *(uint64_t*)bytes_begin;
  begin_idx_ += sizeof(uint64_t);
  return true;
}

bool BytesReader::GetUInt64List(size_t num, std::vector<uint64_t>* result) {
  if (total_size_ < begin_idx_ + num * sizeof(uint64_t)) {
    return false;
  }
  result->resize(num);
  std::copy(bytes_ + begin_idx_,
            bytes_ + begin_idx_ + sizeof(uint64_t) * num,
            (char*)(result->data()));
  begin_idx_ += sizeof(uint64_t) * num;
  return true;
}

bool BytesReader::GetFloat(float* result) {
  if (total_size_ < begin_idx_ + sizeof(float)) {
    return false;
  }
  const char* bytes_begin = bytes_ + begin_idx_;
  *result = *(float*)bytes_begin;
  begin_idx_ += sizeof(float);
  return true;
}

bool BytesReader::GetFloatList(size_t num, std::vector<float>* result) {
  if (total_size_ < begin_idx_ + num * sizeof(float)) {
    return false;
  }
  result->resize(num);
  std::copy(bytes_ + begin_idx_,
            bytes_ + begin_idx_ + sizeof(float) * num,
            (char*)(result->data()));
  begin_idx_ += sizeof(float) * num;
  return true;
}

bool BytesReader::GetDouble(double* result) {
  if (total_size_ < begin_idx_ + sizeof(double)) {
    return false;
  }
  const char* bytes_begin = bytes_ + begin_idx_;
  *result = *(double*)bytes_begin;
  begin_idx_ += sizeof(double);
  return true;
}

bool BytesReader::GetString(size_t length, std::string* result) {
  if (total_size_ < begin_idx_ + length) {
    return false;
  }
  result->resize(length);
  std::copy(bytes_ + begin_idx_, bytes_ + begin_idx_ + length,
            result->begin());
  begin_idx_ += length;
  return true;
}

}  // namespace common
}  // namespace euler
