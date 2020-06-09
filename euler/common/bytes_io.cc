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

#include "euler/common/bytes_io.h"

#include <string>
#include <algorithm>
#include <iostream>

namespace euler {

bool BytesReader::Read(std::string* result) {
  uint32_t length = 0;
  if (!Read(&length)) { return false; }

  if (total_size_ < begin_idx_ + length) {
    return false;
  }

  result->resize(length);
  std::copy(bytes_ + begin_idx_, bytes_ + begin_idx_ + length,
            result->begin());
  begin_idx_ += length;
  return true;
}


bool BytesWriter::Write(const void* data, size_t size) {
  buffer_.append(reinterpret_cast<const char*>(data), size);
  return true;
}

}  // namespace euler
