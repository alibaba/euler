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

#include "euler/common/slice.h"

#include <algorithm>
#include <iostream>

namespace euler {

std::ostream& operator<<(std::ostream& o, Slice slice) {
  o.write(slice.data(), slice.size());
  return o;
}

size_t Slice::find(Slice pattern, size_t pos) {
  if (pos >= size_) {
    return npos;
  }

  // TODO(xingwo): refine this
  std::string s(*this);
  return s.find(std::string(pattern), pos);
}

size_t Slice::find(char c, size_t pos) const {
  if (pos >= size_) {
    return npos;
  }
  const char* result =
      reinterpret_cast<const char*>(memchr(data_ + pos, c, size_ - pos));
  return result != nullptr ? result - data_ : npos;
}

size_t Slice::rfind(char c, size_t pos) const {
  if (size_ == 0) return npos;
  for (const char* p = data_ + std::min(pos, size_ - 1); p >= data_; p--) {
    if (*p == c) {
      return p - data_;
    }
  }
  return npos;
}

Slice Slice::substr(size_t pos, size_t n) const {
  if (pos > size_) pos = size_;
  if (n > size_ - pos) n = size_ - pos;
  return Slice(data_ + pos, n);
}

}  // namespace euler
