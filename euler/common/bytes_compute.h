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

#ifndef EULER_COMMON_BYTES_COMPUTE_H_
#define EULER_COMMON_BYTES_COMPUTE_H_

#include <string>
#include <vector>

namespace euler {

template<typename T>
inline uint32_t BytesSize(const T&) {
  return sizeof(T);
}

template<>
inline uint32_t BytesSize<std::string>(const std::string& s) {
  return sizeof(uint32_t) + s.size();
}

template<typename T>
inline uint32_t BytesSize(const std::vector<T>& v) {
  return sizeof(T)* v.size() + sizeof(uint32_t);
}

template<typename T>
inline uint32_t BytesSize(const std::vector<std::vector<T>>& v) {
  uint32_t num = v.size();
  uint32_t total = sizeof(num);
  for (uint32_t i = 0; i < num; ++i) {
    total += BytesSize(v[i]);
  }
  return total;
}

template<>
inline uint32_t BytesSize<std::string>(const std::vector<std::string>& v) {
  uint32_t len = sizeof(uint32_t);
  for (size_t i = 0; i < v.size(); ++i) {
    len += BytesSize(v[i]);
  }
  return len;
}

}  // namespace euler

#endif  // EULER_COMMON_BYTES_COMPUTE_H_
