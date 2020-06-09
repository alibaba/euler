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

#ifndef EULER_COMMON_ALIAS_METHOD_H_
#define EULER_COMMON_ALIAS_METHOD_H_

#include <map>
#include <vector>
#include <string>

#include "euler/common/random.h"

namespace euler {
namespace common {

class AliasMethod {
 public:
  void Init(const std::vector<float>& weights);

  int64_t Next() const;

  size_t GetSize() const;

  std::string ShowData() const;

 private:
  std::vector<float> prob_;
  std::vector<int64_t> alias_;
  int64_t NextLong(int64_t n) const;
};

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_ALIAS_METHOD_H_
