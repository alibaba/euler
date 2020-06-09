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

#ifndef EULER_CORE_INDEX_INDEX_TYPES_H_
#define EULER_CORE_INDEX_INDEX_TYPES_H_

#include <unordered_map>
#include <string>

#include "euler/core/framework/types.h"

namespace euler {

enum IndexType : int32_t {
  HASHINDEX = 0,
  RANGEINDEX,
  HASHRANGEINDEX
};

enum IndexResultType : int32_t {
  HASHINDEXRESULT = 0,
  RANGEINDEXRESULT,
  COMMONINDEXRESULT
};

enum IndexSearchType : int32_t {
  LESS = 0,
  LESS_EQ,
  EQ,
  GREATER,
  GREATER_EQ,
  NOT_EQ,
  IN,
  NOT_IN
};

typedef DataType IndexDataType;

}  // namespace euler

#endif  // EULER_CORE_INDEX_INDEX_TYPES_H_
