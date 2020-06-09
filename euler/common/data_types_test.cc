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

#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "euler/common/data_types.h"

namespace euler {
namespace common {

TEST(DataTypes, EdgeIDHashFuncTest) {
  EdgeIDHashFunc f;
  EdgeID eid = std::make_tuple(1, 2, 1);
  std::cout << std::hex << f(eid) << std::endl;
}
}  // namespace common
}  // namespace euler

