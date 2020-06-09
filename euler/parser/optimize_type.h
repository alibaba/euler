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

#ifndef EULER_PARSER_OPTIMIZE_TYPE_H_
#define EULER_PARSER_OPTIMIZE_TYPE_H_

namespace euler {

enum OptimizerType {
  local = 0,
  distribute = 1,
  graph_partition = 2
};

/* run mode -> optimize type */
extern bool compatible_matrix[3][3];

}  // namespace euler

#endif  // EULER_PARSER_OPTIMIZE_TYPE_H_
