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

#include "euler/common/timmer.h"

namespace euler {
namespace common {

thread_local struct timeval start;
thread_local struct timeval end;

void TimmerBegin() {
  gettimeofday(&start, NULL);
}

uint64_t GetTimmerInterval() {
  gettimeofday(&end, NULL);
  uint64_t diff = 1000000 * (end.tv_sec - start.tv_sec)
                  + end.tv_usec - start.tv_usec;
  return diff;
}

}  // namespace common
}  // namespace euler
