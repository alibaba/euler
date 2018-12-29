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
#ifndef EULER_COMMON_STRING_UTIL_H_
#define EULER_COMMON_STRING_UTIL_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace euler {
namespace common {

int32_t split_string(const std::string &s, char delim,
                     std::vector<std::string> *v);
std::string join_string(const std::vector<std::string> &parts,
                        const std::string &separator);
std::string& ltrim(const std::string& chars, std::string& str);
std::string& rtrim(const std::string& chars, std::string& str);
std::string& trim(const std::string& chars, std::string& str);

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_STRING_UTIL_H_
