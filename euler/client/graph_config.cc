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

#include "euler/client/graph_config.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "glog/logging.h"

#include "euler/common/string_util.h"

namespace euler {
namespace client {

GraphConfig::GraphConfig() {
}

bool GraphConfig::Load(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    LOG(INFO) << "Open graph config file: " << filename << "failed!";
    return false;
  }
  char* line = NULL;
  size_t n = 0;
  ssize_t nread = 0;
  while ((nread = getline(&line, &n, fp)) > 0) {
    line[nread] = '\0';
    std::vector<std::string> vec;
    euler::common::split_string(line, '=', &vec);
    if (vec.size() != 2) {
      continue;
    }
    Add(euler::common::trim("\t\r ", vec[0]),
        euler::common::trim("\t\r ", vec[1]));
  }
  free(line);
  fclose(fp);
  return true;
}

bool GraphConfig::Get(const std::string& key, std::string* value) const {
  auto it = config_.find(key);
  if (it != config_.end()) {
    *value = it->second;
    return true;
  }
  return false;
}

bool GraphConfig::Get(const std::string& key, int* value) const {
  std::string value_string;
  if (!Get(key, &value_string)) {
    return false;
  }

  try {
    *value = std::stoi(value_string);
    return true;
  } catch (std::invalid_argument e) {
  }
  return false;
}

void GraphConfig::Add(const std::string& key,
                      const std::string& value) {
  config_.insert({key, value});
}

void GraphConfig::Add(const std::string& key, int value) {
  Add(key, std::to_string(value));
}

void GraphConfig::Remove(const std::string& key) {
  config_.erase(key);
}

std::string GraphConfig::DebugString() const {
  std::string out = "{\n";
  for (auto it = config_.begin(); it != config_.end(); ++it) {
    out.append(it->first);
    out.append(" = ");
    out.append(it->second);
    out.append("\n");
  }
  out.append("}");
  return out;
}

}  // namespace client
}  // namespace euler
