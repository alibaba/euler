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

#ifndef EULER_CLIENT_GRAPH_CONFIG_H_
#define EULER_CLIENT_GRAPH_CONFIG_H_

#include <string>
#include <map>

namespace euler {

class GraphConfig {
 public:
  GraphConfig();

  bool Load(const std::string& filename);

  bool Get(const std::string& key, std::string* value) const;
  bool Get(const std::string& key, int* value) const;
  void Add(const std::string& key, const std::string& value);
  void Add(const std::string& key, int value);
  void Remove(const std::string& key);

  std::string DebugString() const;

 private:
  std::map<std::string, std::string> config_;
};

}  // namespace euler

#endif  // EULER_CLIENT_GRAPH_CONFIG_H_
