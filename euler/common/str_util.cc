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

#include "euler/common/str_util.h"

#include <string>
#include <sstream>
#include<vector>

#include "euler/common/slice.h"

namespace euler {

std::string JoinString(const std::vector<std::string>& parts,
                       const std::string& separator) {
  std::stringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << separator;
    }
    ss << parts[i];
  }
  return ss.str();
}

std::string JoinPath_impl_(std::initializer_list<Slice> paths) {
  std::string result;
  for (auto path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    if (path[0] == '/') {  // remove '/' in path
      path.remove_prefix(1);
    }

    if (path[path.size() - 1] == '/') {
      path.remove_suffix(1);
    }

    result.append("/");
    result.append(path.data(), path.size());
  }
  return result;
}

void ParseURI(Slice uri, Slice* scheme, Slice* host, Slice* path) {
  *scheme = "file";
  auto pos = uri.find("://");
  if (pos == Slice::npos) {
    *path = uri;
    return;
  }

  *scheme = Slice(uri.data(), pos);
  uri.remove_prefix(pos + 3);

  pos = uri.find("/");
  if (pos == Slice::npos) {  // No path
    *host = uri;
    return;
  }

  *host = Slice(uri.data(), pos);

  uri.remove_prefix(pos);
  *path = uri;
}

}  // namespace euler
