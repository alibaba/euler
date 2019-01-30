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
#ifndef EULER_COMMON_FILE_IO_FACTORY_H_
#define EULER_COMMON_FILE_IO_FACTORY_H_

#include <unordered_map>
#include <functional>

#include "euler/common/file_io.h"

namespace euler {
namespace common {

class FileIOFactory {
 public:
  virtual FileIO* GetFileIO(const FileIO::ConfigMap& config) = 0;

  virtual std::vector<std::string> ListFile(
      const std::string& addr,
      int32_t port, const std::string& dir,
      const std::function<bool(std::string input)>& filter_fn = {}) = 0;
};

extern std::unordered_map<std::string, FileIOFactory*> factory_map;

}  // common
}  // namespace euler


#endif  // EULER_COMMON_FILE_IO_FACTORY_H_
