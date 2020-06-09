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

#include "euler/common/file_io.h"

#include <utility>
#include <unordered_map>

#include "euler/common/logging.h"

namespace euler {

typedef std::unordered_map<std::string,
                           FileIORegistrar::Factory> FileIORegistry;

static FileIORegistry& GlobalFileIORegistry() {
  static FileIORegistry registry;
  return registry;
}

void FileIORegistrar::Register(const std::string& scheme, Factory&& factory) {
  if (!GlobalFileIORegistry().insert({scheme, factory}).second) {
    EULER_LOG(ERROR) << "Register File IO for scheme: " << scheme << " failed!";
  }
}

Status CreateFileIO(const std::string& scheme, std::unique_ptr<FileIO>* out) {
  auto& registry = GlobalFileIORegistry();
  auto it = registry.find(scheme);
  if (it == registry.end()) {
    return Status::NotFound("No FileIO for scheme: ", scheme);
  }

  std::unique_ptr<FileIO> io((*it->second)());
  *out = std::move(io);
  return Status::OK();
}

}  // namespace euler
