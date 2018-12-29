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

#include "euler/common/local_file_io.h"

#include "glog/logging.h"

namespace euler {
namespace common {

LocalFileIO::LocalFileIO(): read_(true) {
}

LocalFileIO::~LocalFileIO() {
  file_.close();
}

bool LocalFileIO::Initialize(const ConfigMap& conf) {
  initialized_ = false;
  read_ = true;

  std::string filename;
  auto it = conf.find("filename");
  if (it == conf.end()) {
    LOG(ERROR) << "Please specify filename in config";
    return false;
  }
  filename = it->second;

  it = conf.find("read");
  if (it != conf.end() && (it->second == "false" ||
                           it->second == "no" ||
                           it->second == "0")) {
    read_ = false;
  }

  std::ios::openmode mode = std::fstream::in | std::fstream::binary;
  if (!read_) {
    mode = std::fstream::out | std::fstream::binary;
  }
  file_.open(filename.c_str(), mode);
  if (!file_.is_open()) {
    LOG(ERROR) << "file :" << filename << " open failed";
    return false;
  }

  initialized_ = true;
  return true;
}

bool LocalFileIO::ReadData(void* data, size_t size) {
  if (!initialized_) {
    return false;
  }
  file_.read(static_cast<char*>(data), size);
  return file_.good();
}

bool LocalFileIO::WriteData(const void* data, size_t size) {
  if (!initialized_) {
    return false;
  }
  file_.write(static_cast<const char*>(data), size);
  return file_.good();
}

}  // namespace common
}  // namespace euler
