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

#include "euler/common/hdfs_file_io.h"

#include <string>
#include <vector>

#include "glog/logging.h"

#define BUFFER_SIZE 4096

namespace euler {
namespace common {

HdfsFileIO::HdfsFileIO() : read_(true), buffer_(nullptr),
                           capacity_(0), read_index_(0) {
}

bool HdfsFileIO::Initialize(const ConfigMap& conf) {
  // Reset state
  initialized_ = false;
  read_ = true;
  if (buffer_ != nullptr) {
    free(buffer_);
  }
  buffer_ = nullptr;
  capacity_ = 0;
  read_index_ = 0;
  int flags = O_RDONLY;

  std::string addr;
  int port = 0;
  std::string path;

  auto it = conf.find("addr");
  if (it == conf.end()) {
    LOG(ERROR) << "Please specify hdfs address";
    return false;
  }
  addr = it->second;

  it = conf.find("port");
  if (it == conf.end()) {
    LOG(ERROR) << "Please specify hdfs port";
    return false;
  }
  port = atoi(it->second.c_str());

  it = conf.find("path");
  if (it  == conf.end()) {
    LOG(ERROR) << "Please specify hdfs dir path";
    return false;
  }
  path = it->second;

  it = conf.find("read");
  if (it != conf.end() && (it->second == "false" ||
                           it->second == "no" ||
                           it->second == "0")) {
    read_ = false;
  }

  if (!read_) {
    flags = O_WRONLY | O_CREAT;
  }
  handle_ = hdfsConnect(addr.c_str(), port);
  file_ = hdfsOpenFile(handle_, path.c_str(), flags, 0, 0, 0);
  if (!file_) {
    LOG(ERROR) << "Open hdfs file: hdfs://" << addr << ":" << port
               << "/" << path << " failed!";
    return false;
  }

  buffer_ = malloc(BUFFER_SIZE);
  initialized_ = true;
  return true;
}

HdfsFileIO::~HdfsFileIO() {
  hdfsCloseFile(handle_, file_);
  if (buffer_ != nullptr) {
    free(buffer_);
  }
}

bool HdfsFileIO::ReadData(void* data, size_t size) {
  if (!initialized_) {
    return false;
  }

  char* dst = static_cast<char*>(data);
  char* src = static_cast<char*>(buffer_) + read_index_;
  while (read_index_ + size > capacity_) {
    memcpy(dst, src, capacity_ - read_index_);
    size -= capacity_ - read_index_;
    dst += capacity_ - read_index_;
    read_index_ = 0;
    capacity_ = hdfsRead(handle_, file_, buffer_, BUFFER_SIZE);
    src = static_cast<char*>(buffer_) + read_index_;
    if (capacity_ < static_cast<size_t>(BUFFER_SIZE)) {
      break;
    }
  }

  if (read_index_ + size > capacity_) {
    return false;
  }

  memcpy(dst, src, size);
  read_index_ += size;
  return true;
}

bool HdfsFileIO::WriteData(const void* data, size_t size) {
  if (!initialized_) {
    return false;
  }

  auto ret = hdfsWrite(handle_, file_, data, size);
  return (static_cast<size_t>(ret) == size);
}

std::vector<std::string> ListFile(const std::string& addr, int32_t port,
                                  const std::string& dir) {
  std::vector<std::string> file_list;
  int32_t num = 0;
  hdfsFS fs = hdfsConnect(addr.c_str(), port);
  hdfsFileInfo* info = hdfsListDirectory(fs, dir.c_str(), &num);
  for (int32_t i = 0; i < num; ++i) {
    std::string path_file_name = info->mName;
    file_list.push_back(path_file_name);
    ++info;
  }
  return file_list;
}

}  // namespace common
}  // namespace euler
