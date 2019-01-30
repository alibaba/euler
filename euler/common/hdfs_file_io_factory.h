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
#ifndef EULER_COMMON_HDFS_FILE_IO_FACTORY_H_
#define EULER_COMMON_HDFS_FILE_IO_FACTORY_H_

#include <string>
#include <vector>
#include <iostream>

#include "glog/logging.h"

#include "euler/common/hdfs_file_io.h"
#include "euler/common/file_io_factory.h"
namespace euler {
namespace common {

class HdfsFileIOFactory : public FileIOFactory {
 public:
  FileIO* GetFileIO(const FileIO::ConfigMap& config) override {
    FileIO* file_io = new HdfsFileIO();
    if (file_io->Initialize(config)) {
      return file_io;
    } else {
      return nullptr;
    }
  }

  std::vector<std::string> ListFile(
      const std::string& addr,
      int32_t port, const std::string& dir,
      const std::function<bool(std::string input)>& filter_fn = {}) override {
    std::vector<std::string> file_list;
    int32_t num = 0;
    hdfsFS fs = hdfsConnect(addr.c_str(), port);
    hdfsFileInfo* info = hdfsListDirectory(fs, dir.c_str(), &num);
    for (int32_t i = 0; i < num; ++i) {
      std::string path_file_name = info->mName;
      if (filter_fn && filter_fn(path_file_name)) {
        file_list.push_back(path_file_name);
      } else if (!filter_fn) {
        file_list.push_back(path_file_name);
      }
      ++info;
    }
    return file_list;
  }
};

class HdfsFileIOFactoryReg {
 public:
  HdfsFileIOFactoryReg() {
    if (factory_map.find("hdfs") == factory_map.end()) {
      std::cout << "hdfs file io factory register" << std::endl;
      factory_map["hdfs"] = new HdfsFileIOFactory();
    }
  }
};

}  // namespace common
}  // namespace euler

#endif
