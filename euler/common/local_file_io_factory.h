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
#ifndef EULER_COMMON_LOCAL_FILE_IO_FACTORY_H_
#define EULER_COMMON_LOCAL_FILE_IO_FACTORY_H_
#include <dirent.h>
#include <string>
#include <vector>
#include <iostream>

#include "glog/logging.h"

#include "euler/common/local_file_io.h"
#include "euler/common/file_io_factory.h"

namespace euler {
namespace common {

class LocalFileIOFactory : public FileIOFactory {
 public:
  FileIO* GetFileIO(const FileIO::ConfigMap& config) override {
    FileIO* file_io = new LocalFileIO();
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
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL) {
      LOG(ERROR) << "directory " << dir << " invalid";
      return file_list;
    }
    while ((dirp = readdir(dp)) != NULL) {
      std::string file_name(dirp->d_name);
      if (filter_fn && filter_fn(file_name)) {
        file_list.push_back(dir + "/" + file_name);
      } else if (!filter_fn) {
        file_list.push_back(dir + "/" + file_name);
      }
    }
    closedir(dp);
    return file_list;
  }
};

class LocalFileIOFactoryReg {
 public:
  LocalFileIOFactoryReg() {
    if (factory_map.find("local") == factory_map.end()) {
      std::cout << "local file io factory register" << std::endl;
      factory_map["local"] = new LocalFileIOFactory();
    }
  }
};

}  // namespace common
}  // namespace euler

#endif
