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

#ifndef EULER_COMMON_HDFS_FILE_IO_H_
#define EULER_COMMON_HDFS_FILE_IO_H_

#include <vector>
#include <string>

#include "third_party/hdfs/hdfs.h"

#include "euler/common/file_io.h"

namespace euler {
namespace common {

class HdfsFileIO : public FileIO {
 public:
  HdfsFileIO();

  ~HdfsFileIO();

  bool Initialize(const ConfigMap& conf) override;

  bool ReadData(void* data, size_t size) override;
  bool WriteData(const void* data, size_t size) override;

  bool FileEnd() override { return false; }

 private:
  bool read_;
  void* buffer_;
  size_t capacity_;
  size_t read_index_;
  hdfsFS handle_;
  hdfsFile file_;
};

std::vector<std::string> ListFile(const std::string& addr,
                                  int32_t port, const std::string& dir);

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_HDFS_FILE_IO_H_
