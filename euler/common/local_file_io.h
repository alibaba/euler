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

#ifndef EULER_COMMON_LOCAL_FILE_IO_H_
#define EULER_COMMON_LOCAL_FILE_IO_H_

#include <fstream>
#include <string>

#include "euler/common/file_io.h"

namespace euler {
namespace common {

class LocalFileIO : public FileIO {
 public:
  LocalFileIO();

  virtual ~LocalFileIO();

  bool Initialize(const ConfigMap& conf) override;

  bool ReadData(void* data, size_t size) override;
  bool WriteData(const void* data, size_t size) override;

  bool FileEnd() override { return file_.eof();}

 private:
  bool read_;
  std::fstream file_;
};

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_LOCAL_FILE_IO_H_
