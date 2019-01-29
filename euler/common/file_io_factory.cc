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

#include "euler/common/file_io_factory.h"

namespace euler {
namespace common {
std::unordered_map<std::string, FileIOFactory*> factory_map;
}  // namespace euler
}  // namespace common

#ifdef HDFS
#include "euler/common/hdfs_file_io_factory.h"
static euler::common::HdfsFileIOFactoryReg hdfs_reg;
#endif

#include "euler/common/local_file_io_factory.h"
static euler::common::LocalFileIOFactoryReg local_reg;
