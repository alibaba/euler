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

#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "euler/common/logging.h"
#include "euler/common/file_io.h"

namespace euler {

TEST(HdfsLocalFileIOTest, TestRead) {
  {
    FileIO::ConfigMap config;
    config["scheme"] = "hdfs";
    config["namenode"] = "gpu1.hs.na61:9000";
    config["path"] = "/data/xxx/euler.meta";
    config["read"] = "true";

    std::unique_ptr<FileIO> reader;
    ASSERT_TRUE(CreateFileIO("hdfs", &reader).ok());
    ASSERT_TRUE(reader->Initialize(config));
    EULER_LOG(INFO) << "File size: " << reader->file_size();

    std::string s;
    EXPECT_TRUE(reader->Read(&s));
    EXPECT_EQ("Graph", s);
  }
}

TEST(HdfsLocalFileIOTest, TestListDirectory) {
  {
    FileIO::ConfigMap config;
    config["scheme"] = "hdfs";
    config["namenode"] = "gpu1.hs.na61:9000";
    config["path"] = "/data/xxx";
    config["read"] = "true";

    std::unique_ptr<FileIO> reader;
    ASSERT_TRUE(CreateFileIO("hdfs", &reader).ok());
    ASSERT_TRUE(reader->Initialize(config));
    EULER_LOG(INFO) << "File size: " << reader->file_size();

    ASSERT_TRUE(reader->IsDirectory());
    auto filter = [] (const std::string&) { return true; };
    auto files = reader->ListDirectory(filter);
    ASSERT_EQ(1u, files.size());
    for (auto& file : files) {
      EULER_LOG(INFO) << file;
    }
  }
}

}  // namespace euler
