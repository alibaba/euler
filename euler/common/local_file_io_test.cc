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

#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "euler/common/file_io_factory.h"

namespace euler {
namespace common {

TEST(LocalFileIOTest, TestRead) {
  FileIOFactory* file_io_factory = factory_map["local"];
  {
    FileIO::ConfigMap config;
    config["filename"] = "test.dat";
    config["read"] = "false";
    FileIO* writer = file_io_factory->GetFileIO(config);
    EXPECT_TRUE(writer->Append(static_cast<int32_t>(30)));
    EXPECT_TRUE(writer->Append(static_cast<int64_t>(9999)));
    EXPECT_TRUE(writer->Append(static_cast<float>(3.14)));
    EXPECT_TRUE(writer->Append(static_cast<double>(3.1415926)));
    EXPECT_TRUE(writer->Append(std::string("0123456789")));

    // Append List
    {
      std::vector<int> list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      EXPECT_TRUE(writer->Append(list));
    }

    {
      std::vector<double> list = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
      EXPECT_TRUE(writer->Append(list));
    }
    delete writer;
  }

  {
    FileIO::ConfigMap config;
    config["filename"] = "test.dat";
    config["read"] = "true";
    FileIO* reader = file_io_factory->GetFileIO(config);

    int32_t i32;
    EXPECT_TRUE(reader->Read(&i32));
    EXPECT_EQ(30, i32);

    int64_t i64;
    EXPECT_TRUE(reader->Read(&i64));
    EXPECT_EQ(9999, i64);

    float f;
    EXPECT_TRUE(reader->Read(&f));
    EXPECT_FLOAT_EQ(3.14, f);

    double d;
    EXPECT_TRUE(reader->Read(&d));
    EXPECT_DOUBLE_EQ(3.1415926, d);

    std::string s;
    EXPECT_TRUE(reader->Read(10, &s));
    EXPECT_EQ(std::string("0123456789"), s);

    std::vector<int> v_i;
    std::vector<int> b1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_TRUE(reader->Read(b1.size(), &v_i));
    EXPECT_EQ(b1, v_i);

    std::vector<double> v_d;
    std::vector<double> b2 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    EXPECT_TRUE(reader->Read(b2.size(), &v_d));
    EXPECT_EQ(b2, v_d);
    delete reader;
  }

  system("rm -f test.dat");
}
}  // namespace common
}  // namespace euler
