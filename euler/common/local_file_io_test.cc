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

namespace euler {
namespace common {

TEST(LocalFileIOTest, TestRead) {
  LocalFileIO local_reader;
  FileIO& reader = local_reader;

  {
    LocalFileIO local_wrrter;
    FileIO& writer = local_wrrter;
    EXPECT_TRUE(writer.Initialize("filename=test.dat;read=false"));
    EXPECT_TRUE(writer.Append(static_cast<int32_t>(30)));
    EXPECT_TRUE(writer.Append(static_cast<int64_t>(9999)));
    EXPECT_TRUE(writer.Append(static_cast<float>(3.14)));
    EXPECT_TRUE(writer.Append(static_cast<double>(3.1415926)));
    EXPECT_TRUE(writer.Append(std::string("0123456789")));

    // Append List
    {
      std::vector<int> list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      EXPECT_TRUE(writer.Append(list));
    }

    {
      std::vector<double> list = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
      EXPECT_TRUE(writer.Append(list));
    }
  }

  EXPECT_TRUE(reader.Initialize("filename=test.dat;read=true"));

  {
    int32_t a;
    EXPECT_TRUE(reader.Read(&a));
    EXPECT_EQ(30, a);
  }

  {
    int64_t a;
    EXPECT_TRUE(reader.Read(&a));
    EXPECT_EQ(9999, a);
  }

  {
    float a;
    EXPECT_TRUE(reader.Read(&a));
    EXPECT_FLOAT_EQ(3.14, a);
  }

  {
    double a;
    EXPECT_TRUE(reader.Read(&a));
    EXPECT_DOUBLE_EQ(3.1415926, a);
  }

  {
    std::string a;
    EXPECT_TRUE(reader.Read(10, &a));
    EXPECT_EQ(std::string("0123456789"), a);
  }

  {
    std::vector<int> a;
    std::vector<int> b = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_TRUE(reader.Read(b.size(), &a));
    EXPECT_EQ(b, a);
  }

  {
    std::vector<double> a;
    std::vector<double> b = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    EXPECT_TRUE(reader.Read(b.size(), &a));
    EXPECT_EQ(b, a);
  }

  system("rm -f test.dat");
}

}  // namespace common
}  // namespace euler
