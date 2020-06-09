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

#include "euler/common/bytes_io.h"
#include "euler/common/bytes_compute.h"

#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"


namespace euler {

TEST(BytesIOTest, TestReadWriter) {
  {
    int32_t i = 15;
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    int32_t ii;
    reader.Read(&ii);
    EXPECT_EQ(ii, i);
    delete[] buffer;
  }

  {
    uint64_t i = 15;
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    uint64_t ii;
    reader.Read(&ii);
    EXPECT_EQ(ii, i);
    delete[] buffer;
  }

  {
    float i = 1.5;
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    float ii;
    reader.Read(&ii);
    EXPECT_EQ(ii, i);
    delete[] buffer;
  }

  {
    std::string i = "hell world";
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    std::string ii;
    reader.Read(&ii);
    EXPECT_EQ(ii, i);
    delete[] buffer;
  }

  {
    std::vector<std::string> i = {"1", "11", "123"};
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    std::vector<std::string> ii;
    reader.Read(&ii);
    EXPECT_EQ(ii.size(), i.size());
    for (size_t j = 0; j < i.size(); ++j) {
      EXPECT_EQ(ii[j], i[j]);
    }
    delete[] buffer;
  }

  {
    std::vector<uint64_t> i = {1, 2, 4, 5};
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    std::vector<uint64_t> ii;
    reader.Read(&ii);
    EXPECT_EQ(ii.size(), i.size());
    for (size_t j = 0; j < i.size(); ++j) {
      EXPECT_EQ(ii[j], i[j]);
    }
    delete[] buffer;
  }

  {
    std::vector<std::vector<float>> i = {{1.0}, {1.0, 2.0}, {1.0, 2.0, 3.0}};
    uint32_t size = BytesSize(i);
    char* buffer = new char[size];
    BytesWriter writer;
    EXPECT_TRUE(writer.Write(i));
    BytesReader reader(writer.data().c_str());
    std::vector<std::vector<float>> ii;
    reader.Read(&ii);
    EXPECT_EQ(ii.size(), i.size());
    for (size_t j = 0; j < i.size(); ++j) {
      EXPECT_EQ(ii[j].size(), i[j].size());
      for (size_t k = 0; k < i[j].size(); ++k) {
        EXPECT_EQ(ii[j][k], i[j][k]);
      }
    }
    delete[] buffer;
  }
}

}  // namespace euler
