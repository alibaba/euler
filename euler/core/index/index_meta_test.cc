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

#include <iostream>
#include <string>
#include <memory>

#include "gtest/gtest.h"

#include "euler/core/index/index_meta.h"
#include "euler/common/env.h"

using std::string;

namespace euler {

TEST(IndexMeta, AddHas) {
  IndexMeta im;
  IndexMetaRecord record;
  record.type = HASHINDEX;
  ASSERT_TRUE(im.AddMeta("name", record));
  ASSERT_TRUE(im.AddMeta("age", record));
  ASSERT_TRUE(!im.AddMeta("age", record));
  ASSERT_TRUE(im.HasIndex("age"));
  ASSERT_TRUE(im.HasIndex("name"));
  ASSERT_TRUE(!im.HasIndex("family"));
}

TEST(IndexMeta, Serialize) {
  IndexMeta im;

  IndexMetaRecord record;
  record.type = HASHINDEX;
  record.idType = kUInt32;
  record.valueType = kString;
  ASSERT_TRUE(im.AddMeta("name", record));

  IndexMetaRecord record2;
  record2.type = RANGEINDEX;
  record2.idType = kUInt32;
  record2.valueType = kUInt32;
  ASSERT_TRUE(im.AddMeta("age", record2));

  string buffer;
  ASSERT_TRUE(im.Serialize(&buffer));
  IndexMeta im2;
  ASSERT_TRUE(im2.Deserialize(buffer.c_str(), buffer.size()));
  ASSERT_TRUE(im2.HasIndex("age"));
  ASSERT_TRUE(im2.HasIndex("name"));
  ASSERT_EQ(im2.GetMetaRecord("age").type, RANGEINDEX);
  ASSERT_EQ(im2.GetMetaRecord("name").type, HASHINDEX);
  ASSERT_EQ(im2.GetMetaRecord("name").idType, kUInt32);
  ASSERT_EQ(im2.GetMetaRecord("name").valueType, kString);
  ASSERT_EQ(im2.GetMetaRecord("age").valueType, kUInt32);
  ASSERT_EQ(im2.GetIndexNum(), 2);
}

TEST(IndexMeta, SerializeFileIO) {
  {
    std::unique_ptr<FileIO> writer;
    ASSERT_TRUE(Env::Default()->NewFileIO(
        "index_meta_test.dat", false, &writer).ok());

    IndexMeta im;

    IndexMetaRecord record;
    record.type = HASHINDEX;
    record.idType = kUInt32;
    record.valueType = kString;
    ASSERT_TRUE(im.AddMeta("name", record));

    IndexMetaRecord record2;
    record2.type = RANGEINDEX;
    record2.idType = kUInt32;
    record2.valueType = kUInt32;
    ASSERT_TRUE(im.AddMeta("age", record2));

    ASSERT_TRUE(im.Serialize(writer.get()));
  }

  {
    std::unique_ptr<FileIO> reader;
    ASSERT_TRUE(Env::Default()->NewFileIO(
        "index_meta_test.dat", true, &reader).ok());

    IndexMeta im;
    ASSERT_TRUE(im.Deserialize(reader.get()));
    if (system("rm -f index_meta_test.dat") != 0) {
      return;
    }

    ASSERT_TRUE(im.HasIndex("age"));
    ASSERT_TRUE(im.HasIndex("name"));
    ASSERT_EQ(im.GetMetaRecord("age").type, RANGEINDEX);
    ASSERT_EQ(im.GetMetaRecord("name").type, HASHINDEX);
    ASSERT_EQ(im.GetMetaRecord("name").idType, kUInt32);
    ASSERT_EQ(im.GetMetaRecord("name").valueType, kString);
    ASSERT_EQ(im.GetMetaRecord("age").valueType, kUInt32);
    ASSERT_EQ(im.GetIndexNum(), 2);
  }
}

}  // namespace euler
