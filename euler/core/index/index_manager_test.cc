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
#include <vector>

#include "gtest/gtest.h"

#include "euler/core/index/index_manager.h"
#include "euler/common/env.h"

using std::vector;
using std::string;

namespace euler {

TEST(IndexManagerTest, Deserialize) {
  IndexManager im;
  im.Deserialize("/tmp/euler/Index");
  {
    auto index = im.GetIndex("edge_type");
    auto v = index->Search(EQ, "1");
    auto r = v->GetIds();
    ASSERT_EQ(r.size(), 7);
  }
  {
    auto index = im.GetIndex("edge_value");
    auto v = index->Search(LESS, "30");
    auto r = v->GetIds();
    ASSERT_EQ(r.size(), 5);
  }
  {
    auto index = im.GetIndex("node_type");
    auto v = index->Search(EQ, "0");
    auto r = v->GetIds();
    ASSERT_EQ(r.size(), 3);
  }
  {
    auto index = im.GetIndex("price");
    auto v = index->Search(LESS, "5");
    v = v->Intersection(index->Search(GREATER, "2"));
    auto r = v->GetIds();
    ASSERT_EQ(r.size(), 3);
  }
  {
    auto index = im.GetIndex("att");
    auto v = index->Search(LESS, "1::5");
    auto r = v->GetIds();
    ASSERT_EQ(r.size(), 3);
  }
  {
    auto index = im.GetIndex("edge_att");
    auto v = index->Search(LESS, "1::14");
    auto r = v->GetIds();
    ASSERT_EQ(r.size(), 2);
  }
}

}  // namespace euler

