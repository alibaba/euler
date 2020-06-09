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

#include "euler/core/index/common_index_result.h"
#include "euler/core/index/hash_sample_index.h"
#include "euler/core/index/range_sample_index.h"

namespace euler {

using std::vector;
using std::string;
using std::pair;
using std::make_pair;

TEST(RangeIndexResultTest, Func) {
  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> values = {1.0, 2.0, 2.0, 3.0, 4.0};
  vector<float> weights = {3.0, 1.0, 2.0, 1.0, 1.0};
  rsIndex.Init(ids, values, weights);
  {
    auto r = rsIndex.Search(LESS, "3");
    {
      auto v = r->GetIds();
      ASSERT_EQ(v.size(), 3);
      ASSERT_EQ(v[0], 0);
      ASSERT_EQ(v[1], 1);
      ASSERT_EQ(v[2], 2);
      auto f = r->SumWeight();
      ASSERT_EQ(f, 6);
    }
    auto c = r->ToCommonIndexResult();
    {
      auto v = c->GetIds();
      ASSERT_EQ(v.size(), 3);
      ASSERT_EQ(v[0], 0);
      ASSERT_EQ(v[1], 1);
      ASSERT_EQ(v[2], 2);
      auto f = c->SumWeight();
      ASSERT_EQ(f, 6);
    }
    {
      vector<pair<uint64_t, float>> data;
      data.push_back(make_pair(1, 1.0));
      data.push_back(make_pair(2, 2.0));
      data.push_back(make_pair(4, 1.0));
      CommonIndexResult* c = new CommonIndexResult("common", data);
      std::shared_ptr<IndexResult> c_ptr(c);
      {
        auto result = r->Intersection(c_ptr);
        auto v = result->GetIds();
        ASSERT_EQ(v.size(), 2);
        ASSERT_EQ(v[0], 1);
        ASSERT_EQ(v[1], 2);
        auto f = result->SumWeight();
        ASSERT_EQ(f, 3);
        auto v2 = result->Sample(100000);
        vector<int32_t> cnts(5);
        for (auto& i : v2) {
          cnts[i.first] += 1;
        }
        ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
      }
      {
        auto result = r->Union(c_ptr);
        auto v = result->GetIds();
        ASSERT_EQ(v.size(), 4);
        ASSERT_EQ(v[0], 0);
        ASSERT_EQ(v[1], 1);
        ASSERT_EQ(v[3], 4);
        auto f = result->SumWeight();
        ASSERT_EQ(f, 7);
        auto v2 = result->Sample(100000);
        vector<int32_t> cnts(5);
        for (auto& i : v2) {
          cnts[i.first] += 1;
        }
        ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
        ASSERT_TRUE(cnts[2]*1.0/cnts[4] > 1.9 && cnts[2]*1.0/cnts[4] < 2.1);
        ASSERT_TRUE(cnts[0]*1.0/cnts[4] > 2.9 && cnts[0]*1.0/cnts[4] < 3.1);
      }
    }
    {
      vector<pair<uint64_t, float>> data;
      data.push_back(make_pair(0, 1.0));
      data.push_back(make_pair(1, 1.0));
      data.push_back(make_pair(2, 2.0));
      data.push_back(make_pair(4, 1.0));
      CommonIndexResult* c = new CommonIndexResult("common", data);
      std::shared_ptr<IndexResult> c_ptr(c);
      {
        auto r = rsIndex.Search(LESS, "2");
        r = r->Union(rsIndex.Search(GREATER, "3"));
        auto result = r->Intersection(c_ptr);
        auto v = result->GetIds();
        ASSERT_EQ(v.size(), 2);
        ASSERT_EQ(v[0], 0);
        ASSERT_EQ(v[1], 4);
        auto f = result->SumWeight();
        ASSERT_EQ(f, 2);
        auto v2 = result->Sample(100000);
        vector<int32_t> cnts(5);
        for (auto& i : v2) {
          cnts[i.first] += 1;
        }
        ASSERT_TRUE(cnts[4]*1.0/cnts[0] > 0.9 && cnts[4]*1.0/cnts[0] < 1.1);
      }
    }
  }
  {
    HashSampleIndex<int32_t, string> hsIndex("hash");
    {
      vector<int32_t> ids = {2, 4};
      vector<float> weights = {2.0, 1.0};
      ASSERT_TRUE(hsIndex.AddItem("name", ids, weights));
    }
    {
      vector<int32_t> ids = {1, 5};
      vector<float> weights = {1.0, 2.0};
      ASSERT_TRUE(hsIndex.AddItem("age", ids, weights));
    }

    auto h = hsIndex.Search(EQ, "name");
    auto r = rsIndex.Search(GREATER_EQ, "3");
    auto result = h->Intersection(r);
    auto v = result->GetIds();
    ASSERT_EQ(v.size(), 1);
    ASSERT_EQ(v[0], 4);
    auto f = result->SumWeight();
    ASSERT_EQ(f, 1);

    result = h->Union(r);
    v = result->GetSortedIds();
    ASSERT_EQ(v.size(), 3);
    ASSERT_EQ(v[0], 2);
    ASSERT_EQ(v[2], 4);
    f = result->SumWeight();
    ASSERT_EQ(f, 4);
    auto v2 = result->Sample(100000);
    vector<int32_t> cnts(5);
    for (auto& i : v2) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[2]*1.0/cnts[3] > 1.9 && cnts[2]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[4] > 1.9 && cnts[2]*1.0/cnts[4] < 2.1);
  }
}

}  // namespace euler
