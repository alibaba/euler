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

#include "euler/core/index/hash_range_sample_index.h"
#include "euler/common/env.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

namespace euler {

TEST(HashSampleIndexTest, AddItem) {
  std::string filename = "hash_range_sample_test.dat";
  HashRangeSampleIndex<int32_t, float> hrIndex("hashrange");
  {
    HashRangeSampleIndex<int32_t, float>::RangeIndex rsIndex(
                      new RangeSampleIndex<int32_t, float>("range1"));
    vector<int32_t> ids = {0, 1, 2};
    vector<float> values = {1.0, 2.0, 2.0};
    vector<float> weights = {2.0, 4.0, 8.0};
    ASSERT_TRUE(rsIndex->Init(ids, values, weights));
    ASSERT_TRUE(hrIndex.AddItem(10, rsIndex));
  }
  {
    HashRangeSampleIndex<int32_t, float>::RangeIndex rsIndex(
                      new RangeSampleIndex<int32_t, float>("range2"));
    vector<int32_t> ids = {2, 3, 4, 5};
    vector<float> values = {2.0, 3.0, 4.0, 5.0};
    vector<float> weights = {2.0, 4.0, 4.0, 8.0};
    ASSERT_TRUE(rsIndex->Init(ids, values, weights));
    ASSERT_TRUE(hrIndex.AddItem(11, rsIndex));
  }
  {
    std::unique_ptr<FileIO> writer;
    ASSERT_TRUE(Env::Default()->NewFileIO(filename, false, &writer).ok());
    ASSERT_TRUE(hrIndex.Serialize(writer.get()));
  }
  HashRangeSampleIndex<int32_t, float> hrIndex2("hashrange2");
  {
    std::unique_ptr<FileIO> reader;
    ASSERT_TRUE(Env::Default()->NewFileIO(filename, true, &reader).ok());
    ASSERT_TRUE(hrIndex2.Deserialize(reader.get()));
  }
  system("rm -f hash_range_sample_test.dat");

  {
    auto v = hrIndex2.Search(GREATER_EQ, "11::4");
    auto ids = v->GetIds();
    ASSERT_EQ(ids.size(), 2);
    ASSERT_EQ(ids[0], 4);
    ASSERT_EQ(ids[1], 5);
    auto r = v->Sample(100000);

    vector<int32_t> cnts(6);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[5]*1.0/cnts[4] > 1.9 && cnts[5]*1.0/cnts[4] < 2.1);
  }
  {
    auto v = hrIndex2.Search(LESS, "10::2");
    auto ids = v->GetIds();
    ASSERT_EQ(ids.size(), 1);
    ASSERT_EQ(ids[0], 0);
    auto r = v->Sample(100000);

    for (auto& i : r) {
      ASSERT_EQ(i.first, 0);
    }
  }
}

TEST(HashSampleIndexTest, Merge) {
  HashRangeSampleIndex<int32_t, float> hrIndex("hashrange");
  {
    HashRangeSampleIndex<int32_t, float>::RangeIndex rsIndex(
                      new RangeSampleIndex<int32_t, float>("range1"));
    vector<int32_t> ids = {0, 1, };
    vector<float> values = {1.0, 2.0};
    vector<float> weights = {2.0, 4.0};
    ASSERT_TRUE(rsIndex->Init(ids, values, weights));
    ASSERT_TRUE(hrIndex.AddItem(10, rsIndex));
  }
  {
    HashRangeSampleIndex<int32_t, float>::RangeIndex rsIndex(
                      new RangeSampleIndex<int32_t, float>("range2"));
    vector<int32_t> ids = {2, 3};
    vector<float> values = {2.0, 3.0};
    vector<float> weights = {2.0, 4.0};
    ASSERT_TRUE(rsIndex->Init(ids, values, weights));
    ASSERT_TRUE(hrIndex.AddItem(11, rsIndex));
  }

  HashRangeSampleIndex<int32_t, float> hrIndex2("hashrange2");
  {
    HashRangeSampleIndex<int32_t, float>::RangeIndex rsIndex(
                      new RangeSampleIndex<int32_t, float>("range1"));
    vector<int32_t> ids = {2, 3, };
    vector<float> values = {3.0, 4.0};
    vector<float> weights = {2.0, 4.0};
    ASSERT_TRUE(rsIndex->Init(ids, values, weights));
    ASSERT_TRUE(hrIndex2.AddItem(10, rsIndex));
  }
  {
    HashRangeSampleIndex<int32_t, float>::RangeIndex rsIndex(
                      new RangeSampleIndex<int32_t, float>("range1"));
    vector<int32_t> ids = {0, 1, };
    vector<float> values = {1.0, 2.0};
    vector<float> weights = {2.0, 4.0};
    ASSERT_TRUE(rsIndex->Init(ids, values, weights));
    ASSERT_TRUE(hrIndex2.AddItem(12, rsIndex));
  }
  hrIndex2.Merge(hrIndex);
  {
    auto v = hrIndex2.Search(GREATER_EQ, "10::1");
    auto ids = v->GetIds();
    ASSERT_EQ(ids.size(), 4);
    auto r = v->Sample(100000);

    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 1.9 && cnts[3]*1.0/cnts[2] < 2.1);
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
  }
  {
    auto v = hrIndex2.Search(GREATER_EQ, "12::1");
    auto ids = v->GetIds();
    ASSERT_EQ(ids.size(), 2);
    auto r = v->Sample(100000);

    vector<int32_t> cnts(2);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
  }
}

}  // namespace euler
