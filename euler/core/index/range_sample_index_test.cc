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
#include <memory>

#include "gtest/gtest.h"

#include "euler/core/index/range_sample_index.h"
#include "euler/common/env.h"

using std::vector;
using std::string;
using std::shared_ptr;

namespace euler {

TEST(RangeSampleIndexTest, Init) {
  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> values = {1.0, 2.0, 2.0, 3.0, 4.0};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  ASSERT_TRUE(rsIndex.Init(ids, values, weights));
}

TEST(RangeSampleIndexTest, Merge) {
  RangeSampleIndex<int32_t, float> index1("range");
  {
    vector<int32_t> ids = {0, 1, 2};
    vector<float> values = {1.0, 2.0, 2.0};
    vector<float> weights = {2.0, 4.0, 4.0};
    ASSERT_TRUE(index1.Init(ids, values, weights));
  }
  vector<shared_ptr<SampleIndex>> vr;
  {
    vector<int32_t> ids = {3, 4};
    vector<float> values = {3.0, 4.0};
    vector<float> weights = {2.0, 2.0};
    RangeSampleIndex<int32_t, float>* index =
      new RangeSampleIndex<int32_t, float>("range");
    index->Init(ids, values, weights);
    vr.emplace_back(shared_ptr<SampleIndex>(index));
  }
  {
    vector<int32_t> ids = {5, 6};
    vector<float> values = {3.0, 4.0};
    vector<float> weights = {1.0, 1.0};
    RangeSampleIndex<int32_t, float>* index =
      new RangeSampleIndex<int32_t, float>("range");
    index->Init(ids, values, weights);
    vr.emplace_back(shared_ptr<SampleIndex>(index));
  }
  ASSERT_TRUE(index1.Merge(vr[0]));
  {
    auto v = index1.Search(GREATER_EQ, "1.0");
    v = v->Intersection(index1.Search(EQ, "3"));
    auto ids = v->GetIds();
    ASSERT_EQ(ids.size(), 1);
    ASSERT_EQ(ids[0], 3);

    auto v2 = index1.Search(GREATER, "2");
    v2 = v2->Intersection(index1.Search(LESS, "5"));
    auto r = v2->Sample(100000);

    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_TRUE(cnts[3]*1.0/cnts[4] > 0.9 && cnts[3]*1.0/cnts[4] < 1.1);
  }

  RangeSampleIndex<int32_t, float> index("range");
  index.Merge(vr);
  {
    auto v = index.Search(GREATER, "2");
    v = v->Intersection(index.Search(LESS, "5"));
    v = v->Intersection(index.Search(NOT_EQ, "4"));
    auto r = v->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[3]*1.0/cnts[5] > 1.9 && cnts[3]*1.0/cnts[5] < 2.1);
  }
}

TEST(RangeSampleIndexTest, DeserializeFileIO) {
  std::string filename = "range_sample_test.dat";

  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> values = {1.0, 2.0, 2.0, 3.0, 4.0};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  rsIndex.Init(ids, values, weights);

  {
    std::unique_ptr<FileIO> writer;
    ASSERT_TRUE(Env::Default()->NewFileIO(filename, false, &writer).ok());
    ASSERT_TRUE(rsIndex.Serialize(writer.get()));
  }

  RangeSampleIndex<int32_t, float> newIndex("range");
  {
    std::unique_ptr<FileIO> reader;
    ASSERT_TRUE(Env::Default()->NewFileIO(filename, true, &reader).ok());
    ASSERT_TRUE(newIndex.Deserialize(reader.get()));
  }
  if (system("rm -f range_sample_test.dat")) {
    return;
  }

  auto v = newIndex.Search(GREATER_EQ, "2.0");
  v = v->Intersection(newIndex.Search(LESS, "3.5"));
  auto r = v->Sample(100000);
  ASSERT_EQ(r.size(), 100000);

  vector<int32_t> cnts(5);
  for (auto& i : r) {
    cnts[i.first] += 1;
    ASSERT_EQ(i.second, weights[i.first]);
  }
  ASSERT_EQ(cnts[0], 0);
  ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
  ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
  ASSERT_EQ(cnts[4], 0);
}

TEST(RangeSampleIndexTest, SearchAll) {
  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> values = {1.0, 2.0, 2.0, 3.0, 4.0};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  rsIndex.Init(ids, values, weights);
  auto v = rsIndex.Search(LESS, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  v = rsIndex.Search(LESS_EQ, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 3);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[2], 2);

  v = rsIndex.Search(GREATER, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 3);
  ASSERT_EQ(v[1], 4);

  v = rsIndex.Search(GREATER_EQ, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 4);

  v = rsIndex.Search(EQ, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);

  v = rsIndex.Search(NOT_EQ, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 3);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 3);
  ASSERT_EQ(v[2], 4);

  v = rsIndex.Search(IN, "2.0::3.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 3);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);
  ASSERT_EQ(v[2], 3);

  v = rsIndex.Search(IN, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 1);
  ASSERT_EQ(v[1], 2);

  v = rsIndex.Search(NOT_IN, "2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 3);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 3);
  ASSERT_EQ(v[2], 4);

  v = rsIndex.Search(NOT_IN, "3.0::2.0")->GetSortedIds();
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 4);
}

TEST(RangeSampleIndexTest, Search) {
  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> init_values = {1.0, 2.0, 2.0, 3.0, 4.0};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  rsIndex.Init(ids, init_values, weights);

  auto v = rsIndex.Search(GREATER, "2.0");
  v = v->Intersection(rsIndex.Search(LESS_EQ, "5.0"));
  auto r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 2);
  ASSERT_EQ(r[0], 3);
  ASSERT_EQ(r[1], 4);

  v = rsIndex.Search(LESS, "3");
  v = v->Intersection(rsIndex.Search(GREATER, "1"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 2);
  ASSERT_EQ(r[0], 1);
  ASSERT_EQ(r[1], 2);

  v = rsIndex.Search(GREATER, "3");
  v = v->Intersection(rsIndex.Search(EQ, "1"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 0);

  v = rsIndex.Search(GREATER, "5");
  v = v->Intersection(rsIndex.Search(LESS, "1"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 0);

  v = rsIndex.Search(GREATER, "1");
  v = v->Intersection(rsIndex.Search(NOT_EQ, "3"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 3);
  ASSERT_EQ(r[0], 1);
  ASSERT_EQ(r[1], 2);
  ASSERT_EQ(r[2], 4);

  v = rsIndex.Search(GREATER, "1");
  v = v->Intersection(rsIndex.Search(EQ, "3"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 1);
  ASSERT_EQ(r[0], 3);

  v = rsIndex.Search(GREATER, "1");
  v = v->Intersection(rsIndex.Search(NOT_EQ, "2"));
  v = v->Intersection(rsIndex.Search(LESS, "5"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 2);
  ASSERT_EQ(r[0], 3);
  ASSERT_EQ(r[1], 4);

  v = rsIndex.Search(GREATER, "1");
  v = v->Intersection(rsIndex.Search(NOT_EQ, "2"));
  v = v->Intersection(rsIndex.Search(NOT_EQ, "3"));
  r = v->GetSortedIds();
  ASSERT_EQ(r.size(), 1);
  ASSERT_EQ(r[0], 4);
}

TEST(RangeSampleIndexTest, SearchSample) {
  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> init_values = {1.0, 2.0, 2.0, 3.0, 4.0};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  rsIndex.Init(ids, init_values, weights);

  // only one result
  {
    auto v = rsIndex.Search(LESS_EQ, "1.5");
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 10000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[0], 10000);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[4], 0);
  }
  {
    auto v = rsIndex.Search(LESS, "1.5");
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 10000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[0], 10000);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[4], 0);
  }
  {
    auto v = rsIndex.Search(GREATER, "3");
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 10000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[4], 10000);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[0], 0);
  }
  {
    auto v = rsIndex.Search(GREATER_EQ, "4");
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 10000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[4], 10000);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[0], 0);
  }
  {
    auto v = rsIndex.Search(EQ, "4");
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 10000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[4], 10000);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[0], 0);
  }
  {
    RangeSampleIndex<int32_t, float> rsIndex2("range");
    vector<int32_t> ids = {0, 1, 2};
    vector<float> init_values = {1.0, 1.0, 2.0};
    vector<float> weights = {2.0, 4.0, 8.0};
    rsIndex2.Init(ids, init_values, weights);
    auto v = rsIndex2.Search(NOT_EQ, "1");
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 10000);
    vector<int32_t> cnts(3);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 10000);
    ASSERT_EQ(cnts[0], 0);
  }
  // empty result
  {
    auto v = rsIndex.Search(LESS, "0.5");
    auto rr = v->GetIds();
    ASSERT_EQ(rr.size(), 0);
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 0);
  }
  {
    auto v = rsIndex.Search(LESS_EQ, "0.5");
    auto rr = v->GetIds();
    ASSERT_EQ(rr.size(), 0);
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 0);
  }
  {
    auto v = rsIndex.Search(GREATER, "5");
    auto rr = v->GetIds();
    ASSERT_EQ(rr.size(), 0);
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 0);
  }
  {
    auto v = rsIndex.Search(GREATER_EQ, "5");
    auto rr = v->GetIds();
    ASSERT_EQ(rr.size(), 0);
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 0);
  }
  {
    auto v = rsIndex.Search(EQ, "5");
    auto rr = v->GetIds();
    ASSERT_EQ(rr.size(), 0);
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 0);
  }
  {
    RangeSampleIndex<int32_t, float> rsIndex2("range");
    vector<int32_t> ids = {0, 1, 2};
    vector<float> init_values = {1.0, 1.0, 1.0};
    vector<float> weights = {2.0, 4.0, 8.0};
    rsIndex2.Init(ids, init_values, weights);
    auto v = rsIndex2.Search(NOT_EQ, "1");
    auto rr = v->GetIds();
    ASSERT_EQ(rr.size(), 0);
    auto r = v->Sample(10000);
    ASSERT_EQ(r.size(), 0);
  }
  {
    auto v = rsIndex.Search(GREATER_EQ, "2.0");
    v = v->Intersection(rsIndex.Search(LESS, "3.5"));
    auto r = v->Sample(100000);
    ASSERT_EQ(r.size(), 100000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
      ASSERT_EQ(i.second, weights[i.first]);
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_EQ(cnts[4], 0);
  }
  {
    auto v = rsIndex.Search(GREATER_EQ, "2.0");
    v = v->Intersection(rsIndex.Search(LESS, "4.1"));
    auto r = v->Sample(100000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
  {
    auto v = rsIndex.Search(GREATER, "0.5");
    v = v->Intersection(rsIndex.Search(LESS_EQ, "4"));
    auto r = v->Sample(100000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
  {
    auto v = rsIndex.Search(GREATER, "5");
    v = v->Intersection(rsIndex.Search(LESS_EQ, "1"));
    auto ids = v->GetIds();
    auto r = v->Sample(100);
    ASSERT_EQ(r.size(), 0);
  }
  {
    auto v = rsIndex.Search(GREATER, "1");
    v = v->Intersection(rsIndex.Search(EQ, "3"));
    auto r = v->Sample(100000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 100000);
    ASSERT_EQ(cnts[4], 0);
  }
  {
    auto v = rsIndex.Search(GREATER, "1");
    v = v->Intersection(rsIndex.Search(NOT_EQ, "3"));
    auto r = v->Sample(100000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_TRUE(cnts[4]*1.0/cnts[2] > 1.9 && cnts[4]*1.0/cnts[2] < 2.1);
  }
  {
    auto v = rsIndex.Search(GREATER, "1");
    v = v->Intersection(rsIndex.Search(NOT_EQ, "2"));
    v = v->Intersection(rsIndex.Search(LESS_EQ, "4"));
    auto r = v->Sample(100000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
}

TEST(RangeSampleIndexTest, Union) {
  RangeSampleIndex<int32_t, float> rsIndex("range");
  vector<int32_t> ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  vector<float> init_values = {1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0, 8.0, 4.0, 4.0, 16.0};
  rsIndex.Init(ids, init_values, weights);

  {
    auto v = rsIndex.Search(GREATER_EQ, "2.0");
    v = v->Intersection(rsIndex.Search(LESS, "3.5"));

    auto v2 = rsIndex.Search(GREATER_EQ, "5.0");
    v2 = v2->Intersection(rsIndex.Search(LESS, "7"));
    v = v->Union(v2);
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 5);
    ASSERT_EQ(vv[0], 1);
    ASSERT_EQ(vv[1], 2);
    ASSERT_EQ(vv[2], 3);
    ASSERT_EQ(vv[3], 5);
    ASSERT_EQ(vv[4], 6);

    auto r = v->Sample(1000000);
    ASSERT_EQ(r.size(), 1000000);
    vector<int32_t> cnts(7);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[5]*1.0/cnts[3] > 0.9 && cnts[5]*1.0/cnts[3] < 1.1);
    ASSERT_TRUE(cnts[5]*1.0/cnts[6] > 1.9 && cnts[5]*1.0/cnts[6] < 2.1);
  }
  {
    auto v = rsIndex.Search(GREATER_EQ, "2.0");
    v = v->Intersection(rsIndex.Search(LESS, "3.5"));

    auto v2 = rsIndex.Search(GREATER_EQ, "3.0");
    v2 = v2->Intersection(rsIndex.Search(LESS, "5"));
    v = v->Union(v2);
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 4);
    ASSERT_EQ(vv[0], 1);
    ASSERT_EQ(vv[1], 2);
    ASSERT_EQ(vv[2], 3);
    ASSERT_EQ(vv[3], 4);

    auto r = v->Sample(1000000);
    ASSERT_EQ(r.size(), 1000000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
  {
    auto v = rsIndex.Search(GREATER_EQ, "2.0");
    v = v->Intersection(rsIndex.Search(LESS, "3.5"));

    auto v2 = rsIndex.Search(GREATER_EQ, "4.0");
    v2 = v2->Intersection(rsIndex.Search(LESS, "5"));
    v = v->Union(v2);
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 4);
    ASSERT_EQ(vv[0], 1);
    ASSERT_EQ(vv[1], 2);
    ASSERT_EQ(vv[2], 3);
    ASSERT_EQ(vv[3], 4);

    auto r = v->Sample(1000000);
    ASSERT_EQ(r.size(), 1000000);
    vector<int32_t> cnts(5);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
  {
    auto v = rsIndex.Search(EQ, "2.0");

    auto v2 = rsIndex.Search(NOT_EQ, "4.0");
    v = v->Union(v2);

    auto v3 = rsIndex.Search(GREATER, "6");
    v = v->Union(v3);
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 8);
    ASSERT_EQ(vv[0], 0);
    ASSERT_EQ(vv[1], 1);
    ASSERT_EQ(vv[6], 7);
    ASSERT_EQ(vv[7], 8);

    auto r = v->Sample(1000000);
    ASSERT_EQ(r.size(), 1000000);
    vector<int32_t> cnts(9);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[7]*1.0/cnts[6] > 0.9 && cnts[7]*1.0/cnts[6] < 1.1);
    ASSERT_TRUE(cnts[8]*1.0/cnts[7] > 3.9 && cnts[8]*1.0/cnts[7] < 4.1);
  }
  {
    auto v = rsIndex.Search(NOT_EQ, "4.0");

    auto v2 = rsIndex.Search(GREATER, "2");
    v2 = v2->Intersection(rsIndex.Search(LESS_EQ, "5"));
    v = v->Union(v2);
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 9);

    auto r = v->Sample(1000000);
    ASSERT_EQ(r.size(), 1000000);
    vector<int32_t> cnts(9);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[7]*1.0/cnts[6] > 0.9 && cnts[7]*1.0/cnts[6] < 1.1);
    ASSERT_TRUE(cnts[8]*1.0/cnts[7] > 3.9 && cnts[8]*1.0/cnts[7] < 4.1);
  }
  {
    auto v = rsIndex.Search(NOT_EQ, "4.0");

    auto v2 = rsIndex.Search(GREATER, "3");
    v2 = v2->Intersection(rsIndex.Search(LESS, "5"));
    v = v->Union(v2);
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 9);
  }
}
}  // namespace euler
