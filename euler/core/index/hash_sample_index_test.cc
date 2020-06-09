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

#include "euler/core/index/hash_sample_index.h"
#include "euler/common/env.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::shared_ptr;

namespace euler {

TEST(HashSampleIndexTest, AddItem) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  ASSERT_TRUE(hsIndex.AddItem("name", ids, weights));
  ASSERT_TRUE(hsIndex.AddItem("age", ids, weights));
  ASSERT_FALSE(hsIndex.AddItem("name", ids, weights));
}

TEST(HashSampleIndexTest, Merge) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  {
    vector<int32_t> ids = {0, 1, 2};
    vector<float> weights = {2.0, 4.0, 8.0};
    ASSERT_TRUE(hsIndex.AddItem("name", ids, weights));
  }
  {
    vector<int32_t> ids = {4, 5, 6};
    vector<float> weights = {1.0, 1.0, 2.0};
    ASSERT_TRUE(hsIndex.AddItem("age", ids, weights));
  }

  HashSampleIndex<int32_t, string> newIndex("hash");
  {
    vector<int32_t> ids = {3, 4, 5};
    vector<float> weights = {2.0, 2.0, 1.0};
    ASSERT_TRUE(newIndex.AddItem("name", ids, weights));
  }
  {
    vector<int32_t> ids = {24, 25, 26};
    vector<float> weights = {2.0, 2.0, 4.0};
    ASSERT_TRUE(newIndex.AddItem("xxx", ids, weights));
  }
  ASSERT_TRUE(hsIndex.Merge(newIndex));
  ASSERT_EQ(hsIndex.GetKeys().size(), 3);

  {
    auto r = hsIndex.Search(EQ, "name");
    auto v = r->Sample(300000);
    vector<int32_t> cnts(6);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[3] > 3.9 && cnts[2]*1.0/cnts[3] < 4.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[4] > 0.9 && cnts[3]*1.0/cnts[4] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[5] > 1.9 && cnts[4]*1.0/cnts[5] < 2.1);
  }

  {
    auto r = hsIndex.Search(EQ, "xxx")->Sample(100000);
    vector<int32_t> cnts(3);
    for (auto& i : r) {
      cnts[i.first - 24] += 1;
    }
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 0.9 && cnts[1]*1.0/cnts[0] < 1.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
  }
}

TEST(HashSampleIndexTest, MergeVec) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  {
    vector<int32_t> ids = {0, 1, 2};
    vector<float> weights = {2.0, 1.0, 4.0};
    ASSERT_TRUE(hsIndex.AddItem("name1", ids, weights));
  }
  {
    vector<int32_t> ids = {4, 5, 6};
    vector<float> weights = {1.0, 1.0, 2.0};
    ASSERT_TRUE(hsIndex.AddItem("name2", ids, weights));
  }
  HashSampleIndex<int32_t, string>* index2 =
    new HashSampleIndex<int32_t, string>("hash");
  {
    vector<int32_t> ids = {3, 4, 5};
    vector<float> weights = {2.0, 2.0, 1.0};
    ASSERT_TRUE(index2->AddItem("name1", ids, weights));
  }
  HashSampleIndex<int32_t, string>* index3 =
    new HashSampleIndex<int32_t, string>("hash");
  {
    vector<int32_t> ids = {7, 8};
    vector<float> weights = {2.0, 4.0};
    ASSERT_TRUE(index3->AddItem("name3", ids, weights));
  }
  vector<shared_ptr<SampleIndex>> vh;
  vh.push_back(shared_ptr<SampleIndex>(index2));
  vh.push_back(shared_ptr<SampleIndex>(index3));
  ASSERT_TRUE(hsIndex.Merge(vh));
  ASSERT_EQ(hsIndex.GetKeys().size(), 3);

  {
    auto v = hsIndex.Search(EQ, "name1")->Sample(300000);
    vector<int32_t> cnts(9);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    cout << "distribution should be 2:1:4:2:2:1:0:0:0" << endl;
    cout << "distribution:" << endl;
    for (auto i : cnts) {
      cout << i << " ";
    }
    cout << endl;
    ASSERT_TRUE(cnts[0]*1.0/cnts[1] > 1.9 && cnts[0]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 3.9 && cnts[2]*1.0/cnts[1] < 4.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[3] > 1.9 && cnts[2]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[4] > 0.9 && cnts[3]*1.0/cnts[4] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[5] > 1.9 && cnts[4]*1.0/cnts[5] < 2.1);
    ASSERT_EQ(cnts[6], 0);
    ASSERT_EQ(cnts[7], 0);
    ASSERT_EQ(cnts[8], 0);
  }
  {
    auto v = hsIndex.Search(EQ, "name3")->GetSortedIds();
    ASSERT_EQ(v.size(), 2);
    ASSERT_EQ(v[0], 7);
    ASSERT_EQ(v[1], 8);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name1")->GetSortedIds();
    ASSERT_EQ(v.size(), 5);
    ASSERT_EQ(v[0], 4);
    ASSERT_EQ(v[4], 8);
  }
}

TEST(HashSampleIndexTest, SearchSample) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  hsIndex.AddItem("name", ids, weights);
  auto v = hsIndex.Search(EQ, "name")->Sample(100000);
  vector<int32_t> cnts(5);
  for (auto& i : v) {
    cnts[i.first] += 1;
    ASSERT_EQ(i.second, weights[i.first]);
  }
  ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
  ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
  ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
  ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);

  auto empty = hsIndex.Search(EQ, "age")->Sample(10000);
  ASSERT_EQ(empty.size(), 0);
}

TEST(HashSampleIndexTest, Search) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  vector<int32_t> ids = {0, 1, 2, 3, 4};
  vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  hsIndex.AddItem("name", ids, weights);
  auto v = hsIndex.Search(EQ, "name")->GetIds();
  ASSERT_EQ(v.size(), 5);
  ASSERT_EQ(v[0], 0);

  auto empty = hsIndex.Search(EQ, "age")->GetIds();
  ASSERT_EQ(empty.size(), 0);

  HashSampleIndex<int32_t, int32_t> hsIndex2("hash");
  hsIndex2.AddItem(12, ids, weights);
  auto v2 = hsIndex2.Search(EQ, "12")->GetIds();
  ASSERT_EQ(v2.size(), 5);
  ASSERT_EQ(v2[0], 0);
}

TEST(HashSampleIndexTest, NOTEQ) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  {
    vector<int32_t> ids = {0, 1, 2};
    vector<float> weights = {2.0, 2.0, 4.0};
    hsIndex.AddItem("name", ids, weights);
  }
  {
    vector<int32_t> ids = {3, 4};
    vector<float> weights = {2.0, 4.0};
    hsIndex.AddItem("name2", ids, weights);
  }
  {
    vector<int32_t> ids = {5, 6};
    vector<float> weights = {4.0, 4.0};
    hsIndex.AddItem("name3", ids, weights);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name")->GetIds();
    ASSERT_EQ(v.size(), 4);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name2")->GetSortedIds();
    ASSERT_EQ(v.size(), 5);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name")->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[5]*1.0/cnts[4] > 0.9 && cnts[5]*1.0/cnts[4] < 1.1);
    ASSERT_TRUE(cnts[6]*1.0/cnts[5] > 0.9 && cnts[6]*1.0/cnts[5] < 1.1);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name3")->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[5], 0);
    ASSERT_EQ(cnts[6], 0);
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 0.9 && cnts[1]*1.0/cnts[0] < 1.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[3] > 1.9 && cnts[2]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name");
    v = v->Intersection(hsIndex.Search(EQ, "name2"));
    auto r = v->GetSortedIds();
    ASSERT_EQ(r.size(), 2);
    ASSERT_EQ(r[0], 3);
    ASSERT_EQ(r[1], 4);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name");
    v = v->Intersection(hsIndex.Search(NOT_EQ, "name2"));
    auto r = v->GetSortedIds();
    ASSERT_EQ(r.size(), 2);
    ASSERT_EQ(r[0], 5);
    ASSERT_EQ(r[1], 6);

    auto v3 = v->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v3) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[4], 0);
    ASSERT_TRUE(cnts[6]*1.0/cnts[5] > 0.9 && cnts[6]*1.0/cnts[5] < 1.1);
  }
}

TEST(HashSampleIndexTest, IN) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  {
    vector<int32_t> ids = {0, 1, 2};
    vector<float> weights = {2.0, 2.0, 4.0};
    hsIndex.AddItem("name", ids, weights);
  }
  {
    vector<int32_t> ids = {3, 4};
    vector<float> weights = {2.0, 4.0};
    hsIndex.AddItem("name2", ids, weights);
  }
  {
    vector<int32_t> ids = {5, 6};
    vector<float> weights = {4.0, 4.0};
    hsIndex.AddItem("name3", ids, weights);
  }
  {
    auto v = hsIndex.Search(IN, "name")->GetIds();
    ASSERT_EQ(v.size(), 3);
  }
  {
    auto v = hsIndex.Search(IN, "name2")->GetSortedIds();
    ASSERT_EQ(v.size(), 2);
  }
  {
    auto v = hsIndex.Search(IN, "name::name2")->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[5], 0);
    ASSERT_EQ(cnts[6], 0);
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 0.9 && cnts[1]*1.0/cnts[0] < 1.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[3] > 1.9 && cnts[2]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
  }
  {
    auto v = hsIndex.Search(IN, "name3::name2")->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[5]*1.0/cnts[4] > 0.9 && cnts[5]*1.0/cnts[4] < 1.1);
    ASSERT_TRUE(cnts[6]*1.0/cnts[5] > 0.9 && cnts[6]*1.0/cnts[5] < 1.1);
  }
}

TEST(HashSampleIndexTest, NOT_IN) {
  HashSampleIndex<int32_t, int32_t> hsIndex("hash");
  {
    vector<int32_t> ids = {0, 1, 2};
    vector<float> weights = {2.0, 2.0, 4.0};
    hsIndex.AddItem(100, ids, weights);
  }
  {
    vector<int32_t> ids = {3, 4};
    vector<float> weights = {2.0, 4.0};
    hsIndex.AddItem(200, ids, weights);
  }
  {
    vector<int32_t> ids = {5, 6};
    vector<float> weights = {4.0, 4.0};
    hsIndex.AddItem(300, ids, weights);
  }
  {
    auto v = hsIndex.Search(NOT_IN, "100")->GetIds();
    ASSERT_EQ(v.size(), 4);
  }
  {
    auto v = hsIndex.Search(NOT_IN, "200")->GetSortedIds();
    ASSERT_EQ(v.size(), 5);
  }
  {
    auto v = hsIndex.Search(NOT_IN, "100::200")->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    ASSERT_EQ(cnts[0], 0);
    ASSERT_EQ(cnts[1], 0);
    ASSERT_EQ(cnts[2], 0);
    ASSERT_EQ(cnts[3], 0);
    ASSERT_EQ(cnts[4], 0);
    ASSERT_TRUE(cnts[6]*1.0/cnts[5] > 0.9 && cnts[6]*1.0/cnts[5] < 1.1);
  }
  {
    auto v = hsIndex.Search(NOT_IN, "100::200::300")->Sample(100000);
    vector<int32_t> cnts(7);
    for (auto& i : v) {
      cnts[i.first] += 1;
    }
    for (auto& i : cnts) {
      ASSERT_EQ(i, 0);
    }
  }
}

TEST(HashSampleIndexTest, DeserializeFileIO) {
  std::string filename = "hash_sample_test.dat";
  HashSampleIndex<int32_t, string> hsIndex("hash");
  vector<int32_t> ids = {0, 1, 2};
  vector<float> weights = {2.0, 4.0, 8.0};
  hsIndex.AddItem("name", ids, weights);
  hsIndex.AddItem("name2", ids, weights);
  {
    std::unique_ptr<FileIO> writer;
    ASSERT_TRUE(Env::Default()->NewFileIO(filename, false, &writer).ok());
    ASSERT_TRUE(hsIndex.Serialize(writer.get()));
  }

  HashSampleIndex<int32_t, string> hsIndex2("hash");
  {
    std::unique_ptr<FileIO> reader;
    ASSERT_TRUE(Env::Default()->NewFileIO(filename, true, &reader).ok());
    ASSERT_TRUE(hsIndex2.Deserialize(reader.get()));
  }
  if (system("rm -f hash_sample_test.dat") < 0) {
    return;
  }

  auto v = hsIndex2.Search(EQ, "name")->Sample(100000);
  vector<int32_t> cnts(3);
  for (auto& i : v) {
    cnts[i.first] += 1;
    ASSERT_EQ(i.second, weights[i.first]);
  }
  ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
  ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
}

TEST(HashSampleIndexTest, Union) {
  HashSampleIndex<int32_t, string> hsIndex("hash");
  {
    vector<int32_t> ids = {0, 1, 2, 3, 4};
    vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
    hsIndex.AddItem("name", ids, weights);
  }
  {
    vector<int32_t> ids = {5, 6, 7};
    vector<float> weights = {4.0, 8.0, 8.0};
    hsIndex.AddItem("name2", ids, weights);
  }
  {
    vector<int32_t> ids = {9, 10};
    vector<float> weights = {4.0, 4.0};
    hsIndex.AddItem("name3", ids, weights);
  }
  {
    auto v = hsIndex.Search(NOT_EQ, "name");
    v = v->Union(hsIndex.Search(EQ, "name2"));
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 5);
    ASSERT_EQ(vv[0], 5);
    ASSERT_EQ(vv[4], 10);
    auto r = v->Sample(100000);
    vector<int32_t> cnts(11);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[6]*1.0/cnts[5] > 1.9 && cnts[6]*1.0/cnts[5] < 2.1);
    ASSERT_TRUE(cnts[7]*1.0/cnts[6] > 0.9 && cnts[7]*1.0/cnts[6] < 1.1);
    ASSERT_TRUE(cnts[7]*1.0/cnts[9] > 1.9 && cnts[7]*1.0/cnts[9] < 2.1);
    ASSERT_TRUE(cnts[10]*1.0/cnts[9] > 0.9 && cnts[10]*1.0/cnts[9] < 1.1);
  }
  {
    auto v = hsIndex.Search(EQ, "name");
    v = v->Union(hsIndex.Search(EQ, "name2"));
    auto vv = v->GetSortedIds();
    ASSERT_EQ(vv.size(), 8);
    ASSERT_EQ(vv[0], 0);
    ASSERT_EQ(vv[7], 7);
    auto r = v->Sample(1000000);
    vector<int32_t> cnts(8);
    for (auto& i : r) {
      cnts[i.first] += 1;
    }
    ASSERT_TRUE(cnts[1]*1.0/cnts[0] > 1.9 && cnts[1]*1.0/cnts[0] < 2.1);
    ASSERT_TRUE(cnts[2]*1.0/cnts[1] > 1.9 && cnts[2]*1.0/cnts[1] < 2.1);
    ASSERT_TRUE(cnts[3]*1.0/cnts[2] > 0.9 && cnts[3]*1.0/cnts[2] < 1.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[3] > 1.9 && cnts[4]*1.0/cnts[3] < 2.1);
    ASSERT_TRUE(cnts[4]*1.0/cnts[5] > 3.9 && cnts[4]*1.0/cnts[5] < 4.1);
    ASSERT_TRUE(cnts[6]*1.0/cnts[5] > 1.9 && cnts[6]*1.0/cnts[5] < 2.1);
    ASSERT_TRUE(cnts[7]*1.0/cnts[6] > 0.9 && cnts[7]*1.0/cnts[6] < 1.1);
  }
}

}  // namespace euler
