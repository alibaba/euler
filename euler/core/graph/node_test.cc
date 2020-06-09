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

#include <vector>
#include <string>
#include <iostream>

#include "gtest/gtest.h"

#include "euler/core/graph/node.h"

namespace euler {

Node* GetNode() {
  Node* fn = new Node(1, 1.0, 0);
  std::vector<std::vector<uint64_t>> neighbor_ids;
  std::vector<std::vector<float>> neighbor_weight;
  {
    std::vector<uint64_t> id{1, 3, 5};
    std::vector<float> w{1, 3, 1};
    neighbor_ids.push_back(id);
    neighbor_weight.push_back(w);
  }
  {
    std::vector<uint64_t> id{2, 4};
    std::vector<float> w{0.2, 0.4};
    neighbor_ids.push_back(id);
    neighbor_weight.push_back(w);
  }
  {
    std::vector<uint64_t> id{10, 11, 12, 13};
    std::vector<float> w{1, 1, 2, 2};
    neighbor_ids.push_back(id);
    neighbor_weight.push_back(w);
  }
  std::vector<std::vector<uint64_t>> uint64_features;
  {
    uint64_features.push_back({10001, 10002, 10003});
    uint64_features.push_back({20001, 20002, 20003, 20004});
    uint64_features.push_back({30001, 30002});
  }
  std::vector<std::vector<float>> float_features;
  {
    float_features.push_back({1.1, 1.2});
    float_features.push_back({2.1, 2.2, 2.3});
  }
  std::vector<std::string> binary_features;
  {
    binary_features.push_back("iam1");
    binary_features.push_back("iam22");
    binary_features.push_back("iam333");
  }
  if (fn->Init(neighbor_ids, neighbor_weight,
               uint64_features, float_features,
               binary_features)) {
    return fn;
  }
    return nullptr;
}


#define CHECK_PAIR_VEC(V1, V2, V3) {         \
  ASSERT_EQ(V1.size(), V2.size());           \
  for (size_t i = 0; i < V1.size(); ++i) {   \
    ASSERT_EQ(std::get<0>(V1[i]), V2[i]);    \
    ASSERT_EQ(std::get<1>(V1[i]), V3[i]);    \
  }                                          \
}

#define CHECK_VEC_VEC(V1, V2) {                 \
  ASSERT_EQ(V1.size(), V2.size());              \
  for (size_t i = 0; i < V1.size(); ++i) {      \
    ASSERT_EQ(V1[i].size(), V2[i].size());      \
    for (size_t j = 0; j < V1[i].size(); ++j) { \
    ASSERT_EQ(V1[i][j], V2[i][j]);              \
    }                                           \
  }                                             \
}

TEST(NodeTest, FastNodeFunction) {
  Node* fn = GetNode();
  ASSERT_TRUE(fn != nullptr);
  std::string s;
  ASSERT_TRUE(fn->Serialize(&s));
  Node fn2;
  ASSERT_TRUE(fn2.DeSerialize(s.c_str(), s.length()));
  {
    std::vector<int32_t> edge_types{0};
    auto r = fn2.GetFullNeighbor(edge_types);
    std::vector<uint64_t> neighbor{1, 3, 5};
    std::vector<float> weight{1, 3, 1};
    CHECK_PAIR_VEC(r, neighbor, weight);
  }
  ASSERT_EQ(fn2.GetFloat32FeatureValueNum(), 3);
  ASSERT_EQ(fn2.GetUint64FeatureValueNum(), 4);
  ASSERT_EQ(fn2.GetBinaryFeatureValueNum(), 6);
  {
    std::vector<int32_t> fids{0, 1};
    std::vector<std::vector<uint64_t>> r;
    fn2.GetUint64Feature(fids, &r);
    std::vector<std::vector<uint64_t>> expect;
    expect.push_back({10001, 10002, 10003});
    expect.push_back({20001, 20002, 20003, 20004});
    CHECK_VEC_VEC(r, expect);
  }
  {
    std::vector<int32_t> fids{0, 1};
    std::vector<std::vector<float>> r;
    fn2.GetFloat32Feature(fids, &r);
    std::vector<std::vector<float>> expect;
    expect.push_back({1.1, 1.2});
    expect.push_back({2.1, 2.2, 2.3});
    CHECK_VEC_VEC(r, expect);
  }
  {
    std::vector<int32_t> fids{2};
    std::vector<std::string> r;
    fn2.GetBinaryFeature(fids, &r);
    ASSERT_EQ(r.size(), 1);
    ASSERT_EQ(r[0], "iam333");
  }
  {
    std::vector<int32_t> edge_types{2};
    auto r = fn2.SampleNeighbor(edge_types, 100000);
    ASSERT_EQ(r.size(), 100000);
    std::vector<int32_t> cnts(4);
    for (auto i : r) {
      cnts[std::get<0>(i) - 10] += 1;
    }
    ASSERT_TRUE(cnts[1] * 1.0 / cnts[0] > 0.8 && cnts[1] * 1.0 / cnts[0] < 1.2);
    ASSERT_TRUE(cnts[2] * 1.0 / cnts[1] >1.8 && cnts[2] * 1.0 / cnts[1] < 2.2);
    ASSERT_TRUE(cnts[2] * 1.0 / cnts[3] > 0.8 && cnts[2] * 1.0/cnts[3] < 1.2);
  }
  {
    std::vector<int32_t> edge_types{0, 2};
    auto r = fn2.SampleNeighbor(edge_types, 100000);
    ASSERT_EQ(r.size(), 100000);
    std::vector<int32_t> cnts(14);
    for (auto i : r) {
      cnts[std::get<0>(i)] += 1;
    }
    ASSERT_TRUE(cnts[3] * 1.0 / cnts[1] > 2.8 && cnts[3] * 1.0 / cnts[1] < 3.2);
    ASSERT_TRUE(cnts[10] * 1.0/cnts[5] > 0.8 && cnts[10] * 1.0 / cnts[5] < 1.2);
    ASSERT_TRUE(cnts[12] * 1.0 / cnts[13] > 0.8 &&
                cnts[12] * 1.0 / cnts[13] < 1.2);
  }
}

}  // namespace euler
