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

#include "euler/core/graph/edge.h"

namespace euler {

using std::vector;
using std::string;

Edge* GetEdge() {
  Edge* fn = new Edge(1, 2, 0, 1.0);
  vector<vector<uint64_t>> uint64_features;
  {
    uint64_features.push_back({10001, 10002, 10003});
    uint64_features.push_back({20001, 20002, 20003, 20004});
    uint64_features.push_back({30001, 30002});
  }
  vector<vector<float>> float_features;
  {
    float_features.push_back({1.1, 1.2});
    float_features.push_back({2.1, 2.2, 2.3});
  }
  vector<string> binary_features;
  {
    binary_features.push_back("iam1");
    binary_features.push_back("iam22");
    binary_features.push_back("iam333");
  }
  if (fn->Init(uint64_features, float_features, binary_features)) {
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

TEST(EdgeTest, EdgeFunction) {
  Edge* fn = GetEdge();
  ASSERT_TRUE(fn != nullptr);
  string s;
  ASSERT_TRUE(fn->Serialize(&s));
  Edge fn2;
  ASSERT_TRUE(fn2.DeSerialize(s.c_str(), s.length()));

  ASSERT_EQ(fn2.GetFloat32FeatureValueNum(), 3);
  ASSERT_EQ(fn2.GetUint64FeatureValueNum(), 4);
  ASSERT_EQ(fn2.GetBinaryFeatureValueNum(), 6);
  {
    vector<int32_t> fids{0, 1};
    vector<vector<uint64_t>> r;
    fn2.GetUint64Feature(fids, &r);
    vector<vector<uint64_t>> expect;
    expect.push_back({10001, 10002, 10003});
    expect.push_back({20001, 20002, 20003, 20004});
    CHECK_VEC_VEC(r, expect);
  }
  {
    vector<int32_t> fids{0, 1};
    vector<vector<float>> r;
    fn2.GetFloat32Feature(fids, &r);
    vector<vector<float>> expect;
    expect.push_back({1.1, 1.2});
    expect.push_back({2.1, 2.2, 2.3});
    CHECK_VEC_VEC(r, expect);
  }
  {
    vector<int32_t> fids{2};
    vector<string> r;
    fn2.GetBinaryFeature(fids, &r);
    ASSERT_EQ(r.size(), 1);
    ASSERT_EQ(r[0], "iam333");
  }
}

}  // namespace euler
