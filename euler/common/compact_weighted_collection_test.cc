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

#include "gtest/gtest.h"

#include "euler/common/compact_weighted_collection.h"

namespace euler {
namespace common {

TEST(CompactWeightedCollectionTest, Init) {
  CompactWeightedCollection<int32_t> cwc;
  std::vector<int32_t> ids = {0, 1, 2, 3, 4};
  std::vector<float> weights = {2.0, 4.0, 8.0, 8.0, 16.0};
  cwc.Init(ids, weights);
  // test sampling
  int32_t cnts[5] = {0};
  for (size_t i = 0; i < 100000; ++i) {
    std::pair<int32_t, float> r = cwc.Sample();
    ++cnts[r.first];
  }
  std::cout << "distribution should be 1:2:4:4:8" << std::endl;
  std::cout << "distribution:" << std::endl;
  for (size_t i = 0; i < 5; ++i) {
    std::cout << cnts[i] << " ";
  }
  std::cout << std::endl;

  // test iteration
  size_t size = cwc.GetSize();
  for (size_t i = 0; i < size; ++i) {
    std::pair<int32_t, float> id_weight_pair = cwc.Get(i);
    ASSERT_EQ(id_weight_pair.first, ids[i]);
    ASSERT_EQ(id_weight_pair.second, weights[i]);
  }

  auto w = cwc.GetWeights();
  ASSERT_EQ(w.size(), weights.size());
  for (size_t i = 0; i < w.size(); ++i) {
    ASSERT_EQ(w[i], weights[i]);
  }
}

TEST(CompactWeightedCollectionTest, Sample) {
  CompactWeightedCollection<int32_t> cwc;
  std::vector<int32_t> ids = {3, 4, 0, 2};
  std::vector<float> weights = {1.38629436, 0, 0, 0};
  cwc.Init(ids, weights);
  for (size_t i = 0; i < 100000000; ++i) {
    int32_t id = cwc.Sample().first;
    if (id != 3) {
      EULER_LOG(FATAL) << "error " << id;
    }
  }
}

TEST(CompactWeightedCollectionTest, Sample2) {
  CompactWeightedCollection<int32_t> cwc;
  std::vector<int32_t> ids = {4, 3};
  std::vector<float> weights = {0, 1.38629436};
  cwc.Init(ids, weights);
  for (size_t i = 0; i < 100000000; ++i) {
    int32_t id = cwc.Sample().first;
    if (id != 3) {
      EULER_LOG(FATAL) << "error " << id;
    }
  }
}

}  // namespace common
}  // namespace euler
