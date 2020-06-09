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

#include "euler/common/logging.h"
#include "euler/core/framework/udf.h"
#include "euler/core/kernels/min_udf.cc"
#include "euler/core/kernels/max_udf.cc"
#include "euler/core/kernels/mean_udf.cc"

namespace euler {

class MinUdfTestClass: public MinUdf {
 public:
  explicit MinUdfTestClass(const std::string& name): MinUdf(name) {}
  virtual std::vector<NodesFeature> ProcessTest(
      const std::vector<NodesFeature>& feature_vec) {
    std::vector<std::string*> udf_fids;
    std::vector<Tensor*> udf_params;
    return Process(udf_fids, udf_params, feature_vec);
  }
};

class MaxUdfTestClass: public MaxUdf {
 public:
  explicit MaxUdfTestClass(const std::string& name): MaxUdf(name) {}
  virtual std::vector<NodesFeature> ProcessTest(
      const std::vector<NodesFeature>& feature_vec) {
    std::vector<std::string*> udf_fids;
    std::vector<Tensor*> udf_params;
    return Process(udf_fids, udf_params, feature_vec);
  }
};

class MeanUdfTestClass: public MeanUdf {
 public:
  explicit MeanUdfTestClass(const std::string& name): MeanUdf(name) {}
  virtual std::vector<NodesFeature> ProcessTest(
      const std::vector<NodesFeature>& feature_vec) {
    std::vector<std::string*> udf_fids;
    std::vector<Tensor*> udf_params;
    return Process(udf_fids, udf_params, feature_vec);
  }
};

TEST(UdfTest, Min) {
  MinUdfTestClass min_udf_test("min");
  std::vector<NodesFeature> feature_vec;
  UInt64FeatureVec u_feature_vec(4);
  u_feature_vec[0] = {{5, 2, 3, 2}};
  u_feature_vec[1] = {{2, 3, 1}};
  u_feature_vec[2] = {{1, 6, 3, 8}};
  u_feature_vec[3] = {{}};
  feature_vec.push_back(NodesFeature(u_feature_vec));

  FloatFeatureVec f_feature_vec(4);
  f_feature_vec[0] = {{5, 2, 3, 2}};
  f_feature_vec[1] = {{2, 3, 1}};
  f_feature_vec[2] = {{1, 6, 3, 8}};
  f_feature_vec[3] = {{}};
  feature_vec.push_back(NodesFeature(f_feature_vec));

  std::vector<uint64_t> expect1 = {2, 1, 1};
  std::vector<float> expect2 = {2, 1, 1};

  std::vector<NodesFeature> results =
    min_udf_test.ProcessTest(feature_vec);

  for (size_t j = 0; j < results[0].uv_.size(); ++j) {  // each node
    for (size_t k = 0; k < results[0].uv_[j][0].size(); ++k) {
      ASSERT_EQ(expect1[j], results[0].uv_[j][0][k]);
    }
  }

  for (size_t j = 0; j < results[0].fv_.size(); ++j) {  // each node
    for (size_t k = 0; k < results[0].fv_[j][0].size(); ++k) {
      ASSERT_EQ(expect2[j], results[0].fv_[j][0][k]);
    }
  }
}

TEST(UdfTest, Max) {
  MaxUdfTestClass max_udf_test("min");
  std::vector<NodesFeature> feature_vec;
  UInt64FeatureVec u_feature_vec(4);
  u_feature_vec[0] = {{5, 2, 3, 2}};
  u_feature_vec[1] = {{2, 3, 1}};
  u_feature_vec[2] = {{1, 6, 3, 8}};
  u_feature_vec[3] = {{}};
  feature_vec.push_back(NodesFeature(u_feature_vec));

  FloatFeatureVec f_feature_vec(4);
  f_feature_vec[0] = {{5, 2, 3, 2}};
  f_feature_vec[1] = {{2, 3, 1}};
  f_feature_vec[2] = {{1, 6, 3, 8}};
  f_feature_vec[3] = {{}};
  feature_vec.push_back(NodesFeature(f_feature_vec));
  std::vector<uint64_t> expect1 = {5, 3, 8};
  std::vector<float> expect2 = {5, 3, 8};

  std::vector<NodesFeature> results =
    max_udf_test.ProcessTest(feature_vec);

  for (size_t j = 0; j < results[0].uv_.size(); ++j) {  // each node
    for (size_t k = 0; k < results[0].uv_[j][0].size(); ++k) {
      ASSERT_EQ(expect1[j], results[0].uv_[j][0][k]);
    }
  }
  for (size_t j = 0; j < results[0].fv_.size(); ++j) {  // each node
    for (size_t k = 0; k < results[0].fv_[j][0].size(); ++k) {
      ASSERT_EQ(expect2[j], results[0].fv_[j][0][k]);
    }
  }
}

TEST(UdfTest, Mean) {
  MeanUdfTestClass mean_udf_test("min");
  std::vector<NodesFeature> feature_vec;
  FloatFeatureVec f_feature_vec(4);
  f_feature_vec[0] = {{5, 2, 3, 2}};
  f_feature_vec[1] = {{2, 3, 1}};
  f_feature_vec[2] = {{1, 6, 3, 8}};
  f_feature_vec[3] = {{}};
  feature_vec.push_back(NodesFeature(f_feature_vec));
  std::vector<float> expect = {3, 2, 4.5};

  std::vector<NodesFeature> results =
    mean_udf_test.ProcessTest(feature_vec);

  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < results[i].fv_.size(); ++j) {  // each node
      for (size_t k = 0; k < results[i].fv_[j][0].size(); ++k) {
        ASSERT_EQ(expect[j], results[i].fv_[j][0][k]);
      }
    }
  }
}

}  // namespace euler
