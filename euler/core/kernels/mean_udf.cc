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

#include <climits>

#include "euler/core/framework/udf.h"

namespace euler {
class MeanUdf: public ValuesUdf {
 public:
  explicit MeanUdf(const std::string& name): ValuesUdf(name) {}

 protected:
  virtual std::vector<NodesFeature> Process(
      const std::vector<std::string*>& udf_fids,
      const std::vector<Tensor*>& udf_params,
      const std::vector<NodesFeature>& feature_vec) {
    (void)udf_fids;
    (void)udf_params;
    if (feature_vec.empty()) {
      EULER_LOG(FATAL) << "empty fid";
    }
    for (const NodesFeature& nf : feature_vec) {
      if (nf.feature_type_ != kDense) {
        EULER_LOG(FATAL) << "not support feature type";
      }
    }

    size_t node_num = feature_vec[0].fv_.size();
    std::vector<NodesFeature> result;
    for (size_t k = 0; k < feature_vec.size(); ++k) {
      FloatFeatureVec ffv(node_num);
      for (size_t i = 0; i < node_num; ++i) {
        ffv[i].resize(1);
        if (!feature_vec[k].fv_[i][0].empty()) {
          ffv[i][0].resize(1);
          double sum = 0;
          for (size_t j = 0; j < feature_vec[k].fv_[i][0].size(); ++j) {
            sum += feature_vec[k].fv_[i][0][j];
          }
          ffv[i][0][0] =
            static_cast<float>(sum / feature_vec[k].fv_[i][0].size());
        }
      }
      result.push_back(NodesFeature(ffv));
    }
    return result;
  }
};

REGISTER_UDF("udf_mean", MeanUdf);
}  // namespace euler
