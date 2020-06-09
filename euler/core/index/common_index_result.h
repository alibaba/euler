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

#ifndef EULER_CORE_INDEX_COMMON_INDEX_RESULT_H_
#define EULER_CORE_INDEX_COMMON_INDEX_RESULT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "euler/core/index/index_result.h"
#include "euler/common/fast_weighted_collection.h"

namespace euler {

class CommonIndexResult : public IndexResult {
 public:
  typedef std::vector<std::pair<uint64_t, float>>::const_iterator Iter;

  explicit CommonIndexResult(const std::string& name):
     IndexResult(COMMONINDEXRESULT, name), sorted_data_(), sampler(nullptr) {}

  CommonIndexResult(const std::string& name,
                    const std::vector<std::pair<uint64_t, float>>& data):
                    IndexResult(COMMONINDEXRESULT, name),
                    sorted_data_(data), sampler(nullptr) {}

  void SetData(std::vector<std::pair<uint64_t, float>>* data);

  std::vector<uint64_t> GetIds() const override;

  std::vector<float> GetWeights() const override;

  std::vector<uint64_t> GetSortedIds() const override;

  std::vector<std::pair<uint64_t, float>> Sample(size_t count) const override;

  std::shared_ptr<IndexResult>
  Intersection(std::shared_ptr<IndexResult> indexResult) override;

  std::shared_ptr<IndexResult>
  Union(std::shared_ptr<IndexResult> indexResult) override;

  std::shared_ptr<IndexResult> ToCommonIndexResult() override;

  std::pair<Iter, Iter> GetRangeIter() const;

  float SumWeight() const override;

  size_t size() const override { return sorted_data_.size(); }

  ~CommonIndexResult() {}

 private:
  std::vector<std::pair<uint64_t, float>> sorted_data_;
  mutable std::shared_ptr<
     euler::common::FastWeightedCollection<uint64_t>> sampler;
};

}  // namespace euler

#endif  // EULER_CORE_INDEX_COMMON_INDEX_RESULT_H_
