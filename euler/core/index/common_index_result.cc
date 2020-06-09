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

#include "euler/core/index/common_index_result.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "euler/common/logging.h"
#include "euler/core/index/range_index_result.h"

namespace euler {

void CommonIndexResult::SetData(std::vector<std::pair<uint64_t, float>>* data) {
  swap(sorted_data_, *data);
}

std::vector<uint64_t> CommonIndexResult::GetIds() const {
  std::vector<uint64_t> result(sorted_data_.size());
  auto f = [](const std::pair<uint64_t, float>& a) {return a.first;};
  std::transform(sorted_data_.begin(), sorted_data_.end(), result.begin(), f);
  return result;
}

std::vector<float> CommonIndexResult::GetWeights() const {
  std::vector<float> result(sorted_data_.size());
  auto f = [](const std::pair<uint64_t, float>& a) { return a.second; };
  std::transform(sorted_data_.begin(), sorted_data_.end(), result.begin(), f);
  return result;
}

std::vector<uint64_t> CommonIndexResult::GetSortedIds() const {
  return GetIds();
}

std::vector<std::pair<uint64_t, float>>
      CommonIndexResult::Sample(size_t count) const {
  if (sampler == nullptr) {
    auto p = new euler::common::FastWeightedCollection<uint64_t>();
    p->Init(sorted_data_);
    sampler = std::shared_ptr<
              euler::common::FastWeightedCollection<uint64_t>>(p);
  }
  std::vector<std::pair<uint64_t, float>> result(count);
  for (size_t i = 0; i < count; ++i) {
    result[i] = sampler->Sample();
  }
  return result;
}

std::shared_ptr<IndexResult>
   CommonIndexResult::Intersection(std::shared_ptr<IndexResult> indexResult) {
  CommonIndexResult* cIndexResult = nullptr;
  std::shared_ptr<IndexResult> commonIndexResult(nullptr);

  auto type = indexResult->GetType();
  if (type != COMMONINDEXRESULT) {
    commonIndexResult  = indexResult->ToCommonIndexResult();
    cIndexResult = dynamic_cast<CommonIndexResult*>(commonIndexResult.get());
    if (cIndexResult == nullptr) {
      EULER_LOG(FATAL) << "IndexResult convert to CommonIndexResult ptr error ";
    }
  } else {
    cIndexResult = dynamic_cast<CommonIndexResult*>(indexResult.get());
    if (cIndexResult == nullptr) {
      EULER_LOG(FATAL) << "IndexResult convert to CommonIndexResult ptr error ";
    }
  }

  CommonIndexResult* result = new CommonIndexResult("common");
  auto f = [](const std::pair<uint64_t, float>& a,
              const std::pair<uint64_t, float>& b) {return a.first < b.first;};
  std::set_intersection(sorted_data_.begin(), sorted_data_.end(),
                        cIndexResult->sorted_data_.begin(),
                        cIndexResult->sorted_data_.end(),
                        back_inserter(result->sorted_data_), f);
  return std::shared_ptr<IndexResult>(result);
}

std::shared_ptr<IndexResult>
     CommonIndexResult::Union(std::shared_ptr<IndexResult> indexResult) {
  CommonIndexResult* cIndexResult = nullptr;
  std::shared_ptr<IndexResult> commonIndexResult(nullptr);

  auto type = indexResult->GetType();
  if (type != COMMONINDEXRESULT) {
    commonIndexResult  = indexResult->ToCommonIndexResult();
    cIndexResult = dynamic_cast<CommonIndexResult*>(commonIndexResult.get());
    if (cIndexResult == nullptr) {
      EULER_LOG(FATAL)
          << "CommonIndexResult convert to CommonIndexResult ptr error ";
    }
  } else {
    cIndexResult = dynamic_cast<CommonIndexResult*>(indexResult.get());
    if (cIndexResult == nullptr) {
      EULER_LOG(FATAL)
          << "CommonIndexResult convert to CommonIndexResult ptr error ";
    }
  }

  CommonIndexResult* result = new CommonIndexResult("common");
  auto f = [](const std::pair<uint64_t, float>& a,
              const std::pair<uint64_t, float>& b) {return a.first < b.first;};
  std::set_union(sorted_data_.begin(), sorted_data_.end(),
                 cIndexResult->sorted_data_.begin(),
                 cIndexResult->sorted_data_.end(),
                 back_inserter(result->sorted_data_), f);
  return std::shared_ptr<IndexResult>(result);
}

std::shared_ptr<IndexResult> CommonIndexResult::ToCommonIndexResult() {
  CommonIndexResult* result =
        new CommonIndexResult("common", this->sorted_data_);
  return std::shared_ptr<IndexResult>(result);
}

std::pair<CommonIndexResult::Iter, CommonIndexResult::Iter>
     CommonIndexResult::GetRangeIter() const {
  return std::make_pair(sorted_data_.begin(), sorted_data_.end());
}

float CommonIndexResult::SumWeight() const {
  if (sampler != nullptr) {
    return sampler->GetSumWeight();
  }
  float sum = 0;
  for (auto& v : sorted_data_) {
    sum += v.second;
  }
  return sum;
}

}  // namespace euler
