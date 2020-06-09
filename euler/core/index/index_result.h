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

#ifndef EULER_CORE_INDEX_INDEX_RESULT_H_
#define EULER_CORE_INDEX_INDEX_RESULT_H_

#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "euler/core/index/index_types.h"

namespace euler {

class CommonIndexResult;

class IndexResult {
 public:
  IndexResult(IndexResultType type, const std::string& name)
      : type_(type), name_(name) {}

  virtual std::vector<uint64_t> GetIds() const = 0;

  virtual std::vector<float> GetWeights() const = 0;

  virtual std::vector<uint64_t> GetSortedIds() const {
    auto v = GetIds();
    std::sort(v.begin(), v.end());
    return v;
  }

  virtual IndexResultType GetType() const {
    return type_;
  }

  virtual std::string GetName() const {
    return name_;
  }

  virtual std::vector<std::pair<uint64_t, float>>
  Sample(size_t count) const = 0;

  virtual std::shared_ptr<IndexResult>
  Intersection(std::shared_ptr<IndexResult> indexResult) = 0;

  virtual std::shared_ptr<IndexResult>
  Union(std::shared_ptr<IndexResult> indexResult) = 0;

  virtual std::shared_ptr<IndexResult> ToCommonIndexResult() = 0;

  virtual float SumWeight() const = 0;

  virtual size_t size() const = 0;

  virtual void DebugInfo() const {}

  virtual ~IndexResult() {}

 private:
  IndexResultType type_;
  std::string name_;
};

}  // namespace euler

#endif  // EULER_CORE_INDEX_INDEX_RESULT_H_
