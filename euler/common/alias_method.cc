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

#include "euler/common/alias_method.h"

#include <stdint.h>

namespace euler {
namespace common {

void AliasMethod::Init(const std::vector<float>& weights) {
  prob_.resize(weights.size());
  alias_.resize(weights.size());
  std::vector<int64_t> small, large;
  std::vector<float> weights_(weights);
  double avg = 1 / static_cast<double>(weights_.size());
  for (size_t i = 0; i < weights_.size(); i++) {
    if (weights_[i] > avg) {
      large.push_back(i);
    } else {
      small.push_back(i);
    }
  }

  int64_t less, more;
  while (large.size() > 0 && small.size() > 0) {
    less = small.back();
    small.pop_back();
    more = large.back();
    large.pop_back();
    prob_[less] = weights_[less] * weights_.size();
    alias_[less] = more;
    weights_[more] = weights_[more] + weights_[less] - avg;
    if (weights_[more] > avg) {
      large.push_back(more);
    } else {
      small.push_back(more);
    }
  }  // while (large.size() > 0 && small.size() > 0)
  while (small.size() > 0) {
    less = small.back();
    small.pop_back();
    prob_[less] = 1.0;
  }

  while (large.size() > 0) {
    more = large.back();
    large.pop_back();
    prob_[more] = 1.0;
  }
}  // Init


int64_t AliasMethod::Next() const {
  int64_t column = NextLong(prob_.size());
  bool coinToss = ThreadLocalRandom() < prob_[column];
  return coinToss ? column : alias_[column];
}

size_t AliasMethod::GetSize() const {
  return prob_.size();
}

int64_t AliasMethod::NextLong(int64_t n) const {
  return floor(n * ThreadLocalRandom());
}

std::string AliasMethod::ShowData() const {
  std::string result = "prob: {\n";
  for (auto& prob : prob_) {
    result += std::to_string(prob);
    result += "\n";
  }
  result += "}\n";

  result += "alias: {\n";
  for (auto& alias : alias_) {
    result += std::to_string(alias);
    result += "\n";
  }
  result += "}\n";

  return result;
}

}  // namespace common
}  // namespace euler
