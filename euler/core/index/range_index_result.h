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

#ifndef EULER_CORE_INDEX_RANGE_INDEX_RESULT_H_
#define EULER_CORE_INDEX_RANGE_INDEX_RESULT_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>

#include "euler/common/logging.h"
#include "euler/core/index/index_result.h"
#include "euler/core/index/index_util.h"
#include "euler/core/index/common_index_result.h"
#include "euler/common/random.h"
#include "euler/common/fast_weighted_collection.h"

namespace euler {

template<typename IdType, typename ValueType>
class RangeIndexResult : public IndexResult {
 public:
  typedef typename std::vector<IdType>::const_iterator ResultPos;

  typedef std::pair<ResultPos, ResultPos> ResultRange;

  typedef typename std::vector<IdType>::const_iterator IdI;

  typedef typename std::vector<ValueType>::const_iterator VI;

  typedef std::vector<float>::const_iterator FI;

  explicit  RangeIndexResult(const std::string& name):
      IndexResult(RANGEINDEXRESULT, name) {}

  size_t size() const override {
    size_t res = 0;
    for (auto& r : ranges_) {
      res += std::distance(r.ibeg_, r.iend_);
    }
    return res;
  }

 private:
  struct Section {
    Section(IdI ibeg, IdI iend, IdI isbeg, IdI isend, VI vbeg, FI wbeg):
        ibeg_(ibeg), iend_(iend), isec_beg_(isbeg), isec_end_(isend),
        vbeg_(vbeg), wbeg_(wbeg) {}

    bool empty() {return isec_beg_ >= isec_end_;}

    IdI ibeg_;
    IdI iend_;
    IdI isec_beg_;
    IdI isec_end_;
    VI vbeg_;
    FI wbeg_;
  };

 public:
  bool Init(IdI ibeg, IdI iend, VI vbeg, FI wbeg,
            const std::vector<ResultRange>& r) {
    ranges_.clear();
    for (auto& rr : r) {
      Section s(ibeg, iend, rr.first, rr.second, vbeg, wbeg);
      ranges_.push_back(s);
    }
    auto f = [](const Section i, const Section j)
             {return i.isec_beg_ - i.ibeg_ < j.isec_beg_ - j.ibeg_;};
    std::sort(ranges_.begin(), ranges_.end(), f);
    return true;
  }

  std::vector<uint64_t> GetIds() const override {
    std::vector<uint64_t> result;
    for (auto& r : ranges_) {
      std::copy(r.isec_beg_, r.isec_end_, back_inserter(result));
    }
    return result;
  }

  std::vector<float> GetWeights() const override {
    std::vector<float> result;
    for (auto& r : ranges_) {
      size_t start = result.size() + 1;
      auto iter_begin = r.wbeg_ + (r.isec_beg_ - r.ibeg_);
      auto iter_end = r.wbeg_ + (r.isec_end_ - r.ibeg_);
      std::copy(iter_begin, iter_end, back_inserter(result));
      size_t end = result.size();
      for (size_t j = end; j > start; --j) {
        result[j - 1] -= result[j -2];
      }
    }
    return result;
  }

  std::vector<uint64_t> GetSortedIds() const override {
    return GetIds();
  }

  float SumWeight() const {
    float sum = 0;
    for (auto& r : ranges_) {
      sum += SumWeights(r);
    }
    return sum;
  }

  std::vector<std::pair<uint64_t, float>> Sample(size_t count) const override {
    std::vector<std::pair<uint64_t, float>> result;
    if (ranges_.size() == 0) {
      return result;
    } else if (ranges_.size() == 1) {
      result.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        auto s = Sample(ranges_[0]);
        result.push_back(s);
      }
    } else {
      euler::common::FastWeightedCollection<size_t> fwc;
      std::vector<size_t> ids(ranges_.size());
      std::vector<float> weights(ranges_.size());
      for (size_t i = 0; i < ranges_.size(); ++i) {
        ids[i] = i;
        weights[i] = SumWeights(ranges_[i]);
      }
      fwc.Init(ids, weights);
      result.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        auto id = fwc.Sample();
        result.push_back(Sample(ranges_[id.first]));
      }
    }

    return result;
  }

  std::shared_ptr<IndexResult>
  Intersection(const CommonIndexResult* cIndexResult) {
    CommonIndexResult* result = new CommonIndexResult("common");
    std::vector<std::pair<uint64_t, float>> data;
    auto iter = cIndexResult->GetRangeIter();
    auto begin = iter.first;
    for (auto range : ranges_) {
      auto range_begin = range.isec_beg_;
      while (range_begin != range.isec_end_ && begin != iter.second) {
        if ((*begin).first < uint64_t(*range_begin)) {
          ++begin;
        } else if (uint64_t(*range_begin) < (*begin).first) {
          ++range_begin;
        } else {
          data.push_back(*begin);
          ++range_begin; ++begin;
        }
      }
      if (begin == iter.second) break;
    }
    result->SetData(&data);
    return std::shared_ptr<IndexResult>(result);
  }

  std::shared_ptr<IndexResult>
  Intersection(std::shared_ptr<IndexResult> indexResult) override {
    if (this->GetName() != indexResult->GetName()) {
      if (indexResult->GetType() == COMMONINDEXRESULT) {
        auto cIndexResult = dynamic_cast<CommonIndexResult*>(indexResult.get());
        if (cIndexResult == nullptr) {
          EULER_LOG(FATAL)
              << "IndexResult convert to CommonIndexResult ptr error ";
        }
        return Intersection(cIndexResult);
      }
      auto cIndexResult = ToCommonIndexResult();
      return cIndexResult->Intersection(indexResult);
    }

    RangeIndexResult* rIndexResult =
        dynamic_cast<RangeIndexResult*>(indexResult.get());
    if (rIndexResult == NULL) {
      EULER_LOG(FATAL)
          << "RangeIndexResult convert to RangeIndexResult ptr error ";
    }

    RangeIndexResult* result = new RangeIndexResult(this->GetName());
    for (auto l : ranges_) {
      for (auto r : rIndexResult->ranges_) {
        auto tmp = Intersection(l, r);
        if (!tmp.empty()) {
          result->ranges_.push_back(tmp);
        }
      }
    }

    return std::shared_ptr<IndexResult>(result);
  }

  std::shared_ptr<IndexResult>
  Union(std::shared_ptr<IndexResult> indexResult) override {
    if (this->GetName() != indexResult->GetName()) {
      auto cIndexResult = ToCommonIndexResult();
      return cIndexResult->Union(indexResult);
    }

    RangeIndexResult* rIndexResult =
        dynamic_cast<RangeIndexResult*>(indexResult.get());
    if (rIndexResult == NULL) {
      EULER_LOG(FATAL)
          << "RangeIndexResult convert to RangeIndexResult ptr error ";
    }

    RangeIndexResult* result = new RangeIndexResult(this->GetName());

    std::copy(rIndexResult->ranges_.begin(),
              rIndexResult->ranges_.end(), back_inserter(ranges_));
    if (ranges_.size() <= 1) {
      result->ranges_ = ranges_;
      return std::shared_ptr<IndexResult>(result);
    }

    auto f = [](const Section i, const Section j)
             {return i.isec_beg_ - i.ibeg_ < j.isec_beg_ - j.ibeg_;};
    std::sort(ranges_.begin(), ranges_.end(), f);
    auto tmp = ranges_[0];
    for (size_t i = 1; i < ranges_.size(); ++i) {
      if (!MergeSection(&tmp, ranges_[i])) {
        result->ranges_.push_back(tmp);
        tmp = ranges_[i];
      }
    }
    result->ranges_.push_back(tmp);
    return std::shared_ptr<IndexResult>(result);
  }

  std::shared_ptr<IndexResult> ToCommonIndexResult() override;

  ~RangeIndexResult() {}

 private:
  float SumWeights(const Section& s) const;

  std::vector<std::pair<uint64_t, float>> GetIdWeight() const;

  Section Intersection(const Section& l, const Section& r) const {
    Section result = l;
    result.isec_beg_ = result.ibeg_ +
                       std::max(l.isec_beg_ - l.ibeg_, r.isec_beg_ - r.ibeg_);
    result.isec_end_  = result.ibeg_ +
                        std::min(l.isec_end_ - l.ibeg_, r.isec_end_ - r.ibeg_);
    return result;
  }

  bool MergeSection(Section* l, const Section& r) const {
    if (r.isec_beg_ - r.ibeg_ <= l->isec_end_ - l->ibeg_) {
      l->isec_end_ = l->ibeg_ +
                    std::max(r.isec_end_ - r.ibeg_, l->isec_end_ - l->ibeg_);
      return true;
    }
    return false;
  }

  std::pair<uint64_t, float> Sample(const Section& s) const {
    /* ori_weights: 2  4  8  8  16
       sum_weights: 2  6  14 22 38
       range:          s         e
       this_weights:    4  12 20
    */
    auto begin = s.isec_beg_ - s.ibeg_ + s.wbeg_;
    auto end = s.isec_end_ - s.ibeg_ + s.wbeg_;
    auto len = s.iend_ - s.ibeg_;
    auto pre_end = s.isec_end_ == s.iend_ ? s.wbeg_ + len - 1 : end - 1;
    float bias = begin == s.wbeg_ ? 0 : *(begin-1);
    float limit = *pre_end - bias;

    float r = euler::common::ThreadLocalRandom() *  limit + bias;
    auto p = std::lower_bound(begin, end, r);
    IdType id = *(p - s.wbeg_ + s.ibeg_);

    if (p != s.wbeg_) {
      float weight = *p - *(p - 1);
      return std::make_pair(static_cast<uint64_t>(id), weight);
    }
    return std::make_pair(static_cast<uint64_t>(id), *p);
  }

  std::vector<Section> ranges_;
};

template<typename T1, typename T2>
float RangeIndexResult<T1, T2>::SumWeights(const Section& s) const {
  auto begin = s.isec_beg_ - s.ibeg_ + s.wbeg_;
  auto end = s.isec_end_ - s.ibeg_ + s.wbeg_;
  auto len = s.iend_ - s.ibeg_;
  auto pre_end = s.isec_end_ == s.iend_ ? s.wbeg_ + len - 1 : end - 1;
  float bias = begin == s.wbeg_ ? 0 : *(begin-1);
  return *pre_end - bias;
}

template<typename T1, typename T2>
std::vector<std::pair<uint64_t, float>>
RangeIndexResult<T1, T2>::GetIdWeight() const {
  std::vector<std::pair<uint64_t, float>> result;
  for (auto& s : ranges_) {
    auto begin = s.isec_beg_;
    while (begin != s.isec_end_) {
      auto wbegin = begin - s.ibeg_ + s.wbeg_;
      float bias = wbegin == s.wbeg_ ? 0 : *(wbegin-1);
      result.push_back(std::make_pair(*begin, *wbegin - bias));
      ++begin;
    }
  }
  return result;
}

template<typename T1, typename T2>
std::shared_ptr<IndexResult> RangeIndexResult<T1, T2>::ToCommonIndexResult() {
  std::vector<std::pair<uint64_t, float>> data;
  data = GetIdWeight();
  auto f = [](const std::pair<uint64_t, float>& a,
              const std::pair<uint64_t, float>& b) {return a.first < b.first;};
  std::sort(data.begin(), data.end(), f);
  CommonIndexResult* commonResult = new CommonIndexResult("common", data);
  return std::shared_ptr<IndexResult>(commonResult);
}

}  // namespace euler

#endif  // EULER_CORE_INDEX_RANGE_INDEX_RESULT_H_
