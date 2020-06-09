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

#ifndef EULER_CORE_INDEX_SAMPLE_INDEX_H_
#define EULER_CORE_INDEX_SAMPLE_INDEX_H_

#include <vector>
#include <memory>
#include <string>

#include "euler/core/index/index_types.h"
#include "euler/core/index/index_result.h"
#include "euler/common/file_io.h"
#include "euler/common/slice.h"

namespace euler {

class SampleIndex {
 public:
  explicit SampleIndex(const std::string& name) : name_(name) {}

  virtual bool Serialize(FileIO* file_io) const = 0;

  virtual bool Deserialize(FileIO* file_io) = 0;

  virtual uint32_t SerializeSize() const = 0;

  virtual std::string GetName() const {
    return name_;
  }

  virtual bool Merge(std::shared_ptr<SampleIndex> hIndex) = 0;

  virtual bool Merge(
               const std::vector<std::shared_ptr<SampleIndex>>& hIndex) = 0;

  virtual std::shared_ptr<IndexResult> Search(
                      IndexSearchType op,
                      const std::string& value) const = 0;

  virtual std::shared_ptr<IndexResult> Search(
      const std::string& op,
      const std::string& value) const {
    IndexSearchType type = EQ;
    if (op == "eq") {
      type = EQ;
    } else if (op == "lt") {
      type = LESS;
    } else if (op == "le") {
      type = LESS_EQ;
    } else if (op == "gt") {
      type = GREATER;
    } else if (op == "ge") {
      type = GREATER_EQ;
    } else if (op == "ne") {
      type = NOT_EQ;
    } else {
      EULER_LOG(FATAL) << "not support this op " << op;
    }

    return Search(type, value);
  }

  virtual std::shared_ptr<IndexResult>
  SearchAll() const {return std::shared_ptr<IndexResult> ();}

  virtual ~SampleIndex() {}

 private:
  std::string name_;
};

}  // namespace euler

#endif  // EULER_CORE_INDEX_SAMPLE_INDEX_H_
