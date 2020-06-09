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

#include "euler/common/status.h"

#include <vector>
#include <string>

namespace euler {

Status::Status(): code_(ErrorCode::OK) {
}

Status::Status(ErrorCode code, const Slice& message)
    : code_(code), message_(message.data(), message.size()) {
}

std::string Status::DebugString() const {
  if (ok()) {
    return "OK";
  }

  std::string result(kCodeToMsg[code_]);
  result += ": ";
  result += message_;
  return result;
}

std::ostream& operator<<(std::ostream& os, const Status& s) {
  os << s.DebugString();
  return os;
}

}  // namespace euler
