/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef EULER_CLIENT_STATUS_H_
#define EULER_CLIENT_STATUS_H_

#include <string>

namespace euler {
namespace client {

enum StatusCode {
  OK = 0,
  INVALID_ARGUMENT = 1,
  PROTO_ERROR = 2,
  RPC_ERROR = 3
};

class Status {
 public:
  Status() : code_(StatusCode::OK) { }

  Status(StatusCode code, const std::string &message)
      : code_(code), message_(message) { }

  static const Status &OK;

  bool ok() const { return code_ == StatusCode::OK; }

  std::string message() const { return message_; }

 private:
  StatusCode code_;
  std::string message_;
};

}  // namespace client
}  // namespace euler

#endif  // EULER_CLIENT_STATUS_H_
