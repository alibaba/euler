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

#ifndef EULER_COMMON_ERROR_CODE_H_
#define EULER_COMMON_ERROR_CODE_H_

#include <vector>
#include <string>

namespace euler {

enum ErrorCode {
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
  UNAUTHENTICATED = 16,
  PROTO_ERROR = 17,
  RPC_ERROR = 18
};

const std::vector<std::string> kCodeToMsg = {
  "OK",                      // OK
  "Cancelled",               // CANCELLED
  "Unknown",                 // UNKNOWN
  "Invalid argument",        // INVALID_ARGUMENT
  "Deadline exceeded",       // DEADLINE_EXCEEDED
  "Not found",               // NOT_FOUND
  "Already exists",          // ALREADY_EXISTS
  "Permission denied",       // PERMISSION_DENIED
  "Resource exhausted",      // RESOURCE_EXHAUSTED
  "Failed precondition",     // FAILED_PRECONDITION
  "Aborted",                 // ABORTED
  "Out of range",            // OUT_OF_RANGE
  "Unimplemented",           // UNIMPLEMENTED
  "Internal",                // INTERNAL
  "Unavailable",             // UNAVAILABLE
  "Data loss",               // DATA_LOSS
  "Unauthenticated",         // UNAUTHENTICATED
  "PROTO_ERROR",             // PROTO_ERROR
  "RPC_ERROR"                // RPC_ERROR
};

}  // namespace euler

#endif  // EULER_COMMON_ERROR_CODE_H_
