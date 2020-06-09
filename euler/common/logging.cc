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


#include "euler/common/logging.h"

#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <sstream>
#include <cstring>

#include "euler/common/time_utils.h"

namespace euler {

LogMessage::LogMessage(const char* fname, int line, int severity)
  : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::GenerateLogMessage() {
  uint64_t now_micros = TimeUtils::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t kBufferSize = 30;
  char time_buffer[kBufferSize];
  strftime(time_buffer, kBufferSize, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  fprintf(stderr, "%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder,
          "IWEF"[severity_], fname_, line_, str().c_str());
}

namespace {

int64_t GetLogLevelFromEnv() {
  const char* log_level = getenv("EULER_LOG_LEVEL");
  if (log_level == nullptr) {
    return INFO;
  }

  return atoi(log_level);
}

}  // namespace

LogMessage::~LogMessage() {
  static int64_t min_log_level = GetLogLevelFromEnv();
  if (likely(severity_ >= min_log_level)) {
    GenerateLogMessage();
  }
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  abort();
}

struct timeval GetTime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t;
}

uint64_t GetTimeInterval(const struct timeval& a,
                         const struct timeval& b) {
  return 1000000 * (b.tv_sec - a.tv_sec) + b.tv_usec - a.tv_usec;
}

}  // namespace euler
