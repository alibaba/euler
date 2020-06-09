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


#ifndef EULER_COMMON_LOGGING_H_
#define EULER_COMMON_LOGGING_H_

#include <sys/time.h>
#include <stdlib.h>

#include <limits>
#include <sstream>

namespace euler {

const int DEBUG = 0;
const int INFO = 1;
const int WARNING = 2;
const int ERROR = 3;
const int FATAL = 4;
const int NUM_SEVERITIES = 5;

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _EULER_LOG_INFO                                   \
  ::euler::LogMessage(__FILE__, __LINE__, ::euler::INFO)
#define _EULER_LOG_DEBUG                                  \
  ::euler::LogMessage(__FILE__, __LINE__, ::euler::DEBUG)
#define _EULER_LOG_WARNING                                  \
  ::euler::LogMessage(__FILE__, __LINE__, ::euler::WARNING)
#define _EULER_LOG_ERROR                                  \
  ::euler::LogMessage(__FILE__, __LINE__, ::euler::ERROR)
#define _EULER_LOG_FATAL                        \
  ::euler::LogMessageFatal(__FILE__, __LINE__)

#define EULER_LOG(severity) _EULER_LOG_##severity

#define EULER_DLOG(severity) EULER_LOG(severity)

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define EULER_CHECK(condition)                          \
  if (unlikely(!(condition)))                           \
    EULER_LOG(FATAL) << "Check failed: " #condition " "

#define EULER_CHECK_EQ(lhs, rhs)                          \
  if (unlikely(((lhs) != (rhs))))                         \
    EULER_LOG(FATAL) << "Check failed: " #lhs " == " #rhs


#define EULER_DCHECK(condition) EULER_CHECK(condition)

struct timeval GetTime();

uint64_t GetTimeInterval(const struct timeval& a,
                         const struct timeval& b);
}  // namespace euler

#endif  // EULER_COMMON_LOGGING_H_
