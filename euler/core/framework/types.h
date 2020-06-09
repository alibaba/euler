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

#ifndef EULER_CORE_FRAMEWORK_TYPES_H_
#define EULER_CORE_FRAMEWORK_TYPES_H_

#include <cstdint>
#include <string>

#include "euler/common/logging.h"

namespace euler {

enum DataType : int32_t {
  kInt8 = 0,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat,
  kDouble,
  kBool,
  kString
};

template <class T>
struct DataTypeToEnum {};

template <DataType VALUE>
struct EnumToDataType {};

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)               \
  template <>                                         \
  struct DataTypeToEnum<TYPE> {                       \
    static DataType v() { return DataType::ENUM; }    \
    static constexpr DataType value = DataType::ENUM; \
  };                                                  \
  template <>                                         \
  struct EnumToDataType<DataType::ENUM> {             \
    typedef TYPE Type;                                \
  }

MATCH_TYPE_AND_ENUM(int8_t, kInt8);
MATCH_TYPE_AND_ENUM(int16_t, kInt16);
MATCH_TYPE_AND_ENUM(int32_t, kInt32);
MATCH_TYPE_AND_ENUM(int64_t, kInt64);
MATCH_TYPE_AND_ENUM(uint8_t, kUInt8);
MATCH_TYPE_AND_ENUM(uint16_t, kUInt16);
MATCH_TYPE_AND_ENUM(uint32_t, kUInt32);
MATCH_TYPE_AND_ENUM(uint64_t, kUInt64);
MATCH_TYPE_AND_ENUM(float, kFloat);
MATCH_TYPE_AND_ENUM(double, kDouble);
MATCH_TYPE_AND_ENUM(bool, kBool);
MATCH_TYPE_AND_ENUM(std::string, kString);

#undef MATCH_TYPE_AND_ENUM

#define EULER_TYPE_SINGLE_ARG(...) __VA_ARGS__

#define EULER_TYPE_CASE(TYPE, STMTS)            \
  case ::euler::DataTypeToEnum<TYPE>::value: {  \
    typedef TYPE T;                             \
    STMTS;                                      \
    break;                                      \
  }

#define EULER_TYPE_CASES(TYPE_ENUM, STMTS)                      \
  switch (TYPE_ENUM) {                                          \
    EULER_TYPE_CASE(int8_t, EULER_TYPE_SINGLE_ARG(STMTS));      \
    EULER_TYPE_CASE(int16_t, EULER_TYPE_SINGLE_ARG(STMTS));     \
    EULER_TYPE_CASE(int32_t, EULER_TYPE_SINGLE_ARG(STMTS));     \
    EULER_TYPE_CASE(int64_t, EULER_TYPE_SINGLE_ARG(STMTS));     \
    EULER_TYPE_CASE(uint8_t, EULER_TYPE_SINGLE_ARG(STMTS));     \
    EULER_TYPE_CASE(uint16_t, EULER_TYPE_SINGLE_ARG(STMTS));    \
    EULER_TYPE_CASE(uint32_t, EULER_TYPE_SINGLE_ARG(STMTS));    \
    EULER_TYPE_CASE(uint64_t, EULER_TYPE_SINGLE_ARG(STMTS));    \
    EULER_TYPE_CASE(float, EULER_TYPE_SINGLE_ARG(STMTS));       \
    EULER_TYPE_CASE(double, EULER_TYPE_SINGLE_ARG(STMTS));      \
    EULER_TYPE_CASE(bool, EULER_TYPE_SINGLE_ARG(STMTS));        \
    default: EULER_CHECK(false) << "type error";                \
  }                                                             \

inline size_t SizeOfType(DataType type) {
  if (type == DataType::kString) {
    return sizeof(std::string*);
  }

  size_t ret = 0;
  EULER_TYPE_CASES(type, ret = sizeof(T));
  return ret;
}

}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_TYPES_H_
