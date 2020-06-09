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


#ifndef EULER_COMMON_STR_UTIL_H_
#define EULER_COMMON_STR_UTIL_H_

#include <ctype.h>

#include <string>
#include <vector>
#include <sstream>

#include "euler/common/slice.h"

namespace euler {

template <typename Predicate>
size_t LeftTrim(Slice* text, Predicate p) {
  size_t pos = 0;
  const char* ptr = text->data();
  while (pos < text->size() && p(*ptr)) {
    ++pos;
    ++ptr;
  }
  text->remove_prefix(pos);
  return pos;
}
template <typename Predicate>
size_t RightTrim(Slice* text, Predicate p) {
  if (text->empty()) {
    return 0;
  }

  size_t pos = 0;
  const char* ptr = text->data() + text->size() - 1;
  while (pos < text->size() && p(*ptr)) {
    ++pos;
    --ptr;
  }
  text->remove_suffix(pos);
  return pos;
}
struct IsWhiteSpace {
  bool operator()(char c) const {
    return isspace(c);
  }
};
inline size_t LeftTrim(Slice* text) {
  return LeftTrim(text, IsWhiteSpace());
}
inline size_t RightTrim(Slice* text) {
  return RightTrim(text, IsWhiteSpace());
}
inline size_t LeftTrim(Slice* text, Slice delims) {
  return LeftTrim(text,
                  [delims] (char c) { return delims.find(c) != Slice::npos; });
}
inline size_t RightTrim(Slice* text, Slice delims) {
  return RightTrim(text,
                   [delims] (char c) { return delims.find(c) != Slice::npos; });
}
inline size_t LeftTrim(Slice* text, char c) {
  return LeftTrim(text, [c] (char t) { return c == t; });
}
inline size_t RightTrim(Slice* text, char c) {
  return RightTrim(text, [c] (char t) { return c == t; });
}
inline size_t Trim(Slice* text) {
  LeftTrim(text, IsWhiteSpace());
  return RightTrim(text, IsWhiteSpace());
}

inline std::string Replace(Slice text, Slice old,
                           Slice newb, bool replace_all) {
  std::string result(text);
  size_t pos = 0;
  while ((pos = result.find(old.data(), pos, old.size())) !=
         std::string::npos) {
    result.replace(pos, old.size(), newb.data(), newb.size());
    pos += newb.size();
    if (old.empty()) {
      pos++;
    }
    if (!replace_all) {
      break;
    }
  }
  return result;
}

struct AllowEmpty {
  bool operator()(Slice) const { return true; }
};
struct SkipEmpty {
  bool operator()(Slice sp) const { return !sp.empty(); }
};
struct SkipWhitespace {
  bool operator()(Slice sp) const {
    RightTrim(&sp);
    return !sp.empty();
  }
};

template <typename Predicate>
std::vector<std::string> Split(Slice text, Slice delims, Predicate p) {
  std::vector<std::string> result;
  size_t pos = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; ++i) {
      if (i == text.size() || (delims.find(text[i]) != Slice::npos)) {
        Slice token(text.data() + pos, i - pos);
        if (p(token)) {
          result.push_back(std::string(token));
        }
        pos = i + 1;
      }
    }
  }
  return result;
}
inline std::vector<std::string> Split(Slice text, Slice delims) {
  return Split(text, delims, AllowEmpty());
}
inline std::vector<std::string> Split(Slice text, char delims) {
  return Split(text, Slice(&delims, 1));
}
template <typename Predicate>
std::vector<std::string> Split(Slice text, char delims, Predicate p) {
  return Split(text, Slice(&delims, 1), p);
}

template <typename T>
std::string ToString(T arg) {
  std::stringstream ss;
  ss << arg;
  return ss.str();
}

template <typename T>
T StringTo(const std::string& s) {
  std::stringstream ss(s);
  T t;
  ss >> t;
  return t;
}

template <typename T, typename... Args>
std::string ToString(T arg, Args... args) {
  std::stringstream ss;
  ss << arg;
  return ss.str() + ToString(args...);
}

template <typename T>
std::string Join(const std::vector<T>& parts, const std::string& separator) {
  std::stringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << separator;
    }
    ss << parts[i];
  }
  return ss.str();
}

std::string JoinString(const std::vector<std::string>& parts,
                       const std::string& separator);

std::string JoinPath_impl_(std::initializer_list<Slice> paths);

template<typename... Args>
std::string JoinPath(const Args&... args) {
  return JoinPath_impl_({args...});
}

void ParseURI(Slice uri, Slice* scheme, Slice* host, Slice* path);

}  // namespace euler

#endif  // EULER_COMMON_STR_UTIL_H_
