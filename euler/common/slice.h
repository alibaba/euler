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


#ifndef EULER_COMMON_SLICE_H_
#define EULER_COMMON_SLICE_H_

#include <assert.h>
#include <string.h>

#include <ostream>
#include <string>

namespace euler {

class Slice {
 public:
  typedef size_t size_type;
  static const size_t npos = size_type(-1);

  Slice() : data_(nullptr), size_(0) {}
  Slice(const char* d, size_t n) : data_(d), size_(n) {}
  Slice(const std::string& s) : data_(s.data()), size_(s.size()) {}  // NOLINT
  Slice(const char* s) : data_(s), size_(strlen(s)) {}  // NOLINT

  const char* data() const { return data_; }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  typedef const char* const_iterator;
  typedef const char* iterator;

  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }

  char operator[](size_t n) const {
    assert(n < size());
    return data_[n];
  }

  void remove_prefix(size_t n) {
    assert(n <= size());
    data_ += n;
    size_ -= n;
  }

  void remove_suffix(size_t n) {
    assert(size_ >= n);
    size_ -= n;
  }

  size_t find(Slice pattern, size_t pos = 0);
  size_t find(char c, size_t pos = 0) const;
  size_t rfind(char c, size_t pos = npos) const;

  Slice substr(size_t pos, size_t n = npos) const;

  std::string ToString() const { return std::string(data_, size_); }

  int compare(Slice b) const;

  template <typename A>
  explicit operator std::basic_string<char, std::char_traits<char>, A>() const {
    if (!data()) return {};
    return std::basic_string<char, std::char_traits<char>, A>(data(), size());
  }

 private:
  const char* data_;
  size_t size_;
};

inline bool operator==(Slice x, Slice y) {
  return ((x.size() == y.size()) &&
          (memcmp(x.data(), y.data(), x.size()) == 0));
}

inline bool operator!=(Slice x, Slice y) { return !(x == y); }

inline bool operator<(Slice x, Slice y) { return x.compare(y) < 0; }
inline bool operator>(Slice x, Slice y) { return x.compare(y) > 0; }
inline bool operator<=(Slice x, Slice y) {
  return x.compare(y) <= 0;
}
inline bool operator>=(Slice x, Slice y) {
  return x.compare(y) >= 0;
}

inline int Slice::compare(Slice b) const {
  const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
  int r = memcmp(data_, b.data_, min_len);
  if (r == 0) {
    if (size_ < b.size_)
      r = -1;
    else if (size_ > b.size_)
      r = +1;
  }
  return r;
}

extern std::ostream& operator<<(std::ostream& o, Slice piece);

}  // namespace euler

#endif  // EULER_COMMON_SLICE_H_
