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

#include "euler/common/hash.h"
#include <ctype.h>

namespace euler {

inline void rotl(uint32_t *x, int r) {
  *x = (*x << r) | (*x >> (32 - r));
}

inline void rotl(uint64_t *x, int r) {
  *x = (*x << r) | (*x >> (64 - r));
}

void hash32(void const *data, int size, uint32_t *hash, uint32_t seed) {
  // constants
  uint32_t const c1 = 0xcc9e2d51LU;
  uint32_t const c2 = 0x1b873593LU;
  uint32_t const c3 = 0xe6546b64LU;
  uint32_t const c4 = 0x85ebca6bLU;
  uint32_t const c5 = 0xc2b2ae35LU;


  // initialization
  uint32_t h = seed, k;

  // body
  uint32_t const *block = reinterpret_cast<uint32_t const *>(data);
  int const n = size >> 2;
  for (int i = 0; i < n; ++i) {
    k = *block++;
    k *= c1; rotl(&k, 15); k *= c2;
    h ^= k; rotl(&h, 13); h *= 5;  h += c3;
  }

  // tail
  k = 0;
  uint8_t const *tail = (uint8_t const *) block;
  switch (size & 3) {
    case 3: k ^= uint32_t(tail[2]) << 16;
    case 2: k ^= uint32_t(tail[1]) << 8;
    case 1: k ^= uint32_t(tail[0]);
      k *= c1; rotl(&k, 15); k *= c2; h ^= k;
  }

  // finalization
  h ^= size;
  h ^= h >> 16; h *= c4; h ^= h >> 13; h *= c5; h ^= h >> 16;

  // output
  *hash = h;
}

void hash128(void const *data, int size, uint64_t *hash1, uint64_t *hash2,
             uint32_t seed) {
  // constants
  uint64_t const c1 = 0x87c37b91114253d5LLU;
  uint64_t const c2 = 0x4cf5ad432745937fLLU;
  uint32_t const c3 = 0x52dce729LU;
  uint32_t const c4 = 0x38495ab5LU;
  uint64_t const c5 = 0xff51afd7ed558ccdLLU;
  uint64_t const c6 = 0xc4ceb9fe1a85ec53LLU;

  // initialization
  uint64_t h1 = seed, h2 = seed, k1, k2;

  // body
  uint64_t const *block = reinterpret_cast<uint64_t const *>(data);
  int const n = size >> 4;
  for (int i = 0; i < n; i++) {
    k1 = *block++; k2 = *block++;
    k1 *= c1; rotl(&k1, 31); k1 *= c2;
    h1 ^= k1; rotl(&h1, 27); h1 += h2; h1 *= 5; h1 += c3;
    k2 *= c2; rotl(&k2, 33); k2 *= c1;
    h2 ^= k2; rotl(&h2, 31); h2 += h1; h2 *= 5; h2 += c4;
  }

  // tail
  k1 = 0; k2 = 0;
  uint8_t const *tail = (uint8_t const *) block;
  switch (size & 15) {
    case 15: k2 ^= uint64_t(tail[14]) << 48;
    case 14: k2 ^= uint64_t(tail[13]) << 40;
    case 13: k2 ^= uint64_t(tail[12]) << 32;
    case 12: k2 ^= uint64_t(tail[11]) << 24;
    case 11: k2 ^= uint64_t(tail[10]) << 16;
    case 10: k2 ^= uint64_t(tail[ 9]) << 8;
    case  9: k2 ^= uint64_t(tail[ 8]) << 0;
      k2 *= c2; rotl(&k2, 33); k2 *= c1; h2 ^= k2;
    case  8: k1 ^= uint64_t(tail[ 7]) << 56;
    case  7: k1 ^= uint64_t(tail[ 6]) << 48;
    case  6: k1 ^= uint64_t(tail[ 5]) << 40;
    case  5: k1 ^= uint64_t(tail[ 4]) << 32;
    case  4: k1 ^= uint64_t(tail[ 3]) << 24;
    case  3: k1 ^= uint64_t(tail[ 2]) << 16;
    case  2: k1 ^= uint64_t(tail[ 1]) << 8;
    case  1: k1 ^= uint64_t(tail[ 0]) << 0;
      k1 *= c1; rotl(&k1, 31); k1 *= c2; h1 ^= k1;
  }

  // finalization
  h1 ^= size; h2 ^= size; h1 += h2; h2 += h1;
  h1 ^= h1 >> 33; h1 *= c5; h1 ^= h1 >> 33; h1 *= c6; h1 ^= h1 >> 33;
  h2 ^= h2 >> 33; h2 *= c5; h2 ^= h2 >> 33; h2 *= c6; h2 ^= h2 >> 33;
  h1 += h2; h2 += h1;

  // output
  *hash1 = h1;
  *hash2 = h2;
}

}  // namespace euler
