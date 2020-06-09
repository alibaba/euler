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

#ifndef EULER_COMMON_HASH_H_
#define EULER_COMMON_HASH_H_

#include <stdint.h>

namespace euler {

// MurmurHash3 hash algorithms

// 32 bit hash
void hash32(void const *data, int size, uint32_t *hash, uint32_t seed = 0);

// 32 bit hash
inline uint32_t hash32(void const *data, int size, uint32_t seed = 0) {
  uint32_t hash;
  hash32(data, size, &hash, seed);
  return hash;
}


// 128 bit hash
void hash128(void const *data, int size, uint64_t *hash1, uint64_t *hash2,
             uint32_t seed = 0);

// 64 bit hash
inline void hash64(void const *data, int size, uint64_t *hash,
                   uint32_t seed = 0) {
  uint64_t dummy;
  return hash128(data, size, hash, &dummy, seed);
}

// 64 bit hash
inline uint64_t hash64(void const *data, int size, uint32_t seed = 0) {
  uint64_t hash;
  hash64(data, size, &hash, seed);
  return hash;
}

}  // namespace euler

#endif  // EULER_COMMON_HASH_H_
