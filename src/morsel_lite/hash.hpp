#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#define XXH_INLINE_ALL
#include "deps/xxhash.h"

namespace Contest::MorselLite {

inline uint64_t hash_bytes(const uint8_t *bytes, size_t size) {
  return XXH64(bytes, size, 114514);
}

template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
inline uint64_t hash_integer(T val) {
  return hash_bytes(reinterpret_cast<const uint8_t *>(&val), sizeof(val));
}

}