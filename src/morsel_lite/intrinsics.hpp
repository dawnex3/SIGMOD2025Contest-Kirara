#pragma once

#include <cstdint>
#include <type_traits>

namespace Contest::MorselLite {

template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
[[nodiscard]] inline uint32_t popcount(T val) noexcept {
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return __builtin_popcount(val);
  } else {
    return __builtin_popcountll(val);
  }
}

//template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
//[[nodiscard]] inline uint32_t count_leading_zeros();

template<typename T, typename U>
[[nodiscard]] inline T bit_cast(U val) noexcept {
  static_assert(std::is_trivial_v<T> && std::is_trivial_v<U> && sizeof(T) == sizeof(U));
  T ret;
  __builtin_memcpy(&ret, &val, sizeof(U));
  return ret;
}

}