#pragma once

#include <cstdlib>
#include <cstdio>

#include "deps/span.hpp"

#define _ML_STRINGIFY_IMPL(...) #__VA_ARGS__
#define ML_STRINGIFY(...) _ML_STRINGIFY_IMPL(__VA_ARGS__)

#ifdef NDEBUG
# define ML_ASSERT(...)
#else
# define ML_ASSERT(...) do { if (!static_cast<bool>(__VA_ARGS__)) { \
   ::Contest::MorselLite::details::raise_assertion(__FILE__, __LINE__, ML_STRINGIFY(__VA_ARGS__)); \
   __builtin_trap(); } } while (false)
#endif

namespace Contest::MorselLite {

namespace details {

inline void raise_assertion(const char *filename, int line, const char *expr) {
  ::fprintf(stderr, "[%s:%d] assertion `%s` failed.", filename, line, expr);
}

}

template<typename T, size_t Extent = nonstd::dynamic_extent>
using Span = nonstd::span<T, Extent>;

template<typename T>
struct type_tag_t { using type = T; };

template<typename T>
inline constexpr type_tag_t<T> type_tag{};

#define ML_TYPE_OF_TAG(_tag) typename std::remove_cv_t<decltype(_tag)>::type

[[noreturn]] inline void unreachable_branch() noexcept {
  __builtin_unreachable();
}

template<typename T>
[[nodiscard]] std::unique_ptr<T []> make_unique_array_uninit(size_t size) {
  return std::unique_ptr<T []>(new T[size]);
}

[[nodiscard]] inline uint64_t idiv_ceil(uint64_t lhs, uint64_t rhs) noexcept {
  return lhs == 0 ? 0 : (lhs - 1) / rhs + 1;
}

// `lhs` should never be 0
[[nodiscard]] inline uint64_t idiv_ceil_nzero(uint64_t lhs, uint64_t rhs) noexcept {
  return (lhs - 1) / rhs + 1;
}

[[nodiscard]] inline bool validate_selection_indices(Span<const size_t> indices, size_t num_rows) {
  if (indices.empty()) { return true; }
  if (indices.back() >= num_rows) { return false; }
  for (size_t i = 1; i < indices.size(); ++i) {
    if (indices[i] <= indices[i - 1]) { return false; }
    if (indices[i] >= num_rows) { return false; }
  }
  return true;
}

template<typename BlockT, typename = std::enable_if_t<std::is_unsigned_v<BlockT>>>
[[nodiscard]] inline bool get_bit(const BlockT *bitmap, size_t index) noexcept {
  constexpr size_t WIDTH = sizeof(BlockT) * 8;

  const size_t block_index = index / WIDTH, bit_offset = index % WIDTH;
  return bitmap[block_index] & (static_cast<uint64_t>(1) << bit_offset);
}

template<typename BlockT, typename = std::enable_if_t<std::is_unsigned_v<BlockT>>>
inline void set_bit(BlockT *bitmap, size_t index) noexcept {
  constexpr size_t WIDTH = sizeof(BlockT) * 8;

  const size_t block_index = index / WIDTH, bit_offset = index % WIDTH;
  bitmap[block_index] |= (static_cast<uint32_t>(1) << bit_offset);
}

template<typename BlockT, typename = std::enable_if_t<std::is_unsigned_v<BlockT>>>
inline void unset_bit(BlockT *bitmap, size_t index) noexcept {
  constexpr size_t WIDTH = sizeof(BlockT) * 8;

  const size_t block_index = index / WIDTH, bit_offset = index % WIDTH;
  bitmap[block_index] &= ~(static_cast<uint32_t>(1) << bit_offset);
}

template<typename RandomAccessIterT, typename T>
RandomAccessIterT branchless_lower_bound(RandomAccessIterT first, size_t num, T target) {
  RandomAccessIterT base = first;
  size_t len = num;
  while (len > 0) {
    size_t half = len / 2;
    base += (base[half] < target) * (len - half);
    len = half;
  }
  return base;
}

template<typename RandomAccessIterT, typename T>
RandomAccessIterT branchless_upper_bound(RandomAccessIterT first, size_t num, T target) {
  RandomAccessIterT base = first;
  size_t len = num;
  while (len > 0) {
    size_t half = len / 2;
    base += (base[half] <= target) * (len - half);
    len = half;
  }
  return base;
}

}