#pragma once
#include <stdexcept>
#include <sys/mman.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <mutex>
#include "hardware.h"
#include "Profiler.hpp"

#ifdef SPC__PPC64LE
#define NO_USE_MEMPOOL
#endif

namespace mem {
inline void* malloc_huge(size_t size) {
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ifdef __linux__
   madvise(p, size, MADV_HUGEPAGE);
#endif
   // memset(p, 0, size);
   return p;
}

inline void free_huge(void* p, size_t size) {
   auto r = munmap(p, size);
   if (r) throw std::runtime_error("Memory unmapping failed.");
}
} // namespace mem



class GlobalPool {
private:
  int8_t *start_ = nullptr;
  int8_t *end_ = nullptr;
  std::atomic<int8_t *> next_ = nullptr;

  size_t pool_size_ = (size_t)16 * 1024 * 1024 * 1024;

  int8_t *newChunk(size_t size);
  int8_t *newChunkWithInit(size_t size);

#ifdef NO_USE_MEMPOOL
  std::vector<void *> allocated_;
  size_t allocated_size_;
  std::mutex mtx_;
#endif

public:
  GlobalPool() = default;
  ~GlobalPool() = default;
  GlobalPool(GlobalPool &&) = delete;
  GlobalPool(const GlobalPool &) = delete;

  void init();
  void destroy();
  void *allocate(size_t size);
  void reset();
} global_mempool;

inline void GlobalPool::init() {
#ifndef NO_USE_MEMPOOL
  ProfileGuard profile_guard(global_profiler, "GlobalPool::init");
  start_ = newChunk(pool_size_);
  end_ = start_ + pool_size_;
  next_ = start_;
#endif
}

inline void GlobalPool::destroy() {
#ifndef NO_USE_MEMPOOL
  mem::free_huge(start_, pool_size_);
  start_ = nullptr;
  end_ = nullptr;
  next_ = nullptr;
#else
  for (void *alloc : allocated_) {
    std::free(alloc);
  }
  allocated_.clear();
#endif
}

inline int8_t *GlobalPool::newChunk(size_t size) {
  return (int8_t *)mem::malloc_huge(size);
}

inline int8_t *GlobalPool::newChunkWithInit(size_t size) {
  void *ptr = mem::malloc_huge(size);
  memset(ptr, 0xbe, size);
  return (int8_t *)ptr;
}

inline void *GlobalPool::allocate(size_t size) {
#ifndef NO_USE_MEMPOOL
  int8_t *tmp = nullptr;
  int8_t *alloc = nullptr;

  do {
    tmp = next_.load();
    if (tmp + size >= end_) {
      throw std::bad_alloc();
    }
    alloc = tmp;
  } while (!next_.compare_exchange_weak(tmp, tmp + size));

  return alloc;
#else
  std::lock_guard lock(mtx_);
  void *ptr = malloc(size);
  allocated_.push_back(ptr);
  allocated_size_ += size;
  return ptr;
#endif
}

inline void GlobalPool::reset() {
#ifndef NO_USE_MEMPOOL
  next_ = start_;
#else
  for (void *alloc : allocated_) {
    std::free(alloc);
  }
  allocated_.clear();
//  fmt::println("sql use memory: {}", allocated_size_);
  allocated_size_ = 0;
#endif
}



thread_local class Allocator {
private:
  // start with a huge page
#ifndef NO_USE_MEMPOOL
  size_t init_alloc_ = 64 * 1024 * 1024;
#else
  size_t init_alloc_ = 1 * 1024 * 1024;
#endif
  size_t min_alloc_ = 1 * 1024 * 1024;
  uint8_t *start_ = nullptr;
  size_t free_ = 0;
  GlobalPool *memory_source_ = nullptr;

public:
  Allocator() = default;
  Allocator(Allocator &&) = default;
  Allocator(const Allocator &) = delete;
  GlobalPool *init(GlobalPool *s);
  inline void *allocate(size_t size);
  void reuse();
} local_allocator;

inline void *Allocator::allocate(size_t size) {
  auto aligndiff = 64 - ((uintptr_t)start_ % 64);
  size += aligndiff;
  if (free_ < size) {
    size_t alloc_size = std::max(min_alloc_, size + 64);
    start_ = (uint8_t *)memory_source_->allocate(alloc_size);

    aligndiff = 64 - ((uintptr_t)start_ % 64);
    size += aligndiff;
    free_ = alloc_size;
  }
  auto alloc = start_ + aligndiff;
  start_ += size;
  free_ -= size;
  return alloc;
}

inline GlobalPool *Allocator::init(GlobalPool *source) {
  auto previousSource = memory_source_;
  memory_source_ = source;
  if (source) {
    start_ = (uint8_t *)memory_source_->allocate(init_alloc_);
    free_ = init_alloc_;
  }
  return previousSource;
}

inline void Allocator::reuse() {
  start_ = (uint8_t *)memory_source_->allocate(init_alloc_);
  free_ = init_alloc_;
}

template <typename T>
class LocalAllocatorWrapper {
public:
  using value_type = T;

  LocalAllocatorWrapper() = default;

  template <typename U> LocalAllocatorWrapper(const LocalAllocatorWrapper<U> &) noexcept {}

  T *allocate(std::size_t n) {
    return static_cast<T *>(local_allocator.allocate(n * sizeof(T)));
  }

  bool operator!=(const LocalAllocatorWrapper&) const { return false; } // 假设所有分配器都相等

  void deallocate(T *p, std::size_t n) noexcept {
    // donothing ...
  }

  template <typename U, typename... Args> void construct(U *p, Args &&...args) {
    new (p) U(std::forward<Args>(args)...);
  }

  template <typename U> void destroy(U *p) { p->~U(); }
};

template <typename T>
using LocalVector = std::vector<T, LocalAllocatorWrapper<T>>;