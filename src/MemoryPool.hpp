/**
* @file MemoryPool.hpp
* @brief Defines a high-performance, two-level memory pool for query execution.
*
* This file provides a memory management system designed to handle the allocation patterns
* typical in database query processing: numerous small-to-medium allocations during
* execution, followed by a single deallocation of all memory at the end. This avoids
* the overhead of repeated calls to `malloc`/`free`.
*
* The system consists of two main components:
*
* 1.  **`GlobalPool` (`global_mempool`):**
*     A singleton that pre-allocates a very large, contiguous block of memory from the
*     operating system, optionally using huge pages (`madvise(MADV_HUGEPAGE)`) to reduce
*     TLB misses and improve performance. This global pool acts as a wholesale memory
*     source for thread-local allocators.
*
* 2.  **`Allocator` (`local_allocator`):**
*     A `thread_local` object that serves as the primary interface for allocations
*     within a worker thread. It obtains large chunks of memory from the `GlobalPool`
*     and then serves smaller, per-thread allocation requests from these chunks using
*     simple pointer bumping. This design is extremely fast and lock-free on a per-thread basis.
*
* - **`LocalVector<T>`:** A convenience alias for `std::vector` that uses this memory
*   pool system, allowing STL containers to benefit from the performance improvements.
*
* The entire system can be disabled by defining `NO_USE_MEMPOOL`, which causes all
* allocations to fall back to the standard library's `malloc` and `free`. This is
* useful for debugging and comparison.
*/
#pragma once
#include <stdexcept>
#include <sys/mman.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <new> // Required for placement new
#include "hardware.h"
#include "Profiler.hpp"

/**
* @namespace mem
* @brief Low-level memory utilities for huge page allocation.
*/
namespace mem {
/**
* @brief Allocates a large block of memory, advising the OS to use huge pages.
* @param size The number of bytes to allocate.
* @return A pointer to the allocated memory.
*/
inline void* malloc_huge(size_t size) {
   void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ifdef __linux__
   // This is an advisory call; the OS is not guaranteed to use huge pages.
   madvise(p, size, MADV_HUGEPAGE);
#endif
   return p;
}

/**
* @brief Frees a block of memory previously allocated with `mmap`.
* @param p Pointer to the memory block.
* @param size The size of the memory block.
*/
inline void free_huge(void* p, size_t size) {
   if (munmap(p, size) != 0) {
       throw std::runtime_error("Memory unmapping failed.");
   }
}
} // namespace mem


/**
* @class GlobalPool
* @brief A singleton that manages a large, contiguous pool of memory for the entire application.
*
* It serves as a wholesale memory provider for thread-local `Allocator` instances.
* All allocations are served from a single pre-allocated region via atomic pointer bumping.
*/
class GlobalPool {
private:
   int8_t* start_ = nullptr;
   int8_t* end_ = nullptr;
   /// @brief Atomically points to the next available byte in the pool.
   std::atomic<int8_t*> next_ = nullptr;

   /// @brief The total size of the pre-allocated global memory pool.
   size_t pool_size_ = (size_t)16 * 1024 * 1024 * 1024; // 16 GB

   /// @brief Allocates a raw chunk of memory using huge pages.
   int8_t* newChunk(size_t size);
   int8_t* newChunkWithInit(size_t size);

#ifdef NO_USE_MEMPOOL
   // Fallback state when the memory pool is disabled.
   std::vector<void*> allocated_;
   size_t allocated_size_;
   std::mutex mtx_;
#endif

public:
   GlobalPool() = default;
   ~GlobalPool() = default;
   GlobalPool(GlobalPool&&) = delete;
   GlobalPool(const GlobalPool&) = delete;

   /// @brief Pre-allocates the large memory region for the pool.
   void init();
   /// @brief Releases the large memory region back to the OS.
   void destroy();
   /// @brief Atomically allocates a chunk of memory from the global pool.
   void* allocate(size_t size);
   /// @brief Resets the pool by moving the allocation pointer back to the start.
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
   if (start_) {
       mem::free_huge(start_, pool_size_);
       start_ = nullptr;
   }
#else
   for (void* alloc : allocated_) {
       std::free(alloc);
   }
   allocated_.clear();
#endif
}

inline int8_t* GlobalPool::newChunk(size_t size) {
   return (int8_t*)mem::malloc_huge(size);
}

inline int8_t* GlobalPool::newChunkWithInit(size_t size) {
   void* ptr = mem::malloc_huge(size);
   memset(ptr, 0xbe, size);
   return (int8_t*)ptr;
}

inline void* GlobalPool::allocate(size_t size) {
#ifndef NO_USE_MEMPOOL
   // Atomically bump the `next_` pointer to reserve a chunk of memory.
   return next_.fetch_add(size);
#else
   std::lock_guard lock(mtx_);
   void* ptr = malloc(size);
   allocated_.push_back(ptr);
   allocated_size_ += size;
   return ptr;
#endif
}

inline void GlobalPool::reset() {
#ifndef NO_USE_MEMPOOL
   next_ = start_;
#else
   for (void* alloc : allocated_) {
       std::free(alloc);
   }
   allocated_.clear();
   allocated_size_ = 0;
#endif
}


/**
* @class Allocator
* @brief A thread-local memory allocator that provides fast, lock-free allocations.
*
* Each thread has its own `Allocator` instance (`local_allocator`). It requests large
* chunks of memory from the `GlobalPool` and then sub-allocates from these chunks
* using simple (and very fast) pointer bumping.
*/
thread_local class Allocator {
private:
#ifndef NO_USE_MEMPOOL
   size_t init_alloc_ = 64 * 1024 * 1024;
#else
   size_t init_alloc_ = 1 * 1024 * 1024;
#endif
   size_t min_alloc_ = 1 * 1024 * 1024;
   uint8_t* start_ = nullptr; /// Pointer to the start of the current thread-local chunk.
   size_t free_ = 0;          /// Number of bytes remaining in the current chunk.
   GlobalPool* memory_source_ = nullptr;

public:
   Allocator() = default;
   Allocator(Allocator&&) = default;
   Allocator(const Allocator&) = delete;

   /// @brief Initializes the allocator with a global pool as its source.
   GlobalPool* init(GlobalPool* s);
   /// @brief Allocates memory from the current thread-local chunk.
   inline void* allocate(size_t size);
   /// @brief Discards the current chunk and gets a new one, effectively resetting the allocator for this thread.
   void reuse();
   /// @brief Allocates memory and constructs an object of type T in place.
   template <typename T, typename... Args> T* make(Args&&... args);
} local_allocator;

template <typename T, typename... Args>
inline T* Allocator::make(Args&&... args) {
   // 1. Allocate raw memory.
   void* ptr = allocate(sizeof(T));
   if (!ptr) {
       return nullptr;
   }
   // 2. Use placement new to construct the object in the allocated memory.
   return new (ptr) T(std::forward<Args>(args)...);
}

inline void* Allocator::allocate(size_t size) {
   // Ensure 64-byte alignment for performance (e.g., SIMD).
   auto aligndiff = (64 - ((uintptr_t)start_ % 64)) % 64;
   size_t required_size = size + aligndiff;

   // If the current chunk doesn't have enough space, get a new one.
   if (free_ < required_size) {
       size_t alloc_size = std::max(min_alloc_, required_size);
       start_ = (uint8_t*)memory_source_->allocate(alloc_size);
       free_ = alloc_size;
       aligndiff = (64 - ((uintptr_t)start_ % 64)) % 64;
       required_size = size + aligndiff;
   }

   // Bump the pointer to serve the allocation.
   uint8_t* alloc = start_ + aligndiff;
   start_ += required_size;
   free_ -= required_size;
   return alloc;
}

inline GlobalPool* Allocator::init(GlobalPool* source) {
   auto previousSource = memory_source_;
   memory_source_ = source;
   if (source) {
       start_ = (uint8_t*)memory_source_->allocate(init_alloc_);
       free_ = init_alloc_;
   }
   return previousSource;
}

inline void Allocator::reuse() {
   start_ = (uint8_t*)memory_source_->allocate(init_alloc_);
   free_ = init_alloc_;
}


/**
* @class LocalAllocatorWrapper
* @brief An adapter to make the thread-local `Allocator` compatible with the STL allocator interface.
*
* This allows standard containers like `std::vector` to use our custom memory pool.
* @tparam T The type of the elements to be allocated.
*/
template <typename T>
class LocalAllocatorWrapper {
public:
   using value_type = T;

   LocalAllocatorWrapper() = default;

   template <typename U> LocalAllocatorWrapper(const LocalAllocatorWrapper<U>&) noexcept {}

   T* allocate(std::size_t n) {
       return static_cast<T*>(local_allocator.allocate(n * sizeof(T)));
   }

   void deallocate(T* p, std::size_t n) noexcept {
       // No-op: Memory is reclaimed by resetting the pool, not by individual deallocations.
   }

   // All instances of this allocator are considered equal.
   bool operator!=(const LocalAllocatorWrapper&) const { return false; }
   bool operator==(const LocalAllocatorWrapper&) const { return true; }

   template <typename U, typename... Args> void construct(U* p, Args&&... args) {
       new (p) U(std::forward<Args>(args)...);
   }

   template <typename U> void destroy(U* p) { p->~U(); }
};

/**
* @brief A type alias for `std::vector` that uses the thread-local memory pool.
*/
template <typename T>
using LocalVector = std::vector<T, LocalAllocatorWrapper<T>>;