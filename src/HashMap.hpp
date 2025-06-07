/**
* @file Hashmap.hpp
* @brief Defines a high-performance, concurrent hash map and associated hashing functions.
*
* This file provides a `Hashmap` class optimized for the high-throughput
* demands of a parallel hash join operator.
*/

#pragma once
#include "atomic"
#include <cstdlib>
#include <cassert>
#include <mutex>
#include <vector>
#include "hardware.h"
#include "MemoryPool.hpp"

namespace Contest {

// --- Constants and SIMD Definitions ---

/// @brief A reserved hash value used to represent the hash of a NULL key.
/// Keys that hash to this value are ignored during insertion.
#define NULL_HASH (1642857263)
/// @brief The number of elements processed in a single SIMD operation.
#define SIMD_SIZE 8
/// @brief A helper macro to initialize all elements of a SIMD vector with the same value.
#define INIT_MACRO(X) X,X,X,X,X,X,X,X
/// @brief A SIMD vector type representing 8 unsigned 32-bit integers, using GCC/Clang vector extensions.
typedef uint32_t v8u32 __attribute__((__vector_size__(sizeof(uint32_t) * SIMD_SIZE)));

/**
 * @class Hashmap
 * @brief A high-performance hash map tailored for hash join operations.
 *
 * This implementation uses separate chaining for collision resolution. It is
 * optimized for multi-threaded performance, featuring:
 * - **Concurrent/Non-concurrent Modes:** Templated methods support both lock-free
 *   atomic insertions for parallel builds and faster, non-atomic insertions for
 *   single-threaded builds.
 * - **Pointer Tagging:** The upper 16 bits of bucket pointers are used as a compact
 *   Bloom filter (tags). This allows for a quick check to filter out most non-matching
 *   probes without dereferencing the pointer, significantly reducing random memory access.
 */
class Hashmap {
public:
    using hash_t = uint32_t;
    using ptr_t = uint64_t;

    /// @brief The number of buckets in the hash table (always a power of 2).
    size_t capacity = 0;

    /**
     * @struct EntryHeader
     * @brief The header for each entry stored in the hash map's chains.
     *
     * The actual payload (the rest of the tuple's data) is not part of this struct
     * but immediately follows it in memory.
     */
    struct EntryHeader {
        EntryHeader* next; /// Pointer to the next entry in the bucket's chain.
        hash_t hash;       /// The pre-computed hash of the key.
        uint32_t key;      /// The join key.
        EntryHeader(EntryHeader* n, hash_t hash_, uint32_t key_) : next(n), hash(hash_), key(key_) {}
        inline hash_t getHash() { return hash; }
    };

    /// @brief Finds the head of a bucket's chain after a fast tag-based filter.
    inline EntryHeader* find_chain_tagged(hash_t hash);

    /// @brief Inserts a new entry into the hash map (basic version without tagging).
    template <bool concurrentInsert = true>
    inline void insert(EntryHeader* entry, hash_t hash);

    /// @brief Inserts a new entry and updates the bucket's tag.
    template <bool concurrentInsert = true>
    inline void insert_tagged(EntryHeader* entry, hash_t hash);

    /// @brief Inserts a batch of `n` entries using the tagged insertion logic.
    template <bool concurrentInsert = true>
    inline void insertAll_tagged(EntryHeader* first, size_t n, size_t step);

    /// @brief Sets the size of the hash map, allocating buckets using `malloc`.
    inline size_t setSize(size_t nrEntries);

    /// @brief Sets the size of the hash map, allocating buckets using the `MemoryPool`.
    inline size_t setSizeUseMemPool(size_t nrEntries);

    /// @brief Clears all entries from the hash table, resetting it to an empty state.
    inline void clear();

    /**
     * @brief Prefetches the memory for a bucket's chain head into the CPU cache.
     * @tparam rw 0 for read, 1 for write prefetch.
     * @tparam locality A hint from 0 (low temporal locality) to 3 (high).
     */
    template <size_t rw, size_t locality>
    inline void prefetchBucket(hash_t hash) {
        EntryHeader* bucket_ptr = entries[hash & mask].load(std::memory_order_relaxed);
        __builtin_prefetch(bucket_ptr, rw, locality);
    }

    /// @brief The bucket array. Each element is an atomic pointer to the head of a chain.
    std::atomic<EntryHeader*>* entries = nullptr;
    /// @brief Flag indicating if `entries` was allocated from the memory pool.
    bool entries_use_mem_pool = false;
    /// @brief A bitmask used to map a hash value to a bucket index (equivalent to `hash % capacity`).
    hash_t mask{};

    // --- Pointer Tagging Masks ---
    /// @brief A bitmask to extract the lower 48 bits of a 64-bit pointer, yielding the actual memory address.
    const ptr_t maskPointer = (~(ptr_t)0) >> (16);
    /// @brief A bitmask to extract the upper 16 bits of a pointer, which are used as a compact Bloom filter (tags).
    const ptr_t maskTag = (~(ptr_t)0) << (sizeof(ptr_t) * 8 - 16);

    /// @brief A sentinel value indicating the end of a chain or a miss after tag filtering.
    inline static EntryHeader* end();

    Hashmap() = default;
    Hashmap(const Hashmap&) = delete;
    inline ~Hashmap();

    /**
     * @brief Registers memory allocations associated with hash table entries.
     * @param alloc A vector of memory chunks.
     * @param entry_size The size of a single entry to calculate total memory.
     */
    void addAllocations(const std::vector<std::pair<uint8_t*, size_t>>& alloc, size_t entry_size) {
        size_t alloc_size = 0;
        for (auto [_, n] : alloc) {
            alloc_size += n * entry_size;
        }
        std::lock_guard lock(m_);
        allocations_.insert(allocations_.end(), alloc.begin(), alloc.end());
        total_mem_size_ += alloc_size;
    }

    /// @brief Returns the total memory size allocated for this hash map.
    [[nodiscard]] size_t getMemSize() const {
        return total_mem_size_;
    }

private:
    /// @brief Extracts the raw pointer (address) from a tagged pointer.
    inline EntryHeader* ptr(EntryHeader* p);
    /// @brief Calculates a 16-bit tag from a hash value.
    inline ptr_t tag(hash_t p);
    /// @brief Creates a new tagged pointer by combining a new address, old tags, and a new tag.
    inline EntryHeader* update(EntryHeader* old, EntryHeader* p, hash_t hash);

    std::mutex m_;
    size_t total_mem_size_{0};
    std::vector<std::pair<uint8_t*, size_t>> allocations_;
};


inline Hashmap::EntryHeader* Hashmap::end() { return nullptr; }

inline Hashmap::~Hashmap() {
    if (entries && !entries_use_mem_pool) {
        free(entries);
    }
    // Free all memory chunks registered for the hash table entries.
    for (auto [p, n] : allocations_) {
        if (p) free(p);
    }
}

/**
 * @brief Calculates a 16-bit tag from a hash value.
 * It uses the top 4 bits of the hash to select one of 16 bit positions in the tag space.
 */
inline Hashmap::ptr_t Hashmap::tag(Hashmap::hash_t hash) {
    auto tagPos = hash >> (sizeof(hash_t) * 8 - 4);
    return ((size_t)1) << (tagPos + (sizeof(ptr_t) * 8 - 16));
}

/// @brief Extracts the real address from a tagged pointer by clearing the upper 16 tag bits.
inline Hashmap::EntryHeader* Hashmap::ptr(Hashmap::EntryHeader* p) {
    return (EntryHeader*)((ptr_t)p & maskPointer);
}

/**
 * @brief Combines a new entry's address with existing tags to form a new tagged pointer.
 * The new pointer will contain the new address, OR'ed with the tags from the `old`
 * pointer, and OR'ed with the tag calculated from the new entry's `hash`.
 */
inline Hashmap::EntryHeader* Hashmap::update(EntryHeader* old, EntryHeader* p, hash_t hash) {
    return reinterpret_cast<EntryHeader*>((size_t)p | ((size_t)old & maskTag) | tag(hash));
}

template <bool concurrentInsert>
void inline Hashmap::insert(EntryHeader* entry, hash_t hash) {
    if (hash == NULL_HASH) return;

    const size_t pos = hash & mask;
    if (concurrentInsert) {
        auto locPtr = &entries[pos];
        EntryHeader* loc = locPtr->load();
        do {
            entry->next = loc;
        } while (!locPtr->compare_exchange_weak(loc, entry));
    } else {
        auto& loc = entries[pos];
        entry->next = loc.load(std::memory_order_relaxed);
        loc.store(entry, std::memory_order_relaxed);
    }
}

/**
 * @brief Performs a fast lookup by first checking the pointer tag.
 * If the tag bit corresponding to the hash is set in the bucket pointer, it returns
 * the real pointer to the chain head. Otherwise, it returns `end()` immediately.
 */
inline Hashmap::EntryHeader* Hashmap::find_chain_tagged(hash_t hash) {
    if (__glibc_unlikely(hash == NULL_HASH)) return nullptr;

    auto pos = hash & mask;
    auto candidate = entries[pos].load(std::memory_order_relaxed);

    // Check if the tag bit for this hash is present in the bucket pointer.
    if ((size_t)candidate & tag(hash))
        return ptr(candidate); // Tag match: likely in chain, return real pointer.
    else
        return end(); // Tag miss: definitely not in chain.
}

template <bool concurrentInsert>
void inline Hashmap::insert_tagged(EntryHeader* entry, hash_t hash) {
    if (hash == NULL_HASH) return;

    const size_t pos = hash & mask;
    if (concurrentInsert) {
        auto locPtr = &entries[pos];
        EntryHeader* loc = locPtr->load();
        EntryHeader* newLoc;
        // This is a standard CAS loop for lock-free insertion into a linked list.
        do {
            // The new entry points to the old head of the chain.
            entry->next = ptr(loc);
            // The new head of the chain is our new entry, with its tag OR'd into the pointer.
            newLoc = update(loc, entry, hash);
        } while (!locPtr->compare_exchange_weak(loc, newLoc));
    } else {
        auto& loc = entries[pos];
        auto oldValue = loc.load(std::memory_order_relaxed);
        entry->next = ptr(oldValue);
        loc.store(update(oldValue, entry, hash), std::memory_order_relaxed);
    }
}

template <bool concurrentInsert>
void inline Hashmap::insertAll_tagged(EntryHeader* first, size_t n, size_t step) {
    EntryHeader* e = first;
    for (size_t i = 0; i < n; ++i) {
        insert_tagged<concurrentInsert>(e, static_cast<hash_t>(e->hash));
        e = reinterpret_cast<EntryHeader*>(reinterpret_cast<uint8_t*>(e) + step);
    }
}

size_t inline Hashmap::setSize(size_t nrEntries) {
    if (entries && !entries_use_mem_pool) {
        free(entries);
    }
    entries_use_mem_pool = false;

    // Calculate capacity as the next power of 2 that respects the load factor.
    const auto loadFactor = 0.7;
    size_t exp = (nrEntries > 0) ? 64 - __builtin_clzll(nrEntries) : 0;
    if (((size_t)1 << exp) < nrEntries / loadFactor) {
        exp++;
    }
    capacity = ((size_t)1) << exp;
    mask = capacity - 1;
    entries = static_cast<std::atomic<EntryHeader*>*>(malloc(capacity * sizeof(std::atomic<EntryHeader*>)));
    memset((void*)entries, 0, capacity * sizeof(std::atomic<EntryHeader*>));

    total_mem_size_ = capacity * sizeof(std::atomic<EntryHeader*>);
    return capacity * loadFactor;
}

size_t inline Hashmap::setSizeUseMemPool(size_t nrEntries) {
    if (entries && !entries_use_mem_pool) {
        free(entries);
    }
    entries_use_mem_pool = true;

    const auto loadFactor = 0.7;
    size_t exp = (nrEntries > 0) ? 64 - __builtin_clzll(nrEntries) : 0;
    if (((size_t)1 << exp) < nrEntries / loadFactor) {
        exp++;
    }
    capacity = ((size_t)1) << exp;
    mask = capacity - 1;
    entries = static_cast<std::atomic<EntryHeader*>*>(local_allocator.allocate(capacity * sizeof(std::atomic<EntryHeader*>)));
    memset((void*)entries, 0, capacity * sizeof(std::atomic<EntryHeader*>));

    total_mem_size_ = capacity * sizeof(std::atomic<EntryHeader*>);
    return capacity * loadFactor;
}

void inline Hashmap::clear() {
    if (entries) {
        for (size_t i = 0; i < capacity; i++) {
            entries[i].store(end(), std::memory_order_relaxed);
        }
    }
}


// --- Hashing Functions (Scalar and SIMD) ---

#ifdef SIMD_SIZE
/// @brief Vectorized 32-bit rotate left operation.
static inline v8u32 rotl32(v8u32 x, int r) {
    return (x << r) | (x >> (32 - r));
}

/// @brief Vectorized finalization mix for MurmurHash3.
static inline v8u32 fmix32(v8u32 h) {
    h ^= (h >> 16);
    h *= 0x85ebca6b;
    h ^= (h >> 13);
    h *= 0xc2b2ae35;
    h ^= (h >> 16);
    return h;
}

/// @brief Core of the vectorized MurmurHash3 algorithm, processing 8 keys at once.
static inline v8u32 hash_32(v8u32 key, v8u32 seed) {
    key *= 0xcc9e2d51;
    key = rotl32(key, 15);
    key *= 0x1b873593;

    seed ^= key;
    seed = rotl32(seed, 13);
    seed = seed * 5 + 0xe6546b64;

    return fmix32(seed);
}

/**
 * @brief Computes hashes for an array of keys using SIMD instructions.
 * @param keys Pointer to an array of 8 keys to be hashed.
 * @param hashes Pointer to an array where the 8 computed hashes will be stored.
 */
void inline compute_hashes(const uint32_t* keys, uint32_t* hashes) {
    constexpr v8u32 seed = {INIT_MACRO(4000932304U)};
    *reinterpret_cast<v8u32*>(hashes) = hash_32(*reinterpret_cast<const v8u32*>(keys), seed);
}

#endif

/// @brief Scalar (non-vectorized) 32-bit rotate left operation.
inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

/// @brief Scalar (non-vectorized) finalization mix for MurmurHash3.
inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

/**
 * @brief Computes a hash for a single 32-bit key using the MurmurHash3 algorithm.
 * @param key The key to hash.
 * @param seed A seed value for the hash function.
 * @return The computed 32-bit hash.
 */
uint32_t hash_32(uint32_t key, uint32_t seed = 4000932304) {
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    uint32_t k1 = key;
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;

    return fmix32(h1);
}

} // namespace Contest