#pragma once
#include "atomic"
#include <cstdlib>
#include <cassert>
#include "hardware.h"
#include "MemoryPool.hpp"

namespace Contest {

// the hash value of NULL_INT32. If hash(key) == NULL_HASH, `key` is considered NULL
#define NULL_HASH (1642857263)
// use vector extensions of clang to simplify SIMD code
#define SIMD_SIZE 8
#define INIT_MACRO(X) X,X,X,X,X,X,X,X
typedef uint32_t v8u32 __attribute__((__vector_size__(sizeof(uint32_t) * SIMD_SIZE)));

// Hashmap consists of buckets stored in `entries`, the number of buckets is `capacity`
// The elements in each bucket are pointers to `EntryHeader`, not the 'key' along with payload data
class Hashmap {
public:
    using hash_t = uint32_t;
    using ptr_t = uint64_t;

    size_t capacity = 0;
    // Header for hashtable entries
    struct EntryHeader
    {
        EntryHeader* next;
        hash_t hash;
        uint32_t key;
        EntryHeader(EntryHeader* n, hash_t hash_, uint32_t key_) : next(n), hash(hash_), key(key_) { }
        inline hash_t getHash() {return hash;}
        // payload data follows this header (see Operator.hpp)
    };

    /// Returns the first entry of the chain for the given hash
    /// Uses pointer tagging as a filter to quickly determine whether hash is contained
    inline EntryHeader* find_chain_tagged(hash_t hash);

    /// Insert entry into chain for the given hash
    template <bool concurrentInsert = true>
    inline void insert(EntryHeader* entry, hash_t hash);
    /// Insert entry into chain for the given hash
    /// Updates tag
    template <bool concurrentInsert = true>
    inline void insert_tagged(EntryHeader* entry, hash_t hash);
    /// Insert n entries starting from first, always looking for the next entry
    /// step bytes after the previous
    template <bool concurrentInsert = true>
    inline void insertAll_tagged(EntryHeader* first, size_t n, size_t step);
    /// Set size (no resize functionality)
    inline size_t setSize(size_t nrEntries);
    inline size_t setSizeUseMemPool(size_t nrEntries);
    /// Removes all elements from the hashtable
    inline void clear();

    template <size_t rw, size_t locality>
    inline void prefetchBucket(hash_t hash) {
        // Load the actual bucket pointer from the entries array (using relaxed memory order)
        EntryHeader* bucket_ptr = entries[hash & mask].load(std::memory_order_relaxed);
        __builtin_prefetch(bucket_ptr, rw, locality);
    }

    // `entries` stores all buckets
    std::atomic<EntryHeader*>* entries = nullptr;
    // if entries_use_mem_pool == false, the memory `entries` used are from malloc
    // otherwise, it is allocated from memory pool
    bool entries_use_mem_pool = false;

    // Map any hash value to the index range of the bucket array, i.e. 0~capacity-1
    hash_t mask{};

    // a tagged pointer consists of two parts: high 16bits and low 48bits.
    // extract the real address (low 48 bits) from a tagged pointer.
    const ptr_t maskPointer = (~(ptr_t)0) >> (16);
    // extract the tag part of the pointer (high 16 bits). Tags are used to quickly determine if key is in the bucket
    const ptr_t maskTag = (~(ptr_t)0) << (sizeof(ptr_t) * 8 - 16);

    inline static EntryHeader* end();
    Hashmap() = default;
    Hashmap(const Hashmap&) = delete;
    inline ~Hashmap();

    // increase `total_mem_size_` and record new memory usage by `alloc`
    void addAllocations(const std::vector<std::pair<uint8_t*, size_t>>& alloc, size_t entry_size){
        size_t alloc_size = 0;
        for(auto [_,n]:alloc){
            alloc_size += n*entry_size;
        }
        std::lock_guard lock(m_);
        allocations_.insert(allocations_.end(),alloc.begin(),alloc.end());
        total_mem_size_ += alloc_size;
    }

    [[nodiscard]] size_t getMemSize() const{
        return total_mem_size_;
    }

private:
    inline Hashmap::EntryHeader* ptr(Hashmap::EntryHeader* p);
    inline ptr_t tag(hash_t p);
    inline Hashmap::EntryHeader* update(Hashmap::EntryHeader* old,
        Hashmap::EntryHeader* p, hash_t hash);

    std::mutex m_;
    size_t total_mem_size_{0};
    std::vector<std::pair<uint8_t*, size_t>> allocations_;
};

extern Hashmap::EntryHeader notFound;

inline Hashmap::EntryHeader* Hashmap::end() { return nullptr; }

inline Hashmap::~Hashmap() {
    if (entries && !entries_use_mem_pool){
        free(entries);
    }
    // free all recorded memory in `allocations_`
    for(auto [p,n]:allocations_){
        if(p) free(p);
    }
}

inline Hashmap::ptr_t Hashmap::tag(Hashmap::hash_t hash) {
    auto tagPos = hash >> (sizeof(hash_t) * 8 - 4);    // extract high 4 bits of hash as `tagPos`
    return ((size_t)1) << (tagPos + (sizeof(ptr_t) * 8 - 16));  // `tag` is 1 << (48 + tagPos)
}

inline Hashmap::EntryHeader* Hashmap::ptr(Hashmap::EntryHeader* p) {
    return (EntryHeader*)((ptr_t)p & maskPointer);
}

inline Hashmap::EntryHeader* Hashmap::update(Hashmap::EntryHeader* old,
    Hashmap::EntryHeader* p,
    Hashmap::hash_t hash) {
    // high 16 bits of a 64-bits pointer are always zero, so we can directly use `(size_t)p`
    return reinterpret_cast<EntryHeader*>((size_t)p | ((size_t)old & maskTag) |
                                          tag(hash));
}

template <bool concurrentInsert>
void inline Hashmap::insert(EntryHeader* entry, hash_t hash) {
    if(hash == NULL_HASH) return;   // return directly if hash is the value of NULL

    const size_t pos = hash & mask;
    if (concurrentInsert) {
        auto locPtr = &entries[pos];
        EntryHeader* loc = locPtr->load();
        EntryHeader* newLoc;
        // Attempt to update the value of the bucket by comparing and swapping atomic operations.
        // If the value of the current bucket is equal to loc, update it to newLoc.
        // If other threads make modifications to the bucket during this period, the update will fail and the loc will be updated to the current actual value.
        // The loop will repeat until the update is successful.
        do {
            entry->next = loc;
            newLoc = entry;
        } while (!locPtr->compare_exchange_weak(loc, newLoc));
    } else {
        auto& loc = entries[pos];
        // std::memory_order_relaxed, atomicity and orderliness are not guaranteed
        auto oldValue = loc.load(std::memory_order_relaxed);
        entry->next = oldValue;
        loc.store(entry, std::memory_order_relaxed);
    }
}

inline Hashmap::EntryHeader* Hashmap::find_chain_tagged(hash_t hash) {
    if(__glibc_unlikely(hash == NULL_HASH)) return nullptr;   // return nullptr if hash is the value of NULL

    auto pos = hash & mask;
    auto candidate = entries[pos].load(std::memory_order_relaxed);
    // Quickly determine whether the hash is in the chain bucket through the tag.
    auto filterMatch = (size_t)candidate & tag(hash);
    if (filterMatch)
      return ptr(candidate);
    else
      return end();
}

template <bool concurrentInsert>
void inline Hashmap::insert_tagged(EntryHeader* entry, hash_t hash) {
    if(hash == NULL_HASH) return;   // return directly if hash is the value of NULL

    const size_t pos = hash & mask;
    if (concurrentInsert) {
        auto locPtr = &entries[pos];
        EntryHeader* loc = locPtr->load();
        EntryHeader* newLoc;
        do {
            entry->next = ptr(loc);
            newLoc = update(loc, entry, hash);
        } while (!locPtr->compare_exchange_weak(loc, newLoc));
    } else {
        auto& loc = entries[pos];
        auto oldValue = loc.load(std::memory_order_relaxed);
        entry->next = ptr(oldValue);
        loc.store(update(loc, entry, hash), std::memory_order_relaxed);
    }
}

template <bool concurrentInsert>
void inline Hashmap::insertAll_tagged(EntryHeader* first, size_t n,
    size_t step) {
    EntryHeader* e = first;
    for (size_t i = 0; i < n; ++i) {
        insert_tagged<concurrentInsert>(e, static_cast<hash_t>(e->hash));
        e = reinterpret_cast<EntryHeader*>(reinterpret_cast<uint8_t*>(e) + step);
    }
}

// not using memory pool
size_t inline Hashmap::setSize(size_t nrEntries) {
    if (entries && !entries_use_mem_pool){
        free(entries);
    }
    entries_use_mem_pool = false;

    const auto loadFactor = 0.7;
    size_t exp = 64 - __builtin_clzll(nrEntries);
    if (((size_t)1 << exp) < nrEntries / loadFactor){
        exp++;
    }
    capacity = ((size_t)1) << exp;
    mask = capacity - 1;
    entries = static_cast<std::atomic<EntryHeader*>*>(malloc(capacity * sizeof(std::atomic<EntryHeader*>)));
    memset((void *)entries,0,capacity * sizeof(std::atomic<EntryHeader*>));

    total_mem_size_ = capacity * sizeof(std::atomic<EntryHeader*>);
    return capacity * loadFactor;
}

// using memory pool
size_t inline Hashmap::setSizeUseMemPool(size_t nrEntries) {
    if (entries && !entries_use_mem_pool){
        free(entries);
    }
    entries_use_mem_pool = true;

    const auto loadFactor = 0.7;
    size_t exp = 64 - __builtin_clzll(nrEntries);
    if (((size_t)1 << exp) < nrEntries / loadFactor){
        exp++;
    }
    capacity = ((size_t)1) << exp;
    mask = capacity - 1;
    entries = static_cast<std::atomic<EntryHeader*>*>(local_allocator.allocate(capacity * sizeof(std::atomic<EntryHeader*>)));
    memset((void *)entries,0,capacity * sizeof(std::atomic<EntryHeader*>));

    total_mem_size_ = capacity * sizeof(std::atomic<EntryHeader*>);
    return capacity * loadFactor;
}

void inline Hashmap::clear() {
    for (size_t i = 0; i < capacity; i++) {
        entries[i].store(end(), std::memory_order_relaxed);
    }
}

#ifdef SIMD_SIZE
static inline v8u32 rotl32(v8u32 x, int r) {
    // Using vector operations for shift operations, r must be a constant or scalar
    return (x << r) | (x >> (32 - r));
}

// vector version of the fmix32 function is used for the final mixed hash value
static inline v8u32 fmix32(v8u32 h) {
    h ^= (h >> 16);
    h *= 0x85ebca6b;  // A vector initialized with constants, where all elements are 0x85ebca6b
    h ^= (h >> 13);
    h *= 0xc2b2ae35;
    h ^= (h >> 16);
    return h;
}

// Vectorized MurMurHash3: Processing 8 keys simultaneously
static inline v8u32 hash_32(v8u32 key, v8u32 seed) {
    key *= 0xcc9e2d51;
    key = rotl32(key, 15);
    key *= 0x1b873593;

    seed ^= key;
    seed = rotl32(seed, 13);
    seed = seed * 5 + 0xe6546b64;

    return fmix32(seed);
}

void inline compute_hashes(const uint32_t *keys, uint32_t *hashes) {
    // Unified seed setting, all components have the same value
    constexpr v8u32 seed = {INIT_MACRO(4000932304U)};
    // using reinterpret_cast instead of memcpy to make room for compiler optimization
    *reinterpret_cast<v8u32*>(hashes) = hash_32(*reinterpret_cast<const v8u32*>(keys), seed);
}

#endif

//non-vectorized MurMurHash3 hash algorithm
inline uint32_t rotl32(uint32_t x, int8_t r ) {
    return (x << r) | (x >> (32 - r));
}

inline uint32_t fmix32(uint32_t h ) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uint32_t hash_32(uint32_t key, uint32_t seed=4000932304){
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    uint32_t k1 = key;

    k1 *= c1;
    k1 = rotl32(k1,15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1,13);
    h1 = h1*5+0xe6546b64;

    uint32_t hash = fmix32(h1);
    return hash;
}

}