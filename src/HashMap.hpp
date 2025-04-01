#pragma once
#include "atomic"
#include "stdlib.h"
#include "assert.h"
#include "hardware.h"
#include "MemoryPool.hpp"

//#define PREFETCH

namespace Contest {
#define NULL_HASH (1642857263)  // 这是NULL_INT32算出的哈希值（祈祷它不会发生碰撞）
#if defined(SPC__SUPPORTS_AVX2) || defined(SPC__SUPPORTS_AVX) || \
    defined(SPC__SUPPORTS_NEON) || defined(SPC__SUPPORTS_VSX) || \
    defined(SPC__SUPPORTS_VMX) || defined(SPC__SUPPORTS_SVE) || defined(SPC__SUPPORTS_SVE2)
#define SIMD_SIZE 8
#define INIT_MACRO(X) X,X,X,X,X,X,X,X
typedef uint32_t v8u32 __attribute__((__vector_size__(sizeof(uint32_t) * SIMD_SIZE)));
typedef uint64_t v8u64 __attribute__((__vector_size__(sizeof(uintptr_t) * SIMD_SIZE)));
struct v8u64M {
    v8u64 vec;
    uint8_t mask;
};
#endif

class Hashmap {
public:
    using hash_t = uint32_t;

    size_t capacity = 0;
    struct EntryHeader
    /// Header for hashtable entries
    {
        EntryHeader* next;
        hash_t hash;
        uint32_t key;
        EntryHeader(EntryHeader* n, hash_t hash_, uint32_t key_) : next(n), hash(hash_), key(key_) { }
        inline hash_t getHash() {return hash;}
        // payload data follows this header
    };

    /// Returns the first entry of the chain for the given hash
    inline EntryHeader* find_chain(hash_t hash);
    /// Returns the first entry of the chain for the given hash
    /// Uses pointer tagging as a filter to quickly determine whether hash is
    /// contained
    inline EntryHeader* find_chain_tagged(hash_t hash);
    //inline v8u64M find_chain_tagged(v8u64 hashes);
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
    inline void insertAll(EntryHeader* first, size_t n, size_t step);
    template <bool concurrentInsert = true>
    inline void insertAll_tagged(EntryHeader* first, size_t n, size_t step);
    /// Set size (no resize functionality)
    inline size_t setSize(size_t nrEntries);
    inline size_t setSizeUseMemPool(size_t nrEntries);
    /// Removes all elements from the hashtable
    inline void clear();

    template <size_t rw, size_t locality>
    inline void prefetchBucket(hash_t hash) {
        // 从 entries 数组中加载实际桶指针（使用 relaxed 内存序即可）
        EntryHeader* bucket_ptr = entries[hash & mask].load(std::memory_order_relaxed);
        __builtin_prefetch(bucket_ptr, rw, locality);
    }


#ifdef SIMD_SIZE
    static inline v8u64 tag(v8u32 p);
    inline v8u64 find_chain_tagged(v8u32 hashes);
#endif

    std::atomic<EntryHeader*>* entries = nullptr;
    bool entries_use_mem_pool = false;

    hash_t mask;      // mask 用于将任意哈希值映射到桶数组的索引范围内，即0~capacity-1
    using ptr_t = uint64_t;
    const ptr_t maskPointer = (~(ptr_t)0) >> (16);     // maskPointer 被用于从一个带 tag 的指针中提取纯粹的地址（低48位）。
    const ptr_t maskTag = (~(ptr_t)0) << (sizeof(ptr_t) * 8 - 16);    // maskTag 用于提取指针的 tag 部分（高16位）。

    inline static EntryHeader* end();
    Hashmap() = default;
    Hashmap(const Hashmap&) = delete;
    inline ~Hashmap();


    void addAllocations(const std::vector<std::pair<uint8_t*, size_t>>& alloc, size_t entry_size){
        size_t alloc_size = 0;
        for(auto [_,n]:alloc){
            alloc_size += n*entry_size;
        }
        std::lock_guard lock(m_);
        allocations_.insert(allocations_.end(),alloc.begin(),alloc.end());
        total_mem_size_ += alloc_size;
    }

    size_t getMemSize() const{
        return total_mem_size_;
    }

private:
    inline Hashmap::EntryHeader* ptr(Hashmap::EntryHeader* p);
    inline ptr_t tag(hash_t p);
    //inline v8u64 tag(v8u64 p);
    inline Hashmap::EntryHeader* update(Hashmap::EntryHeader* old,
        Hashmap::EntryHeader* p, hash_t hash);

    std::mutex m_;
    size_t total_mem_size_{0};
    std::vector<std::pair<uint8_t*, size_t>> allocations_;
};

extern Hashmap::EntryHeader notFound;

inline Hashmap::EntryHeader* Hashmap::end() { return nullptr; }

inline Hashmap::~Hashmap() {
    //if (entries) mem::free_huge(entries, capacity * sizeof(std::atomic<EntryHeader*>));
    if (entries && !entries_use_mem_pool) free(entries);
    for(auto [p,n]:allocations_){
        if(p) free(p);
    }
}

inline Hashmap::ptr_t Hashmap::tag(Hashmap::hash_t hash) {
    auto tagPos = hash >> (sizeof(hash_t) * 8 - 4);    // 提取 hash 的最高 4 位
    return ((size_t)1) << (tagPos + (sizeof(ptr_t) * 8 - 16));  // 实际就是1 << (48 + tagPos)
}

//inline v8u64 Hashmap::tag(v8u64 hashes) {
//    auto tagPos = hashes >> (sizeof(hash_t) * 8 - 4);
//    return v8u64(1) << (tagPos + v8u64(sizeof(ptr_t) * 8 - 16));
//}

inline Hashmap::EntryHeader* Hashmap::ptr(Hashmap::EntryHeader* p) {
    return (EntryHeader*)((ptr_t)p & maskPointer);
}

inline Hashmap::EntryHeader* Hashmap::update(Hashmap::EntryHeader* old,
    Hashmap::EntryHeader* p,
    Hashmap::hash_t hash) {
    return reinterpret_cast<EntryHeader*>((size_t)p | ((size_t)old & maskTag) |
                                          tag(hash));
}

inline Hashmap::EntryHeader* Hashmap::find_chain(hash_t hash) {
    if(hash == NULL_HASH) return nullptr;   // 如果是NULL值对应的hash值，返回nullptr。

    auto pos = hash & mask;
    return entries[pos].load(std::memory_order_relaxed);
}

template <bool concurrentInsert>
void inline Hashmap::insert(EntryHeader* entry, hash_t hash) {
    if(hash == NULL_HASH) return;   // 如果是NULL值对应的hash值，不插入直接返回。

    const size_t pos = hash & mask;
    assert(pos <= mask);
    assert(pos < capacity);
    if (concurrentInsert) {

        auto locPtr = &entries[pos];
        EntryHeader* loc = locPtr->load();
        EntryHeader* newLoc;
        // 使用比较并交换原子操作尝试更新桶的值。
        // 如果当前桶的值和 loc 相等，则将其更新为 newLoc。
        // 如果期间有其他线程对该桶进行了修改，更新会失败，loc 会被更新为当前的实际值，循环会重复，直到更新成功。
        do {
            entry->next = loc;
            newLoc = entry;
        } while (!locPtr->compare_exchange_weak(loc, newLoc));
    } else {
        auto& loc = entries[pos];
        auto oldValue = loc.load(std::memory_order_relaxed);  // 使用std::memory_order_relaxed，不保证原子性和顺序性。
        entry->next = oldValue;
        loc.store(entry, std::memory_order_relaxed);
    }
}

inline Hashmap::EntryHeader* Hashmap::find_chain_tagged(hash_t hash) {
    if(__glibc_unlikely(hash == NULL_HASH)) return nullptr;   // 如果是NULL值对应的hash值，返回nullptr。

    //static_assert(sizeof(hash_t) == 8, "Hashtype not supported");
    auto pos = hash & mask;
    auto candidate = entries[pos].load(std::memory_order_relaxed);
    auto filterMatch = (size_t)candidate & tag(hash);  // 通过tag快速判断hash是否在该链桶当中。
    if (filterMatch)
      return ptr(candidate);
    else
      return end();
}

//inline v8u64M Hashmap::find_chain_tagged(v8u64 hashes) {
//    auto pos = hashes & v8u64(mask);
//    v8u64 candidates = _mm512_i64gather_epi64(pos, (const long long int*)entries, 8);
//    v8u64 filterMatch = candidates & tag(hashes);
//    __mmask8 matches = filterMatch != v8u64(uint64_t(0));
//    candidates = candidates & v8u64(maskPointer);
//    return {candidates, matches};
//}

template <bool concurrentInsert>
void inline Hashmap::insert_tagged(EntryHeader* entry, hash_t hash) {
    if(hash == NULL_HASH) return;   // 如果是NULL值对应的hash值，不插入直接返回。

    const size_t pos = hash & mask;
    assert(pos <= mask);
    assert(pos < capacity);
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
void inline Hashmap::insertAll(EntryHeader* first, size_t n, size_t step) {
    EntryHeader* e = first;
    for (size_t i = 0; i < n; ++i) {
        insert<concurrentInsert>(e, static_cast<hash_t>(e->hash));
        e = reinterpret_cast<EntryHeader*>(reinterpret_cast<uint8_t*>(e) + step);
    }
}

template <bool concurrentInsert>
void inline Hashmap::insertAll_tagged(EntryHeader* first, size_t n,
    size_t step) {
    EntryHeader* e = first;
#ifdef PREFETCH
    uint8_t* prefetch_hash = reinterpret_cast<uint8_t*>(first) + offsetof(Hashmap::EntryHeader, hash) + 10 * step;
#endif
    for (size_t i = 0; i < n; ++i) {
#ifdef PREFETCH
        // 如果还有足够的后续元素，预取较远一条记录对应桶的数据
        if (i + 10 < n) {
            prefetchBucket<1, 1>(*((hash_t*)prefetch_hash));
            prefetch_hash += step;
        }
#endif
        insert_tagged<concurrentInsert>(e, static_cast<hash_t>(e->hash));
        e = reinterpret_cast<EntryHeader*>(reinterpret_cast<uint8_t*>(e) + step);
    }
}

size_t inline Hashmap::setSize(size_t nrEntries) {
    assert(nrEntries != 0);
    //if (entries) mem::free_huge(entries, capacity * sizeof(std::atomic<EntryHeader*>));
    if (entries && !entries_use_mem_pool) free(entries);
    entries_use_mem_pool = false;

   const auto loadFactor = 0.7;
    size_t exp = 64 - __builtin_clzll(nrEntries);
    assert(exp < sizeof(hash_t) * 8);
    if (((size_t)1 << exp) < nrEntries / loadFactor) exp++;
   capacity = ((size_t)1) << exp;
    mask = capacity - 1;
    //entries = static_cast<std::atomic<EntryHeader*>*>(mem::malloc_huge(capacity * sizeof(std::atomic<EntryHeader*>)));
    entries = static_cast<std::atomic<EntryHeader*>*>(malloc(capacity * sizeof(std::atomic<EntryHeader*>)));
    memset((void *)entries,0,capacity * sizeof(std::atomic<EntryHeader*>));

    total_mem_size_ = capacity * sizeof(std::atomic<EntryHeader*>);
    return capacity * loadFactor;
}

size_t inline Hashmap::setSizeUseMemPool(size_t nrEntries) {
    assert(nrEntries != 0);
    //if (entries) mem::free_huge(entries, capacity * sizeof(std::atomic<EntryHeader*>));
    if (entries && !entries_use_mem_pool) free(entries);
    entries_use_mem_pool = true;

    const auto loadFactor = 0.7;
    size_t exp = 64 - __builtin_clzll(nrEntries);
    assert(exp < sizeof(hash_t) * 8);
    if (((size_t)1 << exp) < nrEntries / loadFactor) exp++;
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
//#ifdef SPC__SUPPORTS_AVX2
//#include <immintrin.h>
//inline __m256i hash_32_simd(__m256i keys, uint32_t seed =  4000932304) {
//    const __m256i c1 = _mm256_set1_epi32(0xcc9e2d51);
//    const __m256i c2 = _mm256_set1_epi32(0x1b873593);
//    const __m256i seed_vec = _mm256_set1_epi32(seed);
//    const __m256i five = _mm256_set1_epi32(5);
//    const __m256i mix_constant = _mm256_set1_epi32(0xe6546b64);
//
//    // 处理流程
//    __m256i k = _mm256_mullo_epi32(keys, c1);
//    k = _mm256_or_si256(_mm256_slli_epi32(k, 15), _mm256_srli_epi32(k, 17));
//    k = _mm256_mullo_epi32(k, c2);
//
//    __m256i hash = _mm256_xor_si256(seed_vec, k);
//    hash = _mm256_or_si256(_mm256_slli_epi32(hash, 13), _mm256_srli_epi32(hash, 19));
//    hash = _mm256_add_epi32(_mm256_mullo_epi32(hash, five), mix_constant);
//
//    // Finalization mix
//    hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 16));
//    hash = _mm256_mullo_epi32(hash, _mm256_set1_epi32(0x85ebca6b));
//    hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 13));
//    hash = _mm256_mullo_epi32(hash, _mm256_set1_epi32(0xc2b2ae35));
//    hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 16));
//
//    return hash;
//}
//#endif

#ifdef SIMD_SIZE
static inline v8u32 rotl32(v8u32 x, int r) {
    // 利用向量运算进行移位操作，r 必须是常数或标量
    return (x << r) | (x >> (32 - r));
}

// 向量版 fmix32 函数，用于最终混合散列值
static inline v8u32 fmix32(v8u32 h) {
    h ^= (h >> 16);
    h *= 0x85ebca6b;  // 使用常量初始化的向量，所有元素均为 0x85ebca6b
    h ^= (h >> 13);
    h *= 0xc2b2ae35;
    h ^= (h >> 16);
    return h;
}

// 向量版 MurMurHash3：同时处理 8 个 key
static inline v8u32 hash_32(v8u32 key, v8u32 seed) {
    key *= 0xcc9e2d51;
    key = rotl32(key, 15);
    key *= 0x1b873593;

    seed ^= key;
    seed = rotl32(seed, 13);
    seed = seed * 5 + 0xe6546b64;

    return fmix32(seed);
}

// 示例函数：对 8 个 key 计算 hash，并将结果存入数组
void inline compute_hashes(const uint32_t *keys, uint32_t *hashes) {
    // 统一设置种子，所有分量均为同一个值（例如 4000932304）
    constexpr v8u32 seed = {INIT_MACRO(4000932304U)};
//    auto* aligned_hashes = reinterpret_cast<v8u32*>(__builtin_assume_aligned(hashes, 32));
//    *aligned_hashes = hash_32(*reinterpret_cast<const v8u32*>(keys), seed);
    *reinterpret_cast<v8u32*>(hashes) = hash_32(*reinterpret_cast<const v8u32*>(keys), seed);
}

inline v8u64 Hashmap::find_chain_tagged(v8u32 hashes) {
    auto pos = hashes & v8u32{INIT_MACRO(mask)};

    v8u64 candidates;
    const auto aligned_entries = reinterpret_cast<const uint64_t*>(__builtin_assume_aligned(entries, 64));
    for (int i = 0; i < 8; i++) { // 模拟 gather：将 entries 按 pos[i] 位置加载
        candidates[i] = __builtin_nontemporal_load(&aligned_entries[pos[i]]);
    }

    v8u64 filterMatch = candidates & tag(hashes);

    v8u64 masks = __builtin_convertvector(filterMatch != v8u64{0}, v8u64);
    v8u64 results = (candidates & v8u64{INIT_MACRO(maskPointer)}) & masks;
    return results;
}

inline v8u64 Hashmap::tag(v8u32 hashes) {
    v8u32 tagPos = hashes >> (sizeof(hash_t)*8 - 4);
    v8u64 tagPos64 = __builtin_convertvector(tagPos, v8u64);
    return v8u64{INIT_MACRO(1)} << (tagPos64 + sizeof(ptr_t) * 8 - 16);
}
#endif

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

// MurMurHash3哈希算法
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
