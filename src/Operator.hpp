/**
* @file Operator.hpp
* @brief Defines the physical query operators for the vectorized execution engine.
*
* This file contains the implementation of the core building blocks of the physical
* query plan. It follows the vectorized Volcano model, where each operator implements
* a `next()` method that produces a batch of results (`OperatorResultTable`). The
* plan is executed by pulling data from the root operator (the sink), which in turn
* pulls data from its children.
*
* The key operators defined here are:
*
* - **`Operator` (Abstract Base Class):** Defines the common interface for all
*   physical operators.
*
* - **`Scan`:** A leaf operator that reads data from a base table. It is designed
*   for parallel execution, where multiple threads can atomically claim and process
*   chunks of the table.
*
* - **`Hashjoin`:** A parallel hash join operator implementing the classic two-phase
*   (build and probe) algorithm. It uses barriers for synchronization between threads
*   during the build phase and features a resumable probe stage to fit the iterator model.
*
* - **`Naivejoin`:** A specialized and highly optimized nested-loop join for the specific
*   case where the build side has exactly one row. This avoids the overhead of building
*   a hash table for a trivial join.
*
* - **`ResultWriter`:** The sink operator at the root of the plan. It pulls all results
*   from its child operator and consolidates them into a final, paged `ColumnarTable`.
*   Multiple threads write their results into a shared, mutex-protected final table.
*
* Each operator uses a nested `Shared` struct (derived from `SharedState`) to manage
* state that must be coordinated across all worker threads executing that operator instance.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sys/types.h>
#include <utility>
#include <variant>
#include <vector>
#include <iostream>
#include <fstream>
#include "Profiler.hpp"
#include "SharedState.hpp"
#include "attribute.h"
#include "statement.h"
#include "plan.h"
#include "HashMap.hpp"
#include "Barrier.hpp"
#include "DataStructure.hpp"
#include "MemoryPool.hpp"

namespace Contest {
// #define DEBUG_LOG

/**
 * @class Operator
 * @brief The abstract base class for all physical query operators.
 *
 * It defines the core `next()` method, which is the heart of the
 * Volcano model where data is pulled from the top of the operator tree.
 */
class Operator {
public:
    Operator() = default;
    Operator(Operator&&) = default;
    Operator(const Operator&) = delete;

    /**
     * @brief Produces the next batch of results from this operator.
     * @return An `OperatorResultTable` containing a vector of rows. An empty table
     *         (with `num_rows_ == 0`) signifies that the operator is exhausted and
     *         has no more data to produce.
     */
    virtual OperatorResultTable next() = 0;

    /// @brief Returns the number of columns in the operator's output schema.
    virtual size_t resultColumnNum() = 0;
    virtual ~Operator() = default;
};

/**
 * @class Scan
 * @brief A physical operator that scans a base table in parallel.
 *
 * Worker threads collaboratively scan a table by atomically claiming "chunks" of work.
 * This two-level division of work (atomic chunks, which contain multiple processing
 * batches/vectors) is a key performance optimization. It balances the overhead of
 * atomic operations (which are relatively expensive) against the amount of work done
 * per atomic operation, ensuring good scalability.
 */
class Scan : public Operator {
public:
    /**
     * @struct Shared
     * @brief The state shared across all threads executing a single logical Scan operation.
     */
    struct Shared : public SharedState {
        /// @brief An atomic counter that tracks the next available chunk of rows to be scanned.
        /// Each thread atomically increments this to claim a new unit of work.
        std::atomic<size_t> pos_{0};
    };

private:
    /// @brief A reference to the state shared by all threads for this scan.
    Shared& shared_;

    size_t total_rows_;      /// The total number of rows in the table being scanned.
    size_t last_row_{0};     /// The absolute starting row index of the *last* batch returned by `next()`.
    size_t vec_size_;        /// The number of rows in a single processing batch (a "vector").

    size_t chunk_size_;      /// The number of batches contained within a single atomic work unit (a "chunk").
    size_t vec_in_chunk_;    /// A per-thread counter for the number of batches processed within its current chunk.

    /// @brief The result table returned by the previous `next()` call. It is reused
    /// across calls to avoid reallocating its `ContinuousColumn` structures.
    OperatorResultTable last_result_;

public:
    /**
     * @brief Constructs a Scan operator instance.
     * @param shared The shared state for this scan operation.
     * @param total_rows The total number of rows in the source table.
     * @param vec_size The desired size for each output batch.
     * @param columns A vector of pointers to the source columns that this scan will read.
     */
    Scan(Shared& shared, size_t total_rows, size_t vec_size, std::vector<const Column*>& columns)
    : shared_(shared), total_rows_(total_rows), vec_size_(vec_size)
    {
        // Heuristic to determine the chunk size. The goal is to make each chunk
        // large enough (e.g., ~10k rows) to amortize the cost of the atomic `fetch_add`.
        size_t scan_morsel_size = 1024 * 10;
        if (vec_size_ >= scan_morsel_size) {
            // If the batch size is already large, one batch is one chunk.
            chunk_size_ = 1;
        } else {
            // Otherwise, calculate how many batches are needed to form a reasonably sized chunk.
            chunk_size_ = scan_morsel_size / vec_size_ + 1;
        }
        // Initialize the counter to be "full" to force claiming a new chunk on the first `next()` call.
        vec_in_chunk_ = chunk_size_;

        // Pre-construct the result table with non-materialized `ContinuousColumn`s.
        // This is efficient as no data is copied here.
        for (auto column : columns) {
            last_result_.addContinuousColumn(column, 0, 0);
        }
    }

    size_t resultColumnNum() override {
        return last_result_.columns_.size();
    }

    OperatorResultTable next() override {
        ProfileGuard profile_guard(global_profiler, "Scan_" + std::to_string(shared_.get_operator_id()));
        size_t skip_rows; // The number of rows to advance the column cursors by.

        // Check if the current thread has finished processing its assigned chunk.
        if (vec_in_chunk_ == chunk_size_) {
            // Yes, so atomically claim a new chunk from the shared state.
            size_t current_chunk = shared_.pos_.fetch_add(1);
            // Calculate how many rows to skip to get from the end of the last batch to the start of this new chunk.
            skip_rows = current_chunk * chunk_size_ * vec_size_ - last_row_;
            vec_in_chunk_ = 0; // Reset the per-thread chunk counter.
        } else {
            // No, we are still within the current chunk. Just advance to the next batch.
            skip_rows = vec_size_;
        }

        // Update the absolute row position.
        last_row_ += skip_rows;

        // Check if we have scanned past the end of the table.
        if (last_row_ >= total_rows_) {
            last_result_.num_rows_ = 0;
            return last_result_; // Signal that the operator is exhausted.
        } else {
            // We have a valid batch to return.
            // Determine the number of rows in this batch (it might be smaller than vec_size_ if it's the last one).
            last_result_.num_rows_ = std::min(total_rows_ - last_row_, vec_size_);
            // Efficiently advance the underlying `ContinuousColumn` cursors.
            last_result_.skipRows(skip_rows);
            vec_in_chunk_++;

            profile_guard.add_input_row_count(last_result_.num_rows_);
            profile_guard.add_output_row_count(last_result_.num_rows_);

            return last_result_;
        }
    }
};


/**
 * @class Naivejoin
 * @brief A specialized nested-loop join optimized for a single-row build side.
 *
 * This operator is an  implementation of a nested-loop join for the very specific but common
 * pattern where the build side is guaranteed to have exactly one row (e.g., after a
 * unique index lookup). In this scenario, building a hash table would be unnecessary
 * overhead. This operator simply scans the probe side and compares each key against
 * the single key from the build side.
 */
class Naivejoin : public Operator {
public:
    /**
     * @struct Shared
     * @brief Shared state for Naivejoin. This is empty, as the operator is
     * self-contained per thread and requires no cross-thread coordination.
     * It exists for structural consistency with other operators.
     */
    struct Shared : public SharedState {};

private:
    Shared& shared_;
    uint32_t vec_size_; /// The processing batch size.

    // --- Build Side (Left) ---
    /// @brief The columns of the single-row build side.
    std::vector<const Column*> columns_;
    /// @brief The single join key value from the build side, cached in the constructor.
    uint32_t left_key_value_;
    /// @brief The index of the join key column on the build side.
    size_t left_idx_;

    // --- Probe Side (Right) ---
    /// @brief The child operator providing the multi-row probe side.
    Operator* right_;
    /// @brief The index of the join key column on the probe side.
    size_t right_idx_;
    /// @brief The number of rows in the current probe batch.
    size_t num_probe_ = 0;

    // --- Buffers for Processing ---
    /// @brief Stores the indices of matching rows from the probe side.
    uint32_t* probe_matches_;
    /// @brief A buffer to hold keys from the probe side if they are not already materialized.
    uint32_t* probe_keys_;
    /// @brief A pointer to the source of probe keys (either `probe_keys_` or an existing materialized column).
    uint32_t* key_source_ = nullptr;

    // --- Result Handling ---
    /// @brief The result table from the last call to `right_->next()`.
    OperatorResultTable right_result_;
    /// @brief The result table to be returned by this operator's `next()`. It is reused across calls.
    OperatorResultTable last_result_;
    /// @brief A mapping of output columns to their source columns.
    LocalVector<size_t> output_attrs_;

#ifdef DEBUG_LOG
    size_t probe_rows_{0};
    size_t output_rows_{0};
    std::string table_str_;
#endif

public:
    /**
     * @brief Constructs a Naivejoin operator.
     *
     * The constructor performs several key setup steps:
     * 1. Caches the single key value from the build side.
     * 2. Pre-allocates buffers for the output result table (`last_result_`).
     * 3. An important optimization: it pre-populates all output columns that come
     *    from the build side, as these values are constant for every matched row.
     */
    Naivejoin(Shared& shared, size_t vec_size, Operator* right, size_t right_idx,
        std::vector<const Column*> columns, size_t left_idx,
        const std::vector<std::tuple<size_t, DataType>>& output_attrs)
    : shared_(shared), vec_size_(vec_size), right_(right), right_idx_(right_idx),
    columns_(std::move(columns)), left_idx_(left_idx) {

        probe_matches_ = (uint32_t*)local_allocator.allocate(vec_size_ * sizeof(uint32_t));
        probe_keys_ = (uint32_t*)local_allocator.allocate(vec_size_ * sizeof(uint32_t));

        // Read and cache the single key value from the build side.
        ContinuousColumn(columns_[left_idx_], 0, 0).gather(1, reinterpret_cast<uint8_t*>(&left_key_value_), 0);

        // Pre-allocate and pre-populate the output table (`last_result_`).
        for (auto [col_idx, col_type] : output_attrs) {
            output_attrs_.push_back(col_idx);
            void* col_buffer = local_allocator.allocate(vec_size * (col_type == DataType::INT32 ? sizeof(int32_t) : sizeof(uint64_t)));
            last_result_.addInstantiatedColumn(col_type, col_buffer);

            // Optimization: If the output column is from the build side, its value is constant.
            // We can fill the entire output buffer with this constant value once, here.
            if (col_idx < columns_.size()) {
                if (col_type == DataType::INT32) {
                    int32_t value;
                    ContinuousColumn(columns_[col_idx], 0, 0).gather(1, reinterpret_cast<uint8_t*>(&value), 0);
                    std::fill_n(static_cast<int32_t*>(col_buffer), vec_size, value);
                } else {
                    uint64_t value;
                    ContinuousColumn(columns_[col_idx], 0, 0).gather(1, reinterpret_cast<uint8_t*>(&value), 0);
                    std::fill_n(static_cast<uint64_t*>(col_buffer), vec_size, value);
                }
            } else if (col_idx == right_idx_ + columns_.size() && col_type == DataType::INT32) {
                // Also pre-fill the probe-side key column, since its value will always be the matched key.
                std::fill_n(static_cast<int32_t*>(col_buffer), vec_size, left_key_value_);
            }
        }
    }

    ~Naivejoin() override = default;

    size_t resultColumnNum() override {
        return last_result_.columns_.size();
    }

    OperatorResultTable next() override {
        // Main probe loop: continues until a non-empty result batch is produced or the probe side is exhausted.
        while (true) {
            // Get the next batch of rows from the probe side.
            right_result_ = right_->next();
            num_probe_ = right_result_.num_rows_;
            if (num_probe_ == 0) {
                // The probe side is exhausted. Clean up and signal completion.
                last_result_.num_rows_ = 0;
                return last_result_;
            }

            // Get a pointer to the probe keys. If they are already materialized, use the pointer
            // directly. Otherwise, gather them into a temporary buffer.
            auto key_col = right_result_.columns_[right_idx_];
            auto instantiated_column = dynamic_cast<InstantiatedColumn*>(key_col);
            if (instantiated_column) {
                key_source_ = (uint32_t*)instantiated_column->getData();
            } else {
                key_col->gather(right_result_.num_rows_, (uint8_t*)probe_keys_, sizeof(uint32_t));
                key_source_ = probe_keys_;
            }

            // Find all matching rows within the current probe batch.
            uint32_t n = joinAllNaive();
            if (n == 0) {
                // No matches in this batch, so loop to get the next probe batch.
                continue;
            }

            // We found matches. Materialize the final result.
            size_t build_side_col_count = columns_.size();
            last_result_.num_rows_ = n;
            for (size_t out_idx = 0; out_idx < output_attrs_.size(); out_idx++) {
                auto column = dynamic_cast<InstantiatedColumn*>(last_result_.columns_[out_idx]);
                size_t in_idx = output_attrs_[out_idx];

                // Only gather data for probe-side non-key columns. Build-side columns and the
                // probe-side key column were already pre-filled in the constructor.
                if (in_idx >= build_side_col_count && in_idx != right_idx_ + build_side_col_count) {
                    right_result_.columns_[in_idx - build_side_col_count]->gather(
                        n, (uint8_t*)column->getData(),
                        column->getType() == DataType::INT32 ? 4 : 8, probe_matches_);
                }
            }

            return last_result_;
        }
    }

    /**
     * @brief Scans a batch of probe keys and compares each against the single build key.
     * @return The number of matching rows found in the batch.
     */
    uint32_t joinAllNaive() {
        size_t found = 0;
        for (size_t i = 0; i < num_probe_; i++) {
            uint32_t key = key_source_[i];
            if (left_key_value_ == key) {
                // Record the index of the matching probe row.
                probe_matches_[found++] = i;
            }
        }
        return found;
    }
};


/**
 * @class Hashjoin
 * @brief A physical operator for a parallel hash join.
 *
 * Implements the classic two-phase (build and probe) algorithm, optimized
 * for parallel execution within the iterator model.
 *
 * - **Build Phase:** On the first call to `next()`, all worker threads collaboratively
 *   build a single shared hash table. They first read all data from the build-side
 *   child operator, then use a barrier to synchronize and size the hash map based
 *   on the total number of rows. After a second barrier ensures all insertions are
 *   complete, the operator transitions to the probe phase.
 *
 * - **Probe Phase:** In subsequent calls, each thread independently pulls batches from
 *   the probe-side child, probes the shared hash table, and produces matching output rows.
 *   The probe logic is "resumable" via the `IteratorContinuation` struct, allowing it
 *   to pause and continue processing between `next()` calls without losing its place.
 */
class Hashjoin : public Operator {
public:
    /**
     * @struct Shared
     * @brief State shared across all threads executing a single logical Hashjoin.
     */
    struct Shared : public SharedState {
        /// @brief An atomic counter to sum the total number of build-side rows from all threads.
        std::atomic<size_t> found_{0};
        /// @brief A pointer to the single, shared hash table built by all threads.
        Hashmap* hashmap_{nullptr};

        explicit Shared(Hashmap* ptr = nullptr) : found_{0}, hashmap_{ptr} {}
    };

private:
    /**
     * @struct IteratorContinuation
     * @brief Stores the micro-state of the probe phase between `next()` calls.
     *
     * This allows the operator to be "paused" (e.g., after filling an output buffer)
     * and "resumed" seamlessly without losing its position in the probe stream or
     * even in the middle of a hash chain.
     */
    struct IteratorContinuation {
        /// @brief If a probe was paused mid-chain, this stores the key being probed.
        uint32_t probe_key_{0};
        /// @brief If a probe was paused mid-chain, this points to the next entry to check.
        Hashmap::EntryHeader* last_chain_{nullptr};

        /// @brief The total number of rows in the current probe batch (`right_result_`).
        size_t num_probe_{0};
        /// @brief The index of the next row to process in the current probe batch.
        size_t next_probe_{0};
    } cont_;

    Shared& shared_;
    uint32_t vec_size_;

    // --- Child Operators and Schema Info ---
    Operator* left_;      /// Child operator for the build side.
    size_t left_idx_;
    size_t left_col_num_;
    Operator* right_;     /// Child operator for the probe side.
    size_t right_idx_;

    // --- Build-Side Data Layout ---
    /// @brief The size in bytes of a single packed entry in the hash table's memory.
    size_t ht_entry_size_;
    /// @brief Pre-calculated offsets of non-key build columns within a hash table entry.
    LocalVector<size_t> build_value_offsets_;

    // --- State Flags ---
    /// @brief Flag, becomes true after the build phase is complete.
    bool is_build_{false};
    /// @brief Flag, true if the built hash table should be offered to the global cache.
    bool store_hashmap_{false};

    // --- Buffers for Probe-Side Processing ---
    OperatorResultTable right_result_;
    uint32_t* probe_hashs_;
    uint32_t* probe_keys_;
    /// @brief Stores pointers to matching build-side entries from the hash table.
    Hashmap::EntryHeader** build_matches_;
    /// @brief Stores the indices of the corresponding matching rows from the probe side.
    uint32_t* probe_matches_;

    // --- Result Handling ---
    OperatorResultTable last_result_;
    LocalVector<size_t> output_attrs_;
    /// @brief Tracks memory allocated for hash entries by this thread to be managed or freed later.
    std::vector<std::pair<uint8_t*, size_t>> allocations_;

#ifdef DEBUG_LOG
    size_t probe_rows_{0};
    size_t output_rows_{0};
    std::string table_str_;
#endif

public:
    /**
     * @brief Constructs a Hashjoin operator.
     *
     * This complex constructor sets up all necessary data structures, including:
     * 1.  Calculating the packed layout and size of hash table entries for memory efficiency.
     * 2.  Allocating all temporary buffers used during the probe phase.
     * 3.  Pre-allocating the `InstantiatedColumn`s for the output result table.
     */
    Hashjoin(Shared& shared, size_t vec_size, Operator* left, size_t left_idx,
        Operator* right, size_t right_idx,
        const std::vector<std::tuple<size_t, DataType>>& output_attrs, std::vector<std::tuple<size_t, DataType>> left_attrs,
        bool is_build = false, bool store_hashmap = false)
    : shared_(shared), vec_size_(vec_size), left_(left), left_idx_(left_idx),
    right_(right), right_idx_(right_idx), is_build_(is_build), store_hashmap_(store_hashmap) {

        // --- Calculate the packed layout of hash table entries ---
        // To satisfy alignment requirements and minimize padding, columns are reordered:
        // 4-byte values are packed first, followed by 8-byte values.
        left_col_num_ = left_attrs.size();
        build_value_offsets_.resize(left_col_num_);
        ht_entry_size_ = (sizeof(Hashmap::EntryHeader) + 3) & ~3; // Align header to 4 bytes
        for (uint32_t i = 0; i < left_attrs.size(); i++) {
            // The build key is stored in the EntryHeader itself, not as part of the payload.
            if (std::get<1>(left_attrs[i]) == DataType::INT32 && i != left_idx_) {
                build_value_offsets_[i] = ht_entry_size_;
                ht_entry_size_ += 4;
            }
        }
        ht_entry_size_ = (ht_entry_size_ + 7) & ~7; // Align current size to 8 bytes for 8-byte values
        for (uint32_t i = 0; i < left_attrs.size(); i++) {
            if (std::get<1>(left_attrs[i]) != DataType::INT32) {
                build_value_offsets_[i] = ht_entry_size_;
                ht_entry_size_ += 8;
            }
        }

        // --- Allocate all temporary buffers ---
        probe_hashs_ = (uint32_t*)local_allocator.allocate(vec_size_ * sizeof(uint32_t));
        probe_keys_ = (uint32_t*)local_allocator.allocate(vec_size_ * sizeof(uint32_t));
        build_matches_ = (Hashmap::EntryHeader**)local_allocator.allocate(vec_size_ * sizeof(Hashmap::EntryHeader*));
        probe_matches_ = (uint32_t*)local_allocator.allocate((vec_size_ + 1) * sizeof(uint32_t));

        // --- Pre-allocate columns for the output result table ---
        for (auto [col_idx, col_type] : output_attrs) {
            // Optimization: If the build-side key is in the output, remap it to the
            // probe-side key's index. This simplifies the final data gathering step, as
            // the probe key is readily available and doesn't need to be extracted from the HT entry.
            if (col_idx == left_idx) {
                col_idx = right_idx + left_attrs.size();
            }
            output_attrs_.push_back(col_idx);
            void* col_buffer = local_allocator.allocate(vec_size_ * (col_type == DataType::INT32 ? sizeof(int32_t) : sizeof(uint64_t)));
            last_result_.addInstantiatedColumn(col_type, col_buffer);
        }
    }

    ~Hashjoin() override {}

    size_t resultColumnNum() override {
        return last_result_.columns_.size();
    }

    OperatorResultTable next() override {
        ProfileGuard profile_guard(global_profiler, "HashJoin_" + std::to_string(shared_.get_operator_id()));

        // --- BUILD PHASE ---
        if (!is_build_) {
            // In the first call, every thread enters the build phase.
            size_t found = 0;
            // Loop to pull all data batches from the left (build-side) child operator.
            while (true) {
                OperatorResultTable left_table = left_->next();
                size_t n = left_table.num_rows_;
                if (n == 0) break;
                found += n;

                // For each batch, allocate a contiguous memory block for the hash table entries.
                uint8_t* ht_entries = (store_hashmap_)
                                        ? (uint8_t*)malloc(n * ht_entry_size_)
                                        : (uint8_t*)local_allocator.allocate(n * ht_entry_size_);
                allocations_.emplace_back(ht_entries, n);

                // Populate these entries by gathering the key, calculating its hash,
                // and gathering all other payload columns from the build side.
                auto left_key = left_table.columns_[left_idx_];
                left_key->gather(n, ht_entries + offsetof(Hashmap::EntryHeader, key), ht_entry_size_);
                left_key->calculateHash(n, ht_entries + offsetof(Hashmap::EntryHeader, hash), ht_entry_size_);

                for (int col_idx = 0; col_idx < left_table.columns_.size(); col_idx++) {
                    if (col_idx == left_idx_) continue;
                    left_table.columns_[col_idx]->gather(n, ht_entries + build_value_offsets_[col_idx], ht_entry_size_);
                }
            }

            // --- Synchronization Block ---
            // 1. Atomically add this thread's row count to the shared `found_` counter.
            shared_.found_.fetch_add(found);

            // 2. Hit the first barrier. The last thread to arrive executes the lambda, which
            //    uses the final total row count to size the shared hash map.
            profile_guard.pause();
            current_barrier->wait([&]() {
                auto total_found = shared_.found_.load();
                if (total_found) {
                    if (store_hashmap_) {
                        shared_.hashmap_->setSize(total_found);
                    } else {
                        shared_.hashmap_->setSizeUseMemPool(total_found);
                    }
                }
            });
            profile_guard.resume();

            auto total_found = shared_.found_.load();
            if (total_found == 0) {
                // If the build side is empty, the join result is also empty.
                is_build_ = true;
                last_result_.num_rows_ = 0;
                return last_result_;
            }

            // 3. Each thread inserts its locally prepared entries into the now-sized shared hash map.
            for (auto [ht_entries, n] : allocations_) {
                shared_.hashmap_->insertAll_tagged(
                    reinterpret_cast<Hashmap::EntryHeader*>(ht_entries), n, ht_entry_size_);
            }
            if (store_hashmap_) {
                shared_.hashmap_->addAllocations(allocations_, ht_entry_size_);
            }
            is_build_ = true; // Transition to the probe phase.

            // 4. Hit the second barrier to ensure all insertions are complete before any thread proceeds.
            profile_guard.pause();
            current_barrier->wait();
            profile_guard.resume();
        }

        // --- PROBE PHASE ---
        while (true) {
            // If the previous probe batch is exhausted, pull a new batch from the right child.
            if (cont_.next_probe_ >= cont_.num_probe_) {
                right_result_ = right_->next();
                cont_.next_probe_ = 0;
                cont_.num_probe_ = right_result_.num_rows_;

                if (cont_.num_probe_ == 0) {
                    // The probe side is exhausted; the join is complete.
                    last_result_.num_rows_ = 0;
                    return last_result_;
                }
                // Pre-calculate keys and hashes for all rows in the new probe batch.
                right_result_.columns_[right_idx_]->gather(right_result_.num_rows_, (uint8_t*)probe_keys_, sizeof(uint32_t));
                right_result_.columns_[right_idx_]->calculateHash(right_result_.num_rows_, (uint8_t*)probe_hashs_, sizeof(uint32_t));
            }

            // Find all matches within the current batch; this populates `build_matches_` and `probe_matches_`.
            uint32_t n = joinAll();
            if (n == 0) continue; // No matches in this probe slice, loop to continue probing.

            // We found matches. Materialize the final output rows.
            last_result_.num_rows_ = n;
            for (size_t out_idx = 0; out_idx < output_attrs_.size(); out_idx++) {
                auto column = dynamic_cast<InstantiatedColumn*>(last_result_.columns_[out_idx]);
                size_t in_idx = output_attrs_[out_idx];
                if (in_idx < left_col_num_) {
                    // Gather data from the build side by dereferencing the `build_matches_` pointers.
                    gatherEntry(column, n, build_value_offsets_[in_idx]);
                } else {
                    // Gather data from the probe side by using the `probe_matches_` indices.
                    right_result_.columns_[in_idx - left_col_num_]->gather(n, (uint8_t*)column->getData(),
                        column->getType() == DataType::INT32 ? 4 : 8, probe_matches_);
                }
            }

            return last_result_;
        }
    }

private:
    /**
     * @brief Gathers values from a batch of matched hash table entries into a result column.
     *
     * This helper function is used during the final materialization step. It iterates
     * through the `build_matches_` array (which contains pointers to matching HT entries)
     * and copies the value at a specific `offset` within each entry into the `output_column`.
     *
     * @param output_column The destination `InstantiatedColumn` to write to.
     * @param n The number of matched entries to process.
     * @param offset The byte offset of the desired attribute within the `EntryHeader`'s payload.
     */
    void gatherEntry(InstantiatedColumn* output_column, uint32_t n, size_t offset) {
        if (output_column->getType() == DataType::INT32) {
            int32_t* base = (int32_t*)output_column->getData();
            for (uint32_t i = 0; i < n; i++) {
                // For each match, cast the build-side entry pointer, add the offset,
                // dereference to get the value, and store it in the output buffer.
                base[i] = *(int32_t*)((uint8_t*)build_matches_[i] + offset);
            }
        } else if (output_column->getType() == DataType::VARCHAR) {
            uint64_t* base = (uint64_t*)output_column->getData();
            for (uint32_t i = 0; i < n; i++) {
                base[i] = *(uint64_t*)((uint8_t*)build_matches_[i] + offset);
            }
        }
    }

    /**
     * @brief Probes the hash table for all keys in the current probe batch and finds matches.
     *
     * This is the core of the probe phase. It iterates through the probe keys, looks
     * them up in the shared hash table, and for every match found, it records a pair of
     * (`build_matches_` pointer, `probe_matches_` index).
     *
     * If the output buffer (`build_matches_`) fills up, this function saves its exact position
     * (which probe row it was on, and which entry in a hash chain it was traversing)
     * into the `cont_` struct and returns. The next call to `joinAll` will resume from that exact spot.
     *
     * @return The number of matches found in this call (up to `vec_size_`).
     */
    uint32_t joinAll() {
        size_t found = 0;

        // --- Part 1: Resume from a previously interrupted hash chain traversal ---
        if (cont_.last_chain_ != nullptr) {
            for (Hashmap::EntryHeader* entry = cont_.last_chain_; entry != nullptr; entry = entry->next) {
                // We only need to check the key, as the hash was already matched.
                if (entry->key == cont_.probe_key_) {
                    build_matches_[found] = entry;
                    // The probe index is the one we were stuck on from the last call.
                    probe_matches_[found] = cont_.next_probe_;
                    found++;
                    if (__glibc_unlikely(found == vec_size_)) {
                        // The output buffer is full again. Save our state and return.
                        cont_.last_chain_ = entry->next;
                        // If we just finished the chain, we need to advance the probe cursor for the next call.
                        if (cont_.last_chain_ == nullptr) {
                            cont_.next_probe_++;
                        }
                        return vec_size_;
                    }
                }
            }
            // Finished the interrupted chain, so advance to the next probe row.
            cont_.next_probe_++;
        }

        // --- Part 2: Process the rest of the probe batch ---
        for (size_t i = cont_.next_probe_, end = cont_.num_probe_; i < end; i++) {
            uint32_t hash = probe_hashs_[i];
            uint32_t key = probe_keys_[i];

            // For each probe key, traverse the corresponding hash chain.
            for (auto entry = shared_.hashmap_->find_chain_tagged(hash); entry != nullptr; entry = entry->next) {
                if (entry->key == key) {
                    // Match found! Record the build-side entry pointer and the probe-side index.
                    build_matches_[found] = entry;
                    probe_matches_[found] = i;
                    found++;
                    if (found == vec_size_) {
                        // The output buffer is full. Save state to `cont_` and return.
                        if (entry->next != nullptr) {
                            // Case A: We are in the middle of a hash chain.
                            cont_.last_chain_ = entry->next;
                            cont_.probe_key_ = key;
                            cont_.next_probe_ = i;
                        } else {
                            // Case B: We just finished a hash chain, but there are more probe rows left.
                            cont_.last_chain_ = nullptr;
                            cont_.next_probe_ = i + 1;
                        }
                        return vec_size_;
                    }
                }
            }
        }

        // --- Cleanup: The entire probe batch has been processed ---
        cont_.last_chain_ = nullptr;
        cont_.next_probe_ = cont_.num_probe_; // Mark the batch as fully processed.
        return found; // Return the number of matches found (which is less than vec_size_).
    }
};


/**
 * @class ResultWriter
 * @brief The sink operator at the root of the execution plan, responsible for
 *        collecting all results and assembling the final output table.
 *
 * This operator continuously pulls result batches from its child operator until the
 * pipeline is exhausted. For each batch, it takes the materialized data and writes
 * it into a paged format using `TempPage` builders. Once all data is processed,
 * each thread appends its locally generated pages to a single, shared `ColumnarTable`
 * under the protection of a mutex.
 */
class ResultWriter {
public:
    /**
     * @struct Shared
     * @brief The state shared across all threads for the final result collection.
     */
    struct Shared : public SharedState {
        /// @brief The final output table where all threads deposit their results.
        ColumnarTable output_;
        /// @brief A mutex to protect concurrent appends to the `output_` table's page lists.
        std::mutex m_;

        /**
         * @brief Constructs the shared state, pre-creating the column structure
         *        of the final output table based on the expected output schema.
         */
        explicit Shared(const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
            for (auto [_, col_type] : output_attrs) {
                output_.columns.emplace_back(col_type);
            }
        }
    };

    Shared& shared_;
    Operator* child_; /// The child operator pipeline to pull results from (typically a join).

    /// @brief Per-thread buffers to hold fully constructed `Page`s for each column.
    LocalVector<LocalVector<Page*>> page_buffers_;
    /// @brief Per-thread temporary page builders, one for each column, to construct the current, partially-filled page.
    LocalVector<TempPage*> unfilled_page_;

    /**
     * @brief Constructs a ResultWriter instance.
     * @param shared The shared state object for result collection.
     * @param child The root of the upstream operator pipeline.
     */
    ResultWriter(Shared& shared, Operator* child)
    : shared_{shared}, page_buffers_(shared.output_.columns.size()),
    child_(child), unfilled_page_() {
        // Create a temporary page builder for each output column, typed correctly.
        for (auto& column : shared.output_.columns) {
            if (column.type == DataType::INT32) {
                unfilled_page_.push_back(local_allocator.make<TempIntPage>());
            } else if (column.type == DataType::VARCHAR) {
                unfilled_page_.push_back(local_allocator.make<TempStringPage>());
            } else {
                throw std::runtime_error("Unsupported data type in ResultWriter");
            }
        }
    }

    /**
     * @brief Executes the entire child pipeline until it is exhausted and finalizes the result.
     *
     * This method is the entry point for a worker thread to run its assigned query plan.
     * It's not a standard iterator `next()` call; it's a "run-to-completion" method.
     */
    void next() {
        size_t found = 0;
        // Main loop: pull result batches from the child operator until it's exhausted.
        while (true) {
            OperatorResultTable child_result = child_->next();
            size_t row_num = child_result.num_rows_;
            ProfileGuard profile(global_profiler, "ResultWriter");
            global_profiler->add_input_row_count("ResultWriter", row_num);
            global_profiler->add_output_row_count("ResultWriter", row_num);

            if (row_num == 0) break; // Child pipeline is exhausted.

            // Process the batch row-by-row, writing each value into the appropriate TempPage builder.
            for (size_t i = 0; i < child_result.columns_.size(); i++) {
                auto from_column = dynamic_cast<InstantiatedColumn*>(child_result.columns_[i]);
                auto rows = child_result.num_rows_;
                LocalVector<Page*>& pages = page_buffers_[i];

                if (from_column->getType() == DataType::INT32) {
                    auto* temp = (TempIntPage*)unfilled_page_[i];
                    auto* from_data = (int32_t*)from_column->getData();

                    for (size_t j = 0; j < rows; ++j) {
                        // If the temporary page is full, dump it and start a new one.
                        if (temp->is_full()) {
                            pages.emplace_back(temp->dump_page());
                        }
                        temp->add_row(from_data[j]);
                    }
                } else if (from_column->getType() == DataType::VARCHAR) {
                    auto* temp = (TempStringPage*)unfilled_page_[i];
                    auto* from_data = (varchar_ptr*)from_column->getData();

                    for (size_t j = 0; j < rows; ++j) {
                        auto value = from_data[j];
                        if (value.isLongString()) {
                            // Long strings are special; they are already in a paged format.
                            // First, dump any partially-filled page of short strings.
                            if (!temp->is_empty()) {
                                pages.emplace_back(temp->dump_page());
                            }
                            // Then, copy the pages of the long string directly.
                            uint16_t page_num = value.longStringPageNum();
                            Page* const* page_start = value.longStringPage();
                            for (size_t k = 0; k < page_num; k++) {
                                Page* copy_page = new Page;
                                memcpy(copy_page->data, page_start[k]->data, PAGE_SIZE);
                                pages.emplace_back(copy_page);
                            }
                        } else {
                            // For short strings, check if it fits and add it.
                            if (!temp->can_store_string(value)) {
                                pages.emplace_back(temp->dump_page());
                            }
                            temp->add_row(value);
                        }
                    }
                } else {
                    throw std::runtime_error("Unsupported data type in ResultWriter");
                }
            }
            found += row_num;
        }

        // After the loop, dump any remaining partially-filled pages.
        for (size_t i = 0; i < shared_.output_.columns.size(); ++i) {
            if (!unfilled_page_[i]->is_empty()) {
                page_buffers_[i].emplace_back(unfilled_page_[i]->dump_page());
            }
        }

        // --- Finalization Step: Append local pages to the shared result table ---
        // This is the only critical section where threads interact.
        {
            std::lock_guard lock(shared_.m_);
            shared_.output_.num_rows += found;
            for (size_t i = 0; i < shared_.output_.columns.size(); i++) {
                auto& column = shared_.output_.columns[i];
                column.pages.insert(column.pages.end(),
                    page_buffers_[i].begin(), page_buffers_[i].end());
            }
        }

        // Clear local buffers for potential reuse (though not strictly necessary here).
        for (auto& pages : page_buffers_) {
            pages.clear();
        }
    }
};

}