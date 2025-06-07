/**
* @file DataStructure.hpp
* @brief Defines core data structures and accessors for a column-oriented, paged data layout.
*
* This header file provides the fundamental components for representing and interacting
* with data stored in a columnar format, where each column's data is partitioned into
* fixed-size `Page`s. It includes:
*
* - `varchar_ptr`: A specialized 8-byte pointer-like struct to efficiently represent
*   both short and long variable-length strings (VARCHARs). It encodes the string's
*   length and location (either an inline pointer or a reference to a sequence of pages)
*   within a single 64-bit integer.
*
* - `Bitmap`: A utility class for interpreting nullability bitmaps, providing fast
*   checks and population counts for nullable data within a page.
*
* - `PageReader`: A read-only accessor for a raw `Page`. It interprets the page's
*   byte layout based on the column's data type, exposing metadata like row counts
*   and providing access to the raw data and nullability bitmap.
*
* - `ColumnInterface`, `InstantiatedColumn`, `ContinuousColumn`: A class hierarchy
*   representing columns of data. `ColumnInterface` defines a common API for data
*   access via a `gather` method. `ContinuousColumn` represents a non-materialized
*   column that reads data directly from the original paged storage on demand.
*   `InstantiatedColumn` represents a materialized column where data has been
*   copied into a contiguous memory buffer. This abstraction allows query operators
*   to process data transparently, whether it is stored in its original paged format
*   or has been materialized in memory.
*
* - `OperatorResultTable`: A container for the output of a query operator, holding
*   a collection of `ColumnInterface` pointers.
*
* - `TempPage`, `TempIntPage`, `TempStringPage`: Helper classes for building new
*   data pages. They manage the complex layout of data, offsets, and bitmaps within a page.
*
* Overall, these components form the backbone for a vectorized query execution engine,
* enabling efficient, on-demand processing of large datasets stored in a paged,
* columnar layout.
*/

#pragma once

#include "plan.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <cassert>
#include "MemoryPool.hpp"

namespace Contest {

/**
 * @namespace Constants
 * @brief Defines magic numbers and constants used for data representation.
 */
namespace Constants {
/// @brief Represents a NULL value for an INT32 column. This is `INT32_MIN`.
constexpr int32_t NULL_INT32 = -2147483648;
/// @brief Represents a NULL value for a VARCHAR column, stored in a `varchar_ptr`.
constexpr uint64_t NULL_VARCHAR = 0;
/// @brief A 2-byte magic number at the start of a page, indicating it's the beginning of a long string.
constexpr uint16_t LONG_STRING_START = 0xFFFF;
/// @brief A 2-byte magic number at the start of a page, indicating it's a continuation page for a long string.
constexpr uint16_t LONG_STRING_FOLLOW = 0xFFFE;
}

/**
 * @struct varchar_ptr
 * @brief A compact 8-byte representation for variable-length strings (VARCHAR).
 *
 * This struct uses bit-packing to store information about a string within a single
 * 64-bit integer, supporting two main modes:
 *
 * 1.  **Short String (inline):** The most significant bit is 0.
 *     - Bits 62-48 (15 bits): Store the string length (up to 32767).
 *     - Bits 47-0  (48 bits): Store a raw pointer to the string data.
 *
 * 2.  **Long String (paged):** The most significant bit is 1.
 *     - Bit 63: Flag (1 for long string).
 *     - Bits 62-48 (15 bits): Store the number of `Page`s the string spans.
 *     - Bits 47-0  (48 bits): Store a pointer to an array of `Page*`.
 *
 * A value of 0 (`Constants::NULL_VARCHAR`) represents a NULL string.
 */
struct varchar_ptr {
    uint64_t ptr_ = {0};

    /**
     * @brief Constructs a varchar_ptr for a short (inline) string.
     * @param str Pointer to the character data.
     * @param length The length of the string.
     */
    varchar_ptr(const char* str, uint16_t length) { set(str, length); }

    /**
     * @brief Constructs a varchar_ptr for a long (paged) string.
     * @param page Pointer to the start of an array of Page pointers.
     * @param num_page The number of pages the string occupies.
     */
    varchar_ptr(Page* const* page, uint16_t num_page) { set(page, num_page); }

    /// @brief Default constructor, creates a NULL varchar_ptr.
    inline varchar_ptr() = default;

    /**
     * @brief Sets the internal state to represent a short string.
     * @param str Pointer to the character data.
     * @param length The length of the string.
     */
    inline void set(const char* str, uint16_t length) {
        // Layout: [ 16-bit length | 48-bit pointer ]
        ptr_ = ((uint64_t)length << 48) | (uint64_t)(str);
    }

    /**
     * @brief Sets the internal state to represent a long string.
     * @param page Pointer to an array of Page pointers.
     * @param num_page The number of pages.
     */
    inline void set(Page* const* page, uint16_t num_page) {
        // Layout: [ 1-bit flag (1) | 15-bit page count | 48-bit pointer to Page* array ]
        ptr_ = ((uint64_t)num_page << 48) | (uint64_t)(page) | 0x8000000000000000;
    }

    /// @brief Returns the pointer to the character data for a short string.
    inline const char* string() const { return (const char*)(ptr_ & 0x0000FFFFFFFFFFFF); }

    /// @brief Returns the length of a short string.
    inline uint16_t length() const { return (uint16_t)(ptr_ >> 48); }

    /// @brief Checks if the string is NULL.
    inline bool isNull() const { return ptr_ == Constants::NULL_VARCHAR; }

    /// @brief Checks if the string is a long string (stored across pages).
    inline bool isLongString() const { return ptr_ >> 63; }

    /// @brief Returns the pointer to the array of `Page*` for a long string.
    inline Page* const* longStringPage() const {
        return (Page* const*)(ptr_ & 0x0000FFFFFFFFFFFF);
    }

    /// @brief Returns the number of pages for a long string.
    inline uint16_t longStringPageNum() const { return (uint16_t)(ptr_ >> 48) & 0x7FFF; }
};

/**
 * @class Bitmap
 * @brief Provides read-only access and utility functions for a nullability bitmap.
 *
 * A bitmap is a sequence of bytes where each bit corresponds to a row,
 * indicating whether the row's value is non-null (1) or null (0).
 */
class Bitmap {
protected:
    /// @brief A non-owning pointer to the raw bitmap data.
    const uint8_t* bitmap_;
public:
    /**
     * @brief Constructs a Bitmap wrapper.
     * @param bitmap A pointer to the start of the bitmap data. Can be `nullptr`.
     */
    Bitmap(const uint8_t* bitmap) : bitmap_(bitmap) {}

    /**
     * @brief Accesses a full byte of the bitmap.
     * @param index The byte index.
     * @return The byte at the specified index.
     */
    uint8_t operator[](size_t index) const {
        return bitmap_[index];
    }

    /**
     * @brief Checks if the bit at a given index is set (i.e., the value is not NULL).
     * @param idx The bit index (row index).
     * @return `true` if the bit is 1 (not NULL), `false` otherwise. Returns `false` if bitmap is null.
     */
    bool isNotNull(uint16_t idx) const {
        if (!bitmap_) return false;
        auto byte_idx = idx / 8;
        auto bit_idx  = idx % 8;
        return bitmap_[byte_idx] & (1u << bit_idx);
    }

    /**
     * @brief Counts the number of set bits (non-NULL values) from the beginning up to index `n`.
     * @param n The number of bits to count.
     * @return The total count of set bits in the range [0, n).
     */
    uint16_t getNonNullCount(uint16_t n) const {
        if (n == 0 || !bitmap_) return 0;
        uint16_t count = 0;
        uint16_t full_bytes = n / 8;
        uint8_t remaining_bits = n % 8;
        // Use a fast intrinsic to count set bits in full bytes.
        for (uint16_t i = 0; i < full_bytes; ++i) {
            count += __builtin_popcount(bitmap_[i]);
        }
        // Handle the last, potentially partial, byte.
        if (remaining_bits > 0) {
            uint8_t mask = (1 << remaining_bits) - 1;
            count += __builtin_popcount(bitmap_[full_bytes] & mask);
        }
        return count;
    }

    /**
     * @brief Counts the number of set bits (non-NULL values) in a specified range [start, end).
     * @param start The starting bit index (inclusive).
     * @param end The ending bit index (exclusive).
     * @return The total count of set bits in the given range.
     */
    uint16_t getNonNullCount(uint16_t start, uint16_t end) const {
        if (end <= start || !bitmap_) return 0;
        uint16_t count = 0;

        uint16_t start_byte = start / 8;
        uint16_t end_byte = end / 8;
        uint8_t start_bit = start % 8;
        uint8_t end_bit = end % 8;

        // Part 1: Handle the first byte, which may be partially included in the range.
        // This block is executed if the range does not start on a byte boundary.
        if (start_bit) {
            uint8_t mask = (0xFF << start_bit);

            // If the entire range is within this single byte, we also need to mask the end.
            if (start_byte == end_byte) {
                mask &= (0xFF >> (8 - end_bit));
            }

            count += __builtin_popcount(bitmap_[start_byte] & mask);
            start_byte++; // Advance to the next byte as this one is now processed.
        }

        // Part 2: Process all full bytes that are completely within the range.
        while (start_byte < end_byte) {
            count += __builtin_popcount(bitmap_[start_byte++]);
        }

        // Part 3: Handle the last byte if it's partially included and wasn't handled in Part 1.
        if (start_byte == end_byte && end_bit) {
            uint8_t mask = (0xFF >> (8 - end_bit));
            count += __builtin_popcount(bitmap_[start_byte] & mask);
        }

        return count;
    }
};

/**
 * @class PageReader
 * @brief Provides a read-only interface to interpret the raw data of a `Page`.
 *
 * This class decodes the physical layout of a page based on its data type,
 * providing access to metadata (like row counts) and the actual data payload.
 */
class PageReader {
protected:
    DataType type_;
    const Page* page_;

public:
    PageReader(DataType type, const Page* page) : type_(type), page_(page) {}
    PageReader(const Column* column, size_t page_id) : type_(column->type), page_(column->pages[page_id]) {}
    ~PageReader() = default;

    /// @brief Checks if the page is marked as the start of a long string.
    bool isLongStringStart() const {
        uint16_t header = *reinterpret_cast<const uint16_t*>(page_->data);
        return header == Constants::LONG_STRING_START;
    }

    /// @brief Checks if the page is marked as a continuation page of a long string.
    bool isLongStringFollow() const {
        uint16_t header = *reinterpret_cast<const uint16_t*>(page_->data);
        return header == Constants::LONG_STRING_FOLLOW;
    }

    /**
     * @brief Gets the total number of logical rows stored in this page (including NULLs).
     *
     * The first two bytes of a page typically store the total row count. However, this
     * field is repurposed for long string pages:
     * - `0xFFFF` (`LONG_STRING_START`): The page is the start of a long string and represents 1 logical row.
     * - `0xFFFE` (`LONG_STRING_FOLLOW`): The page is a continuation and represents 0 new logical rows.
     *
     * @return For normal pages, the row count. For a long string start page, returns 1. For a follow page, returns 0.
     */
    uint16_t getRowCount() const {
        uint16_t first_2_bytes = *reinterpret_cast<const uint16_t*>(page_->data);
        if (first_2_bytes == Constants::LONG_STRING_START) {
            return 1;
        } else if (first_2_bytes == Constants::LONG_STRING_FOLLOW) {
            return 0;
        } else {
            return first_2_bytes;
        }
    }

    /**
     * @brief Gets the number of non-NULL rows stored in this page.
     *
     * For a normal page, bytes 2 and 3 store the count of non-null values.
     * For long string pages, this concept is not applicable in the same way.
     *
     * @return For normal pages, the non-null row count. For a long string start page, returns 1. For a follow page, returns 0.
     */
    uint16_t getNonNullRowCount() const {
        uint16_t first_2_bytes = *reinterpret_cast<const uint16_t*>(page_->data);
        if (first_2_bytes == Constants::LONG_STRING_START) {
            return 1; // A long string is treated as a single non-null value.
        } else if (first_2_bytes == Constants::LONG_STRING_FOLLOW) {
            return 0;
        } else {
            return *reinterpret_cast<const uint16_t*>(page_->data + 2);
        }
    }

    /**
     * @brief Gets a `Bitmap` object for accessing the page's nullability information.
     *
     * The bitmap is stored at the end of the page and grows backwards.
     *
     * @return A `Bitmap` pointing to the nullability data.
     *         Returns a bitmap with a `nullptr` if the page is for a long string.
     */
    Bitmap getBitmap() const {
        uint16_t first_2_bytes = *reinterpret_cast<const uint16_t*>(page_->data);
        if (first_2_bytes == Constants::LONG_STRING_START || first_2_bytes == Constants::LONG_STRING_FOLLOW) {
            return {nullptr};
        }
        // The bitmap's address is calculated from the end of the page.
        // Its size depends on the total number of rows (including nulls).
        return {reinterpret_cast<const uint8_t*>(page_->data + PAGE_SIZE - (first_2_bytes + 7) / 8)};
    }

    /**
     * @brief Gets a pointer to the start of the page's primary data area.
     *
     * This pointer points past the page header and any other metadata (like varchar offsets).
     * The layout depends on the data type.
     * @return A const pointer to the start of the data payload.
     */
    const uint8_t* getPageData() const {
        if (type_ == DataType::INT32) {
            // Layout: [2B row count][2B non-null count][...data...]
            return reinterpret_cast<const uint8_t*>(page_->data + 4);
        } else if (type_ == DataType::INT64 || type_ == DataType::FP64) {
            // Layout: [2B...][2B...][4B unused][...data...]
            // Assumes a fixed 8-byte header for these types.
            return reinterpret_cast<const uint8_t*>(page_->data + 8);
        } else { // Assumes VARCHAR
            if (isLongStringStart() || isLongStringFollow()) {
                // Layout: [2B magic][2B length][...string data...]
                return reinterpret_cast<const uint8_t*>(page_->data + 4);
            } else {
                // Layout: [2B count][2B non-null count][...offsets...][...string data...]
                // Skips the header and the array of offsets for non-null strings.
                return reinterpret_cast<const uint8_t*>(page_->data + 4 + getNonNullRowCount() * sizeof(uint16_t));
            }
        }
    }

    /**
     * @brief Gets a pointer to the start of the varchar offset array within a page.
     *
     * This is only valid for VARCHAR pages that are not part of a long string.
     * @return A const pointer to the array of `uint16_t` offsets.
     */
    const uint16_t* getVarcharOffset() const {
        // Layout: [2B count][2B non-null count][...offsets start here...]
        return reinterpret_cast<const uint16_t*>(page_->data + 4);
    }
};


/**
 * @class ColumnInterface
 * @brief An abstract base class that defines the common interface for all column types.
 *
 * This interface provides a unified way for query operators to access data from a
 * logical column, regardless of its underlying storage format (e.g., paged and
 * compressed, or fully materialized in a contiguous array).
 */
class ColumnInterface {
public:
    virtual ~ColumnInterface() = default;

    /// @brief Gets the data type of the column (e.g., INT32, VARCHAR).
    virtual DataType getType() const = 0;

    /**
     * @brief The core data retrieval method for accessing column data.
     *
     * This function gathers values from the column and copies them into a target buffer.
     * It supports both sequential (contiguous) and random (indexed) access patterns.
     *
     * @param n The number of rows to gather.
     * @param target A pointer to the destination buffer where data will be written.
     * @param target_step The stride in bytes for the target buffer. This allows writing
     *        to non-contiguous memory, such as a specific field in an array of structs.
     * @param indices An optional array of row indices to gather.
     *        - If `nullptr`, rows are gathered sequentially from the current logical position.
     *        - If provided, it enables random access. The indices in this array specify
     *          which rows to fetch relative to the current logical position.
     *        - **Important:** The `indices` array, if not null, is expected to be
     *          **non-decreasing** (i.e., `indices[i] <= indices[i+1]`). This constraint
     *          allows for more efficient, forward-only scanning of the underlying data pages.
     */
    virtual void gather(size_t n, uint8_t* target, size_t target_step, const uint32_t* indices = nullptr) const = 0;

    /**
     * @brief Computes hash values for a batch of rows.
     *
     * Primarily used in Hash Join.
     *
     * @param n The number of rows to hash.
     * @param target A pointer to the destination buffer for the computed hash values.
     * @param target_step The stride in bytes for the target hash buffer.
     */
    virtual void calculateHash(size_t n, uint8_t* target, size_t target_step) const = 0;

    /**
     * @brief Converts the value at a specific row index to a string representation.
     *
     * Useful for debugging, logging, or printing results.
     * @param row_index The logical index of the row to convert.
     * @return A std::string representation of the value, or "NULL" for null values.
     */
    virtual std::string valueToString(uint32_t row_index) const = 0;

    /**
     * @brief Advances the logical cursor of the column by a specified number of rows.
     *
     * This is used to efficiently skip over rows that have already been processed
     * by a previous operator in a pipeline.
     * @param skip_rows The number of rows to skip.
     */
    virtual void skipRows(size_t skip_rows) = 0;
};

/**
 * @class InstantiatedColumn
 * @brief A concrete column implementation that holds materialized data in a contiguous memory block.
 *
 * This class represents a column whose data has been "materialized" (i.e., copied from its
 * original paged format) into a simple, flat array. This provides fast, cache-friendly access.
 */
class InstantiatedColumn : public ColumnInterface {
private:
    DataType type_;
    /// @brief A non-owning pointer to the contiguous block of materialized data.
    /// The lifetime of this data is managed externally.
    uint8_t* data_;

public:
    /**
     * @brief Constructs an InstantiatedColumn.
     * @param type The data type of the column.
     * @param data A non-owning pointer to the materialized data buffer.
     */
    InstantiatedColumn(DataType type, uint8_t* data)
    : type_(type), data_(data) {}

    DataType getType() const override { return type_; }

    void gather(size_t n, uint8_t* target, size_t target_step, const uint32_t* indices = nullptr) const override {
        for (size_t i = 0; i < n; ++i) {
            // Determine the source index: either sequential (i) or from the indices array.
            const size_t source_idx = indices ? indices[i] : i;
            if (type_ == DataType::INT32) {
                // Copy a 4-byte integer value.
                *reinterpret_cast<uint32_t*>(target) = *reinterpret_cast<const uint32_t*>(data_ + source_idx * sizeof(uint32_t));
            } else {
                // Copy an 8-byte value (e.g., varchar_ptr, INT64, FP64).
                *reinterpret_cast<uint64_t*>(target) = *reinterpret_cast<const uint64_t*>(data_ + source_idx * sizeof(uint64_t));
            }
            // Advance the target pointer by the specified stride.
            target += target_step;
        }
    }

    void calculateHash(size_t n, uint8_t* target, size_t target_step) const override {
        // This function is optimized for INT32, as it's a common hash key type.
        // The data is assumed to be in a contiguous array.
        assert(type_ == DataType::INT32);
        const int32_t* base = reinterpret_cast<const int32_t*>(data_);

#ifdef SIMD_SIZE
        // Vectorized implementation using SIMD intrinsics for high performance.
        size_t i = 0;
        // Fast path for when the target buffer is also contiguous (gather).
        if (target_step == sizeof(Hashmap::hash_t)) { // Assuming Hashmap::hash_t is uint32_t
            for (; i + SIMD_SIZE - 1 < n; i += SIMD_SIZE) {
                // `compute_hashes` is assumed to be a SIMD-accelerated function.
                compute_hashes(reinterpret_cast<const uint32_t*>(base + i), reinterpret_cast<uint32_t*>(target));
                target += SIMD_SIZE * sizeof(uint32_t);
            }
        } else {
            // Path for when the target buffer has a non-standard stride (scatter).
            for (; i + SIMD_SIZE - 1 < n; i += SIMD_SIZE) {
                alignas(32) uint32_t hash_arr[SIMD_SIZE];
                compute_hashes(reinterpret_cast<const uint32_t*>(base + i), hash_arr);
                // Manually copy hashes to the target buffer according to the stride.
                for (int j = 0; j < SIMD_SIZE; ++j) {
                    *reinterpret_cast<uint32_t*>(target) = hash_arr[j];
                    target += target_step;
                }
            }
        }

        // Scalar processing for any remaining elements that don't fit in a SIMD vector.
        for (; i < n; ++i) {
            uint32_t hash = hash_32(base[i]); // `hash_32` is a scalar hash function.
            *reinterpret_cast<uint32_t*>(target) = hash;
            target += target_step;
        }
#else
        // Fallback scalar implementation if SIMD is not enabled.
        for (size_t i = 0; i < n; ++i) {
            *reinterpret_cast<uint32_t*>(target) = hash_32(base[i]);
            target += target_step;
        }
#endif
    }

    std::string valueToString(uint32_t row_index) const override {
        if (type_ == DataType::INT32) {
            int32_t val = *reinterpret_cast<const int32_t*>(data_ + row_index * sizeof(int32_t));
            if (val == Constants::NULL_INT32) return "NULL";
            return std::to_string(val);
        } else { // Assumes VARCHAR, INT64, or FP64 stored as 8-byte values
            varchar_ptr ptr = *reinterpret_cast<const varchar_ptr*>(data_ + row_index * sizeof(varchar_ptr));
            if (ptr.isNull()) return "NULL";

            // Handle long strings that are stored across multiple pages.
            if (ptr.isLongString()) {
                std::string long_str;
                uint16_t page_num = ptr.longStringPageNum();
                Page* const* page_start = ptr.longStringPage();
                // Reconstruct the long string from its constituent pages.
                for (size_t k = 0; k < page_num; k++) {
                    // Assumes page layout: [2B magic][2B chunk_len][...data...]
                    long_str.append(reinterpret_cast<const char*>(page_start[k]->data + 4), *reinterpret_cast<const uint16_t*>(page_start[k]->data + 2));
                }
                return long_str;
            }
            // Handle short, inline strings.
            return {ptr.string(), ptr.length()};
        }
    }

    void skipRows(size_t skip_rows) override {
        // For an instantiated column, skipping rows is a simple pointer arithmetic operation.
        if (type_ == DataType::INT32) {
            data_ += skip_rows * sizeof(uint32_t);
        } else {
            data_ += skip_rows * sizeof(varchar_ptr);
        }
    }

    /// @brief Gets the raw, non-owning pointer to the underlying data buffer.
    uint8_t* getData() {
        return data_;
    }
};

/**
 * @class ContinuousColumn
 * @brief Represents a non-materialized ("lazy") column that reads data directly from its original paged storage.
 *
 * This class acts as a "view" or a "cursor" over a logical sequence of rows that
 * may span multiple physical pages. It avoids copying data into a contiguous buffer,
 * instead reading values on-demand. This is highly efficient for operators like
 * table scans that process data without needing to modify it. It maintains its
 * current position via a page index and a row offset within that page.
 */
class ContinuousColumn : public ColumnInterface {
private:
    const DataType type_;
    /// @brief A non-owning pointer to the source Column object which contains the page array.
    const Column* source_column_;
    /// @brief The index of the page where the current logical view starts.
    uint32_t start_page_idx_;
    /// @brief The row offset within the starting page.
    uint32_t start_row_in_page_;

public:
    /**
     * @brief Constructs a ContinuousColumn, creating a view over the source data.
     * @param source_column The original column containing the page data.
     * @param start_page The starting page index for this view.
     * @param start_row The starting row offset within the start_page.
     */
    ContinuousColumn(const Column* source_column, uint32_t start_page, uint32_t start_row)
    : type_(source_column->type), source_column_(source_column), start_page_idx_(start_page), start_row_in_page_(start_row) {}

    DataType getType() const override { return type_; }

    void gather(size_t n, uint8_t* target, size_t target_step, const uint32_t* indices = nullptr) const override {
        // Dispatch to the appropriate private implementation based on access pattern and data type.
        if (indices) {
            // Indexed access pattern.
            if (type_ == DataType::INT32) {
                gather_indexed_int32(n, target, target_step, indices);
            } else {
                gather_indexed_varchar(n, target, target_step, indices);
            }
        } else {
            // Sequential access pattern.
            if (type_ == DataType::INT32) {
                gather_continuous_int32(n, target, target_step);
            } else {
                gather_continuous_varchar(n, target, target_step);
            }
        }
    }

    void calculateHash(size_t n, uint8_t* target, size_t target_step) const override {
        if (n == 0) return;
        assert(source_column_->type == DataType::INT32);

        size_t rows_to_gather = n;
        uint32_t current_page_idx = start_page_idx_;
        uint32_t start_row = start_row_in_page_;

        // --- Main loop: Iterate through pages until all 'n' rows are processed ---
        while (rows_to_gather > 0 && current_page_idx < source_column_->pages.size()) {
            // Determine how many rows to process from the current page.
            PageReader reader(source_column_, current_page_idx);
            const uint16_t rows_in_this_page = reader.getRowCount();
            size_t end_row = std::min((size_t)rows_in_this_page, start_row + rows_to_gather);

#ifdef SIMD_SIZE
            // --- SIMD-enabled implementation ---

            // Optimization: Use a much faster path if the page has no null values.
            if (__glibc_likely(reader.getNonNullRowCount() == rows_in_this_page)) {
                const int32_t* page_data = reinterpret_cast<const int32_t*>(reader.getPageData()) + start_row;
                size_t j = start_row;
                while (j < end_row) {
                    // Process data in aligned chunks of 8 (matching the SIMD vector width).
                    if ((j % 8 == 0) && (j + 8 <= end_row)) {
                        v8u32 keys;
                        // Load a full vector of 8 keys directly from memory.
                        memcpy(&keys, page_data, sizeof(keys));
                        page_data += 8;

                        alignas(32) uint32_t hash_arr[SIMD_SIZE];
                        compute_hashes(reinterpret_cast<const uint32_t*>(&keys), hash_arr);

                        // Fast path for contiguous output buffer.
                        if (target_step == sizeof(Hashmap::hash_t)) {
                            memcpy(target, hash_arr, sizeof(hash_arr));
                            target += sizeof(v8u32);
                        } else { // Slower path to "scatter" results to a non-contiguous buffer.
                            for (int k = 0; k < SIMD_SIZE; ++k) {
                                *(uint32_t*)target = hash_arr[k];
                                target += target_step;
                            }
                        }
                        j += 8;
                    } else { // Scalar fallback for remaining rows that don't form a full vector.
                        int32_t key = *page_data++;
                        uint32_t hash = hash_32(key);
                        *(uint32_t*)target = hash;
                        target += target_step;
                        j++;
                    }
                }
            } else { // Slower path for pages that contain NULL values.
                Bitmap bitmap = reader.getBitmap();
                // Data is compacted, so find the start by counting non-nulls before this point.
                const int32_t* page_data = reinterpret_cast<const int32_t*>(reader.getPageData()) + bitmap.getNonNullCount(start_row);
                size_t j = start_row;
                while (j < end_row) {
                    // Process in 8-row chunks, which conveniently aligns with 1 byte of the bitmap.
                    if ((j % 8 == 0) && (j + 8 <= end_row)) {
                        uint8_t bitmap_byte = bitmap[j / 8];
                        v8u32 keys;

                        // Fast sub-path: if all 8 values in this chunk are non-null.
                        if (__builtin_popcount(bitmap_byte) == 8) {
                            memcpy(&keys, page_data, sizeof(keys));
                            page_data += 8;
                        } else {
                            // "Re-inflate" the sparse data into a dense vector for SIMD processing.
                            for (int k = 0; k < 8; k++) {
                                if (bitmap.isNotNull(j + k)) {
                                    keys[k] = *page_data++; // Get value from compacted data.
                                } else {
                                    keys[k] = static_cast<uint32_t>(Constants::NULL_INT32); // Insert NULL placeholder.
                                }
                            }
                        }

                        alignas(32) uint32_t hash_arr[SIMD_SIZE];
                        compute_hashes(reinterpret_cast<const uint32_t*>(&keys), hash_arr);
                        // Write results to target buffer (either contiguous or scattered).
                        if (target_step == sizeof(Hashmap::hash_t)) {
                            memcpy(target, hash_arr, sizeof(hash_arr));
                            target += sizeof(v8u32);
                        } else {
                            for (int k = 0; k < SIMD_SIZE; ++k) {
                                *(uint32_t*)target = hash_arr[k];
                                target += target_step;
                            }
                        }
                        j += 8;
                    } else { // Scalar fallback for remaining rows.
                        int32_t key = Constants::NULL_INT32;
                        if (bitmap.isNotNull(j)) {
                            key = *page_data++;
                        }
                        uint32_t hash = hash_32(key);
                        *(uint32_t*)target = hash;
                        target += target_step;
                        j++;
                    }
                }
            }
#else
            // --- Fallback: Scalar implementation if SIMD is not enabled ---
            Bitmap bitmap = reader.getBitmap();
            const int32_t* page_data = reinterpret_cast<const int32_t*>(reader.getPageData()) + bitmap.getNonNullCount(start_row);
            for (size_t j = start_row; j < end_row; j++) {
                int32_t key = Constants::NULL_INT32;
                if (bitmap.isNotNull(j)) {
                    key = *page_data++;
                }
                uint32_t hash = hash_32(key);
                *(uint32_t*)target = hash;
                target += target_step;
            }
#endif
            // --- Update state for the next iteration ---
            rows_to_gather -= (end_row - start_row);
            current_page_idx++;
            start_row = 0; // Subsequent pages are always read from the beginning.
        }
    }

    std::string valueToString(uint32_t row_index) const override {
        // Use the indexed gather functionality to retrieve a single value.
        uint32_t single_index = row_index;
        if (type_ == DataType::INT32) {
            int32_t value;
            gather(1, reinterpret_cast<uint8_t*>(&value), sizeof(int32_t), &single_index);
            if (value == Constants::NULL_INT32) return "NULL";
            return std::to_string(value);
        } else { // Assumes VARCHAR
            varchar_ptr ptr;
            gather(1, reinterpret_cast<uint8_t*>(&ptr), sizeof(varchar_ptr), &single_index);
            if (ptr.isNull()) return "NULL";

            // If it's a long string, it must be reconstructed from its constituent pages.
            if (ptr.isLongString()) {
                std::string long_str;
                uint16_t page_num = ptr.longStringPageNum();
                Page* const* page_start = ptr.longStringPage();
                for (size_t k = 0; k < page_num; k++) {
                    // Append the data chunk from each page. The layout is assumed to be:
                    // [2B magic][2B chunk length][...data...].
                    long_str.append(reinterpret_cast<const char*>(page_start[k]->data + 4), *reinterpret_cast<const uint16_t*>(page_start[k]->data + 2));
                }
                return long_str;
            }
            // For a short string, simply use its inline pointer and length.
            return {ptr.string(), ptr.length()};
        }
    }

    void skipRows(size_t skip_rows) override {
        // Tentatively advance the in-page offset. This may now point beyond the current page.
        start_row_in_page_ += skip_rows;

        // Iterate forward through the pages to find the correct new start page and offset.
        while (start_page_idx_ < source_column_->pages.size()) {
            PageReader reader(source_column_, start_page_idx_);
            uint16_t page_rows = reader.getRowCount();

            // Check if the new logical start position falls within the current page.
            if (start_row_in_page_ < page_rows) {
                // It does. The correct page and in-page offset have been found.
                break;
            }

            // The new start position is beyond this page.
            // Subtract this page's rows from the offset and move to the next page.
            start_row_in_page_ -= page_rows;
            start_page_idx_++;
        }
    }

private:

    /// @brief Gathers a continuous sequence of INT32 values from paged storage.
    void gather_continuous_int32(size_t n, uint8_t* target, size_t target_step) const {
        if (n == 0) return;

        size_t rows_to_gather = n;
        uint32_t current_page_idx = start_page_idx_;
        uint32_t row_in_page_offset = start_row_in_page_;

        // --- Main loop: Iterate through pages until all 'n' rows are gathered ---
        while (rows_to_gather > 0 && current_page_idx < source_column_->pages.size()) {
            PageReader reader(source_column_, current_page_idx);
            const uint16_t rows_in_this_page = reader.getRowCount();
            const size_t rows_to_read_from_this_page = std::min((size_t)rows_in_this_page - row_in_page_offset, rows_to_gather);

            // Fast path: If the page contains no nulls, we can read data directly.
            if (__glibc_likely(reader.getNonNullRowCount() == rows_in_this_page)) {
                const int32_t* page_data = reinterpret_cast<const int32_t*>(reader.getPageData()) + row_in_page_offset;
                for (size_t i = 0; i < rows_to_read_from_this_page; ++i) {
                    *reinterpret_cast<int32_t*>(target) = page_data[i];
                    target += target_step;
                }
            } else { // Slower path: Page contains nulls, requires bitmap checks.
                Bitmap bitmap = reader.getBitmap();
                size_t non_nulls_before = bitmap.getNonNullCount(row_in_page_offset);
                const int32_t* page_data = reinterpret_cast<const int32_t*>(reader.getPageData()) + non_nulls_before;

                for (size_t i = 0; i < rows_to_read_from_this_page; ++i) {
                    if (bitmap.isNotNull(row_in_page_offset + i)) {
                        *reinterpret_cast<int32_t*>(target) = *page_data++;
                    } else {
                        *reinterpret_cast<int32_t*>(target) = Constants::NULL_INT32;
                    }
                    target += target_step;
                }
            }
            // --- Update state for the next iteration ---
            rows_to_gather -= rows_to_read_from_this_page;
            current_page_idx++;
            row_in_page_offset = 0;
        }
    }

    /// @brief Gathers a continuous sequence of VARCHAR values from paged storage.
    void gather_continuous_varchar(size_t n, uint8_t* target, size_t target_step) const {
        if (n == 0) return;

        size_t rows_to_gather = n;
        uint32_t current_page_idx = start_page_idx_;
        uint32_t row_in_page_offset = start_row_in_page_;

        while (rows_to_gather > 0 && current_page_idx < source_column_->pages.size()) {
            PageReader reader(source_column_, current_page_idx);

            // A continuous scan should not start on a "follow" page.
            assert(!reader.isLongStringFollow());

            // --- Special Case: The current page starts a long string ---
            if (reader.isLongStringStart()) {
                // Find all pages that belong to this single long string.
                uint32_t end_page_idx = current_page_idx + 1;
                while (end_page_idx < source_column_->pages.size() && PageReader(source_column_, end_page_idx).isLongStringFollow()) {
                    end_page_idx++;
                }
                // Create a 'long string' varchar_ptr pointing to the sequence of pages.
                varchar_ptr val(&source_column_->pages[current_page_idx], end_page_idx - current_page_idx);
                *reinterpret_cast<varchar_ptr*>(target) = val;
                target += target_step;

                // We have processed one logical row (the long string). Update state and continue.
                rows_to_gather--;
                current_page_idx = end_page_idx;
                row_in_page_offset = 0;
                continue;
            }

            // --- Normal Case: The current page contains short strings ---
            const uint16_t rows_in_this_page = reader.getRowCount();
            const size_t rows_to_read_from_this_page = std::min((size_t)rows_in_this_page - row_in_page_offset, rows_to_gather);

            Bitmap bitmap = reader.getBitmap();
            size_t non_nulls_before = bitmap.getNonNullCount(row_in_page_offset);

            const char* page_data = reinterpret_cast<const char*>(reader.getPageData());
            const uint16_t* current_offset_ptr = reader.getVarcharOffset() + non_nulls_before;
            // The start offset of the first string is the end offset of the previous one.
            uint16_t last_offset = (non_nulls_before == 0) ? 0 : *(current_offset_ptr - 1);

            // Fast path for pages with no null values.
            if (__glibc_likely(reader.getNonNullRowCount() == rows_in_this_page)) {
                for (size_t i = 0; i < rows_to_read_from_this_page; ++i) {
                    uint16_t current_offset = *current_offset_ptr;
                    varchar_ptr val(page_data + last_offset, current_offset - last_offset);
                    *reinterpret_cast<varchar_ptr*>(target) = val;
                    last_offset = current_offset;
                    current_offset_ptr++;
                    target += target_step;
                }
            } else { // Slower path for pages with nulls.
                for (size_t i = 0; i < rows_to_read_from_this_page; ++i) {
                    if (bitmap.isNotNull(row_in_page_offset + i)) {
                        uint16_t current_offset = *current_offset_ptr;
                        varchar_ptr val(page_data + last_offset, current_offset - last_offset);
                        *reinterpret_cast<varchar_ptr*>(target) = val;
                        last_offset = current_offset;
                        current_offset_ptr++;
                    } else {
                        *reinterpret_cast<uint64_t*>(target) = Constants::NULL_VARCHAR;
                    }
                    target += target_step;
                }
            }

            // --- Update state for the next iteration ---
            rows_to_gather -= rows_to_read_from_this_page;
            current_page_idx++;
            row_in_page_offset = 0;
        }
    }

    /// @brief Gathers INT32 values from paged storage using a non-decreasing array of indices.
    void gather_indexed_int32(size_t n, uint8_t* target, size_t target_step, const uint32_t* indices) const {
        if (n == 0) return;

        // --- State variables for the forward scan ---
        uint32_t current_offset = indices[0] + start_row_in_page_;
        // `prev_offset` tracks the last accessed offset *on the current page* to calculate deltas.
        uint32_t prev_offset = current_offset;
        uint32_t current_page_idx = start_page_idx_;
        // `prev_page_idx` is used to detect when we cross a page boundary.
        uint32_t prev_page_idx = std::numeric_limits<uint32_t>::max();

        PageReader reader(source_column_, current_page_idx);
        uint32_t rows_on_page = reader.getRowCount();

        // --- Page-local cached state ---
        Bitmap bitmap = Bitmap(nullptr);
        bool is_nonnull_page = false;
        // `nonnull_count` tracks the number of non-nulls up to the current position *within a sparse page*.
        size_t nonnull_count = 0;

        // --- Main loop: Iterate through the requested indices ---
        for (size_t i = 0; i < n; ++i) {
            // Step 1: Locate the correct page for the current logical offset.
            while (current_offset >= rows_on_page) {
                // The target row is not on this page; advance to the next.
                current_offset -= rows_on_page;
                current_page_idx++;
                reader = PageReader(source_column_, current_page_idx);
                rows_on_page = reader.getRowCount();
            }

            // Step 2: If we've moved to a new page, reset page-local state.
            if (prev_page_idx != current_page_idx) {
                bitmap = reader.getBitmap();
                // Recalculate the base non-null count up to our new starting offset on this page.
                nonnull_count = bitmap.getNonNullCount(current_offset);
                is_nonnull_page = (rows_on_page == reader.getNonNullRowCount());
                prev_offset = current_offset; // Reset previous offset for the new page.
                prev_page_idx = current_page_idx;
            }

            // Step 3: Retrieve the data from the now-located position.
            if (__glibc_likely(is_nonnull_page)) {
                // Fast path: No nulls, so logical offset equals physical offset.
                const int32_t* data_ptr = reinterpret_cast<const int32_t*>(reader.getPageData()) + current_offset;
                *reinterpret_cast<int32_t*>(target) = *data_ptr;
            } else if (bitmap.isNotNull(current_offset)) {
                // Slow path (with nulls): The value is not null.
                // Advance the non-null counter by the number of set bits since the last access *on this page*.
                nonnull_count += bitmap.getNonNullCount(prev_offset, current_offset);
                const int32_t* data_ptr = reinterpret_cast<const int32_t*>(reader.getPageData()) + nonnull_count;
                *reinterpret_cast<int32_t*>(target) = *data_ptr;
                prev_offset = current_offset; // Update the last-accessed position for the next delta calculation.
            } else {
                // Slow path (with nulls): The value is null.
                *reinterpret_cast<int32_t*>(target) = Constants::NULL_INT32;
            }

            target += target_step;

            // Step 4: Advance the logical offset for the next iteration.
            if (i + 1 < n) {
                current_offset += (indices[i + 1] - indices[i]);
            }
        }
    }

    /// @brief Gathers VARCHAR values using a non-decreasing array of indices.
    void gather_indexed_varchar(size_t n, uint8_t* target, size_t target_step, const uint32_t* indices) const {
        if (n == 0) return;

        // --- State variables for the forward scan ---
        uint32_t current_offset = indices[0] + start_row_in_page_;
        uint32_t prev_offset = current_offset;
        uint32_t current_page_idx = start_page_idx_;
        uint32_t prev_page_idx = std::numeric_limits<uint32_t>::max();

        PageReader reader(source_column_, current_page_idx);
        uint32_t rows_on_page = reader.getRowCount();

        // --- Page-local cached state ---
        Bitmap bitmap = Bitmap(nullptr);
        bool is_nonnull_page = false;
        size_t nonnull_count = 0;

        for (uint32_t i = 0; i < n; i++) {
            // Step 1: Locate the correct page for the current logical offset.
            while (current_offset >= rows_on_page) {
                current_offset -= rows_on_page;
                current_page_idx++;
                reader = PageReader(source_column_, current_page_idx);
                rows_on_page = reader.getRowCount();
            }

            // Step 2: If we've moved to a new page, reset page-local state.
            if (prev_page_idx != current_page_idx) {
                bitmap = reader.getBitmap();
                nonnull_count = bitmap.getNonNullCount(current_offset);
                is_nonnull_page = (rows_on_page == reader.getNonNullRowCount());
                prev_offset = current_offset;
                prev_page_idx = current_page_idx;
            }

            assert(!reader.isLongStringFollow());

            // --- Path 1: The current row is a long string that starts on this page ---
            if (reader.isLongStringStart()) {
                // Find all consecutive pages belonging to this single long string.
                uint32_t end_page_idx = current_page_idx + 1;
                while (end_page_idx < source_column_->pages.size() && PageReader(source_column_, end_page_idx).isLongStringFollow()) {
                    end_page_idx++;
                }
                // Create a 'long string' varchar_ptr pointing to the sequence of pages.
                varchar_ptr val(&source_column_->pages[current_page_idx], end_page_idx - current_page_idx);
                *reinterpret_cast<varchar_ptr*>(target) = val;

                // --- Update state for the *next* iteration ---
                // This logic is complex because a multi-page long string consumes only one logical row index.
                if (i + 1 < n && indices[i + 1] > indices[i]) {
                    // Jump the page cursor past all pages of the long string we just processed.
                    current_page_idx = end_page_idx;
                    reader = PageReader(source_column_, current_page_idx);
                    rows_on_page = reader.getRowCount();
                    // The new offset is the delta between indices, minus 1 for the long string itself.
                    current_offset = indices[i + 1] - indices[i] - 1;
                } else {
                    // This happens if the next requested index is the same as the current one
                    // (i.e., we need to fetch the same long string again). Reset offset for the next loop.
                    current_offset = 0;
                }
            } else {
                // --- Path 2: The current row is a short string or a NULL value ---

                // NOTE: This optimization is UNUSED.
                if (false && __glibc_likely(is_nonnull_page)) {
                    const uint16_t* str_end = reader.getVarcharOffset() + current_offset;
                    uint16_t str_begin = (current_offset == 0) ? 0 : *(str_end - 1);
                    varchar_ptr val(reinterpret_cast<const char*>(reader.getPageData()) + str_begin, *str_end - str_begin);
                    *reinterpret_cast<varchar_ptr*>(target) = val;
                } else if (bitmap.isNotNull(current_offset)) {
                    // For non-null values, first find the physical index in the offset array (`nonnull_count`)...
                    nonnull_count += bitmap.getNonNullCount(prev_offset, current_offset);
                    const uint16_t* str_end_ptr = reader.getVarcharOffset() + nonnull_count;
                    // ...then use the offsets at that index to locate the string data.
                    uint16_t str_begin_offset = (nonnull_count == 0) ? 0 : *(str_end_ptr - 1);
                    varchar_ptr val(reinterpret_cast<const char*>(reader.getPageData()) + str_begin_offset, *str_end_ptr - str_begin_offset);
                    *reinterpret_cast<varchar_ptr*>(target) = val;
                    prev_offset = current_offset;
                } else {
                    // The value is null.
                    *reinterpret_cast<uint64_t*>(target) = Constants::NULL_VARCHAR;
                }

                // Advance the logical offset using the delta from the indices array for the next iteration.
                if (i + 1 < n) {
                    current_offset += (indices[i + 1] - indices[i]);
                }
            }
            target += target_step;
        }
    }
};




/**
 * @class OperatorResultTable
 * @brief Represents a logical table, typically the output of a query operator's `next()` call.
 *
 * This class acts as a container for a batch of rows. A key feature is that its columns
 * can be either materialized (`InstantiatedColumn`) or non-materialized (`ContinuousColumn`).
 * This flexibility allows for efficient data flow between operators:
 *
 * - A `Scan` operator produce a result table with only `ContinuousColumn`s,
 *   representing a view over a range of rows in the source data without copying anything.
 * - A `HashJoin` operator produce a table where all columns are
 *   materialized (`InstantiatedColumn`) for performance
 */
class OperatorResultTable {
public:
    /// @brief The number of logical rows in this result batch.
    size_t num_rows_{0};
    /// @brief A vector of column pointers, each representing a column in the logical table.
    /// The memory for the columns is managed by `local_allocator`.
    LocalVector<ColumnInterface*> columns_;

    /// @brief Checks if the result table contains any rows.
    bool isEmpty() const { return num_rows_ == 0; }

    /// @brief Prints a string representation of the table to standard output.
    void print() const {
        std::cout << toString();
    }

    /**
     * @brief Adds a non-materialized, continuous column to the table.
     * @param source_column The original column containing the page data.
     * @param start_page_idx The starting page index for this column's view.
     * @param start_row_in_page The starting row offset within the start page.
     */
    void addContinuousColumn(const Column* source_column, uint32_t start_page_idx, uint32_t start_row_in_page) {
        columns_.push_back(local_allocator.make<ContinuousColumn>(source_column, start_page_idx, start_row_in_page));
    }

    /**
     * @brief Adds a materialized, instantiated column to the table.
     * @param type The data type of the column.
     * @param data A non-owning pointer to the materialized data buffer.
     */
    void addInstantiatedColumn(DataType type, void* data) {
        columns_.push_back(local_allocator.make<InstantiatedColumn>(type, (uint8_t*)data));
    }

    /**
     * @brief Advances the logical cursor of every column in the table by a number of rows.
     *
     * This is used to efficiently skip over rows that have already been processed.
     * @param skip_rows The number of rows to skip.
     */
    void skipRows(size_t skip_rows) {
        for (ColumnInterface* column : columns_) {
            column->skipRows(skip_rows);
        }
    }

    /**
     * @brief Generates a string representation of the entire table.
     * @param head If true, includes a header with table dimensions.
     * @return A string containing the formatted table data.
     */
    std::string toString(bool head = true) const {
        std::ostringstream oss;
        if (head) {
            oss << "table size: " << num_rows_ << " rows * " << columns_.size() << " cols\n";
        }

        for (size_t i = 0; i < num_rows_; ++i) {
            for (size_t j = 0; j < columns_.size(); ++j) {
                // `valueToString` is a virtual call, correctly handled for both column types.
                oss << columns_[j]->valueToString(i) << "\t\t";
            }
            oss << "\n";
        }
        return oss.str();
    }
};


/**
 * @class TempPage
 * @brief An abstract base class for in-memory page builders.
 *
 * This class defines the interface for helper classes that construct the complex
 * byte layout of a data page in memory before it is finalized and "dumped" into
 * a persistent `Page` object. It also provides static utility functions for
 * manipulating bitmaps.
 */
class TempPage {
public:
    /// @brief Finalizes the in-memory data and returns a newly allocated `Page` object.
    virtual Page* dump_page() = 0;
    /// @brief Checks if the temporary page currently holds any data.
    virtual bool is_empty() = 0;
    virtual ~TempPage() = default;

    /**
     * @brief A static helper to set a bit in a bitmap vector.
     *
     * The vector will be automatically grown if the index is out of bounds.
     * @tparam Alloc The allocator type of the vector.
     * @param bitmap The bitmap vector to modify.
     * @param idx The bit index to set to 1 (non-null).
     */
    template <typename Alloc>
    static void set_bitmap(std::vector<uint8_t, Alloc>& bitmap, uint16_t idx) {
        // Ensure the vector is large enough to contain the target bit.
        while (bitmap.size() * 8 <= idx) {
            bitmap.emplace_back(0);
        }
        auto byte_idx = idx / 8;
        auto bit = idx % 8;
        bitmap[byte_idx] |= (1u << bit);
    }

    /**
     * @brief A static helper to unset a bit in a bitmap vector.
     *
     * The vector will be automatically grown if the index is out of bounds.
     * @tparam Alloc The allocator type of the vector.
     * @param bitmap The bitmap vector to modify.
     * @param idx The bit index to set to 0 (null).
     */
    template <typename Alloc>
    static void unset_bitmap(std::vector<uint8_t, Alloc>& bitmap, uint16_t idx) {
        // Ensure the vector is large enough to contain the target bit.
        while (bitmap.size() * 8 <= idx) {
            bitmap.emplace_back(0);
        }
        auto byte_idx = idx / 8;
        auto bit = idx % 8;
        bitmap[byte_idx] &= ~(1u << bit);
    }
};


/**
 * @class TempIntPage
 * @brief A concrete page builder for INT32 data.
 *
 * This class handles the in-memory construction of a page for integer values.
 * It manages the page header (row counts), the compacted data array for non-null
 * values, and the nullability bitmap. The actual `Page` object is allocated
 * lazily on the first call to `add_row`.
 */
class TempIntPage : public TempPage {
private:
    /// @brief A temporary, growing buffer for the nullability bitmap.
    LocalVector<uint8_t> bitmap;
    /// @brief A pointer to the `Page` object being built. It is `nullptr` until the first row is added.
    Page* page = nullptr;

    /// @brief A private helper for lazy allocation and initialization of the Page object.
    void alloc_page() {
#ifdef PROFILER
        if (page != nullptr) {
            throw std::runtime_error("page is not null");
        }
#endif
        page = new Page;
        memset(page->data, 0, PAGE_SIZE);
    }

public:
    TempIntPage() {
        // Pre-allocate some memory for the bitmap to reduce reallocations.
        bitmap.reserve(256);
    }

    /// @brief Provides direct reference-based access to the total row count in the page header.
    uint16_t& num_rows() {
        return *(uint16_t*)(page->data);
    }

    /// @brief Provides direct reference-based access to the non-null value count in the page header.
    uint16_t& num_values() {
        return *(uint16_t*)(page->data + 2);
    }

    /// @brief Provides a direct pointer to the start of the data section within the page.
    int32_t* data() {
        return (int32_t*)(page->data + 4);
    }

    /**
     * @brief Checks if adding one more value would cause the page to exceed its capacity.
     *
     * The calculation accounts for the header (4 bytes), the space for all existing
     * non-null values plus one new one, and the size of the nullability bitmap.
     * @return `true` if the page is full, `false` otherwise.
     */
    bool is_full() {
        if (page == nullptr) {
            return false;
        } else {
            return 4 + (num_values() + 1) * 4 + (num_rows() / 8 + 1) > PAGE_SIZE;
        }
    }

    /// @brief Checks if the builder is in a fresh state (no page allocated yet).
    bool is_empty() override {
        return page == nullptr;
    }

    /**
     * @brief Adds a new integer value to the page.
     *
     * Lazily allocates the page on the first call. It correctly handles null and
     * non-null values by updating the bitmap and, if non-null, appending the
     * value to the compacted data array.
     * @param value The integer value to add.
     */
    void add_row(int value) {
#ifdef PROFILER
        if (is_full()) {
            throw std::runtime_error("page is full");
        }
#endif
        if (is_empty()) {
            alloc_page();
        }

        if (value != Constants::NULL_INT32) {
            set_bitmap(bitmap, num_rows());
            data()[num_values()] = value;
            num_values()++;
        } else {
            // A value of 0 in the bitmap is sufficient; unsetting is for clarity.
            // unset_bitmap(bitmap, num_rows());
        }
        num_rows()++;
    }

    /**
     * @brief Finalizes the page construction and returns the completed `Page`.
     *
     * This method copies the temporary bitmap to its final destination at the end
     * of the page buffer, then resets the builder's state so it can be reused.
     * Ownership of the returned `Page` is transferred to the caller.
     * @return The newly constructed `Page` object.
     */
    Page* dump_page() override {
        Page* result_page = page;
        // Copy the bitmap from its temporary vector to its final position at the end of the page.
        memcpy(page->data + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());
        bitmap.clear();
        page = nullptr; // Reset state for potential reuse.
        return result_page;
    }
};

/**
 * @class TempStringPage
 * @brief A concrete page builder for VARCHAR data.
 *
 * This class constructs a page for short, variable-length strings. It uses three
 * separate temporary vectors to manage the character data heap, the end-offsets array,
 * and the nullability bitmap. In the `dump_page` method, these components are
 * assembled into the final, correct page layout.
 */
class TempStringPage : public TempPage {
private:
    uint16_t num_rows = 0;
    /// @brief A temporary buffer for the concatenated character data of all non-null strings.
    LocalVector<char> data;
    /// @brief A temporary buffer for the end-offset of each non-null string within the data heap.
    LocalVector<uint16_t> offsets;
    /// @brief A temporary buffer for the nullability bitmap.
    LocalVector<uint8_t> bitmap;

public:
    TempStringPage() {
        data.reserve(8192);
        offsets.reserve(4096);
        bitmap.reserve(512);
    }

    /// @brief Checks if any rows have been added to the page.
    bool is_empty() override {
        return num_rows == 0;
    }

    /**
     * @brief Checks if a given string can be stored on the current page without exceeding its capacity.
     * @param value The `varchar_ptr` representing the string to add.
     * @return `true` if there is sufficient space, `false` otherwise.
     */
    bool can_store_string(const varchar_ptr value) {
        // A single long string must occupy an entire page by itself.
        if (value.isLongString()) {
            return num_rows == 0;
        }

        // Calculate the total required space:
        // header (4) + offsets array + data heap + bitmap.
        size_t required_space;
        if (value.isNull()) {
            required_space = 4 + offsets.size() * 2 + data.size() + (num_rows / 8 + 1);
        } else {
            // For a non-null value, add space for one more offset and the string's characters.
            required_space = 4 + (offsets.size() + 1) * 2 + (data.size() + value.length()) + (num_rows / 8 + 1);
        }
        return required_space <= PAGE_SIZE;
    }

    /**
     * @brief Adds a new string value to the page.
     *
     * For non-null values, it appends the character data to the `data` heap and
     * records the new cumulative size in the `offsets` vector.
     * @param value The `varchar_ptr` representing the string to add.
     */
    void add_row(const varchar_ptr value) {
#ifdef PROFILER
        if (!can_store_string(value)) {
            throw std::runtime_error("page is full");
        }
        // This check appears to be for a different, unsupported long string mechanism.
        if (value.length() > PAGE_SIZE - 7) {
            throw std::runtime_error("long string is not support");
        }
#endif
        if (value.isNull()) {
            // unset_bitmap(bitmap, num_rows);
        } else {
            set_bitmap(bitmap, num_rows);
            // Append the string's character data to the data heap.
            data.insert(data.end(), value.string(), value.string() + value.length());
            // Record the new end-offset.
            offsets.emplace_back(data.size());
        }
        ++num_rows;
    }

    /**
     * @brief Assembles and finalizes the page from internal buffers.
     *
     * This method allocates a new `Page` and meticulously copies the header, offset array,
     * data heap, and bitmap into their correct positions, creating the final page layout.
     * It then resets the builder's state for reuse.
     * @return The newly constructed `Page` object.
     */
    Page* dump_page() override {
        auto* page = new Page;
        // Write header: total rows and non-null rows.
        *reinterpret_cast<uint16_t*>(page->data) = num_rows;
        *reinterpret_cast<uint16_t*>(page->data + 2) = static_cast<uint16_t>(offsets.size());
        // Write the offset array after the header.
        memcpy(page->data + 4, offsets.data(), offsets.size() * 2);
        // Write the character data heap after the offset array.
        memcpy(page->data + 4 + offsets.size() * 2, data.data(), data.size());
        // Write the bitmap at the very end of the page.
        memcpy(page->data + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());

        // Reset state for potential reuse.
        num_rows = 0;
        data.clear();
        offsets.clear();
        bitmap.clear();
        return page;
    }
};

} // namespace Contest