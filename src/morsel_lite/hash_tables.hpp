#pragma once

#include "hardware.h"

#ifdef SPC__X86_64
# include <immintrin.h>
#endif

#include "column_vectors.hpp"
#include "concurrent_segment_vector.hpp"
#include "data_chunck.hpp"
#include "hash.hpp"

namespace Contest::MorselLite {

class ExecutionContext;

enum class HashSetType {
  SINGLE_VALUE, CHAINED
};

class HashSet {
public:
  using key_type = int32_t;

public:
  HashSet(HashSetType type) : type_(type) { }

  HashSet(const HashSet &) = delete;
  HashSet(HashSet &&) = delete;

  virtual ~HashSet() = 0;

  HashSet &operator=(const HashSet &) = delete;
  HashSet &operator=(HashSet &&) = delete;

  [[nodiscard]] HashSetType type() const noexcept { return type_; }

  [[nodiscard]] virtual size_t size() const = 0;
  [[nodiscard]] bool empty() const { return size() == 0; }

  virtual void insert(uint32_t thread_id, const FlatColumnVector &key_column) = 0;
  virtual void clear() = 0;

  virtual void lookup(uint32_t thread_id, Span<const DataChunck> &manage_data_chuncks, 
                      const FlatColumnVector &key_column, DataChunck &data_chunck, 
                      Span<const ProjectColumnDesc> probed_output_column_descs) const = 0;

private:
  HashSetType type_;
};

enum class HashMapType {
  SINGLE_VALUE, CHAINED
};

class HashMap {
public:
  using key_type = int32_t;
  static constexpr uint64_t DATA_REFID_BITS = 48;
  static constexpr uint64_t DATA_REFID_MASK = (static_cast<uint64_t>(1) << DATA_REFID_BITS) -1 ;

public:
  struct Entry {
    uint64_t key;
    uint64_t val;

    [[nodiscard]] uint64_t chunck_id() const { return val >> DATA_REFID_BITS; }
    [[nodiscard]] uint64_t chunck_offset() const { return val & DATA_REFID_MASK; }

    [[nodiscard]] static Entry make() noexcept {
      return Entry{~static_cast<uint64_t>(0), ~static_cast<uint64_t>(0)};
    }

    [[nodiscard]] static Entry make(int32_t key, uint64_t val) noexcept {
      return Entry{static_cast<uint64_t>(key), val};
    }

    [[nodiscard]] static Entry make(uint64_t key, uint64_t chunck_id, uint64_t chunck_offset) noexcept {
      return Entry{key, (chunck_id << 48) | chunck_offset};
    }
  };

public:
  HashMap(HashMapType type) : type_(type) { }

  HashMap(const HashMap &) = delete;
  HashMap(HashMap &&) = delete;

  HashMap &operator=(const HashMap &) = delete;
  HashMap &operator=(HashMap &&) = delete;

  virtual ~HashMap() = default;

  [[nodiscard]] HashMapType type() const noexcept { return type_; }
  [[nodiscard]] virtual size_t size() const = 0;
  [[nodiscard]] bool empty() const { return size() == 0; }

  virtual void insert(uint32_t thread_id, const FlatColumnVector &key_column, uint64_t data_chunck_id) = 0;

  virtual void clear() = 0;

  virtual void lookup(uint32_t thread_id, Span<const DataChunck> managed_data_chuncks, 
                      const FlatColumnVector &key_column, DataChunck &data_chunck, 
                      Span<const ProjectColumnDesc> probed_output_column_descs,
                      Span<const ProjectColumnDesc> joined_output_column_descs) const = 0;
private:
  HashMapType type_;
};

class SingleValueHashMap : public HashMap {
public:
  SingleValueHashMap()
    : HashMap(HashMapType::SINGLE_VALUE) { }
  
  ~SingleValueHashMap() = default;

  [[nodiscard]] size_t size() const override { 
    return entry_.key != ~static_cast<uint64_t>(0); 
  }

  void insert(uint32_t thread_id, const FlatColumnVector &key_column, uint64_t data_chunck_id) override {
    ML_ASSERT(empty());
    ML_ASSERT(key_column.num_rows() == 1);
    entry_ = Entry::make(key_column.int32_view()[0], data_chunck_id, 0);
  }
  
  void clear() override {
    entry_ = Entry::make();
  }

  void lookup(uint32_t thread_id, Span<const DataChunck> managed_data_chuncks, 
              const FlatColumnVector &key_column, DataChunck &data_chunck, 
              Span<const ProjectColumnDesc> probed_output_column_descs,
              Span<const ProjectColumnDesc> joined_output_column_descs) const override {
    ML_ASSERT(key_column.data_type() == DataType::INT32);
    auto key_view = key_column.int32_view();
    const size_t num_output_columns = probed_output_column_descs.size() + joined_output_column_descs.size();
    std::vector<ColumnVector> columns(num_output_columns);

    const int32_t key = entry_.key;
    const auto &joined_data_chunck = managed_data_chuncks[entry_.chunck_id()];
    
    std::vector<size_t> matched_row_ids;
    for (size_t i = 0; i < key_view.size(); ++i) {
      if (key_view[i] == key) {
        matched_row_ids.push_back(i);
      }
    }

    for (const auto &desc : probed_output_column_descs) {
      if (desc.data_type == DataType::VARCHAR) {
        columns[desc.out_index] = RowIdColumnVector(*data_chunck.column(desc.ref_index).as_rowid_vector().column());
      } else {
        columns[desc.out_index] = FlatColumnVector(desc.data_type);
      }
    }

    for (const auto &desc : joined_output_column_descs) {
      if (desc.data_type == DataType::VARCHAR) {
        columns[desc.out_index] = RowIdColumnVector();
        columns[desc.out_index].as_rowid_vector().resize(matched_row_ids.size());
      } else {
        columns[desc.out_index] = FlatColumnVector(desc.data_type);
        columns[desc.out_index].as_flat_vector().resize(matched_row_ids.size());
      }
    }

    std::array<int16_t, 16> col_flags;
    col_flags.fill(-1);

    for (const auto &desc : probed_output_column_descs) {
      if (col_flags[desc.ref_index] == -1) {
        auto &old_column = data_chunck.column(desc.ref_index);
        old_column.select(matched_row_ids);
        columns[desc.out_index] = std::move(old_column);
        col_flags[desc.ref_index] = desc.out_index;
      } else {
        columns[desc.out_index] = columns[col_flags[desc.ref_index]].clone();
      }
    }

    for (const auto &desc : joined_output_column_descs) {
      const auto &src_col = joined_data_chunck.column(desc.ref_index);

      auto impl = [&](auto src_view) {
        auto val = src_view[entry_.chunck_offset()];
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, size_t>) {
          auto dst_view = columns[desc.out_index].as_rowid_vector().view();
          std::fill_n(dst_view.begin(), dst_view.size(), val);
        } else {
          auto dst_view = columns[desc.out_index].as_flat_vector().view_as<T>();
          std::fill_n(dst_view.begin(), dst_view.size(), val);
        }
      };

      switch (desc.data_type) {
      case DataType::INT32   : impl(src_col.as_flat_vector().int32_view()); break;
      case DataType::INT64   : impl(src_col.as_flat_vector().int64_view()); break;
      case DataType::FP64    : impl(src_col.as_flat_vector().float64_view()); break;
      case DataType::VARCHAR : impl(src_col.as_rowid_vector().view()); break;
      }
    }

    data_chunck = DataChunck{std::move(columns)};
  }

private:
  Entry entry_;
};

class ChainedHashMap : public HashMap {
public:
  static constexpr size_t DEFAULT_NUM_BUCKETS = (1 << 18) / sizeof(void *);

  struct Bucket {
    std::atomic<uint16_t>          bloom{0};
    ConcurrentSegmentVector<Entry> entries;
  };

public:
  ChainedHashMap()
    : HashMap(HashMapType::CHAINED) { }

  ~ChainedHashMap() override = default;

  [[nodiscard]] size_t size() const override { 
    size_t num = 0;
    for (const auto &bucket : buckets_) {
      num += bucket.entries.size();
    }
    return num;
  }

  void insert(uint32_t thread_id, const FlatColumnVector &key_column, uint64_t data_chunck_id) override {
    ML_ASSERT(key_column.data_type() == DataType::INT32);
    ML_ASSERT(data_chunck_id < (1 << 16));

    auto key_view = key_column.int32_view();
    ML_ASSERT(key_view.size() <= (static_cast<size_t>(1) << 48));

    for (size_t i = 0; i < key_view.size(); ++i) {
      auto entry = Entry::make(key_view[i], data_chunck_id, i);

      auto hashval = key_to_hash(key_view[i]);
      uint64_t bucket_index = hashval % DEFAULT_NUM_BUCKETS;
      uint64_t tag = static_cast<uint16_t>(hashval >> (sizeof(hashval) * 8 - 4));

      auto &bucket = buckets_[bucket_index];
      bucket.bloom.fetch_or(1 << tag, std::memory_order_relaxed);
      bucket.entries.push_back(entry);
    }
    size_.fetch_add(key_column.num_rows(), std::memory_order_relaxed);
  }

  void clear() override {
    if (size_.load() == 0) { return; }

    for (size_t i = 0; i < DEFAULT_NUM_BUCKETS; ++i) {
      buckets_[i].bloom = 0;
      buckets_[i].entries.clear();
    }
    size_.store(0);
  }

#if 0
  void insert(const FlatColumnVector &key_column, uint64_t data_chunck_id, 
              Span<std::pair<uint16_t, std::vector<Entry>>> bucket_buf) override {
    ML_ASSERT(key_column.data_type() == DataType::INT32);
    ML_ASSERT(data_chunck_id < (1 << 16));

    auto key_view = key_column.int32_view();
    ML_ASSERT(key_view.size() <= (static_cast<size_t>(1) << 48));

    for (size_t i = 0; i < DEFAULT_NUM_BUCKETS; ++i) {
      bucket_buf[i].first = 0;
      bucket_buf[i].second.clear();
    }

    for (size_t i = 0; i < key_view.size(); ++i) {
      uint64_t hashval = hash_integer(key_view[i]);
      uint64_t bucket_idx = hashval % DEFAULT_NUM_BUCKETS;
      uint16_t tag_mask = static_cast<uint16_t>(hashval >> 48);

      auto &bkt_buf = bucket_buf[bucket_idx];
      bkt_buf.first |= tag_mask;
      bkt_buf.second.push_back(Entry::make(key_view[i], data_chunck_id, i));
    }

    for (size_t i = 0; i < DEFAULT_NUM_BUCKETS; ++i) {
      if (bucket_buf[i].second.empty()) { continue; }
      buckets_[i].entries.append(bucket_buf[i].second.begin(), bucket_buf[i].second.size());
      buckets_[i].bloom |= bucket_buf[i].first;
    }
    size_.fetch_add(key_column.num_rows(), std::memory_order_relaxed);
  }
#endif

  void lookup(uint32_t thread_id, Span<const DataChunck> managed_data_chuncks, 
              const FlatColumnVector &key_column, DataChunck &data_chunck, 
              Span<const ProjectColumnDesc> probed_output_column_descs,
              Span<const ProjectColumnDesc> joined_output_column_descs) const override {
    ML_ASSERT(key_column.data_type() == DataType::INT32);
    auto key_view = key_column.int32_view();

    const size_t num_output_columns = probed_output_column_descs.size() + joined_output_column_descs.size();
    std::vector<ColumnVector> columns(num_output_columns);

    std::vector<size_t> matched_row_ids;
    std::vector<size_t> factors;
    std::vector<uint64_t> joined_ref_ids;

    size_t num_joined_matched = 0;

    for (size_t i = 0; i < key_view.size(); ++i) {
      auto hashval = key_to_hash(key_view[i]);
      size_t bucket_index = hashval % DEFAULT_NUM_BUCKETS;
      uint16_t tag = static_cast<uint16_t>(hashval >> (sizeof(hashval) * 8 - 4));

      const auto &bucket = buckets_[bucket_index];

      //if ((bucket.bloom.load(std::memory_order_relaxed) & (1 << tag)) == 0) {
      //  continue;
      //}

      size_t num_matched = 0;
      size_t bucket_size = bucket.entries.size();
      for (size_t j = 0; j < bucket_size; ++j) {
        auto entry = bucket.entries[j];
        if (entry.key == key_view[i]) {
          joined_ref_ids.push_back(entry.val);
          ++num_matched;
        }
      }

      if (num_matched != 0) {
        matched_row_ids.push_back(i);
        factors.push_back(num_matched);
        num_joined_matched += num_matched;
      }
    }

    for (const auto &desc : probed_output_column_descs) {
      if (desc.data_type == DataType::VARCHAR) {
        columns[desc.out_index] = RowIdColumnVector(*data_chunck.column(desc.ref_index).as_rowid_vector().column());
      } else {
        columns[desc.out_index] = FlatColumnVector(desc.data_type);
      }
    }

    for (const auto &desc : joined_output_column_descs) {
      if (desc.data_type == DataType::VARCHAR) {
        columns[desc.out_index] = RowIdColumnVector();
        columns[desc.out_index].as_rowid_vector().resize(joined_ref_ids.size());
      } else {
        columns[desc.out_index] = FlatColumnVector(desc.data_type);
        columns[desc.out_index].as_flat_vector().resize(joined_ref_ids.size());
      }
    }

    if (num_joined_matched == 0) {
      data_chunck = DataChunck{std::move(columns)};
      return;
    }

    std::array<int16_t, 16> col_flags;
    col_flags.fill(-1);

    for (const auto &desc : probed_output_column_descs) {
      if (col_flags[desc.ref_index] == -1) {
        auto &old_column = data_chunck.column(desc.ref_index);
        old_column.select_and_apply_factors(matched_row_ids, factors);
        columns[desc.out_index] = std::move(old_column);
        col_flags[desc.ref_index] = desc.out_index;
      } else {
        columns[desc.out_index] = columns[col_flags[desc.ref_index]].clone();
      }
    }

    auto gather_data = [&](ColumnVector &dst_col, auto dst_col_view, 
                          const ProjectColumnDesc &desc) {
      using T = typename std::remove_cv_t<std::remove_reference_t<decltype(dst_col_view)>>::value_type;

      for (size_t i = 0; i < joined_ref_ids.size(); ++i) {
        uint64_t ref_id = joined_ref_ids[i];
        uint64_t chunck_id = ref_id >> DATA_REFID_BITS;
        uint64_t chunck_offset = ref_id & DATA_REFID_MASK;
        const auto &src_col = managed_data_chuncks[chunck_id].column(desc.ref_index);

        if constexpr (std::is_same_v<T, size_t>) {
          if (dst_col.as_rowid_vector().column() == nullptr) {
            dst_col.as_rowid_vector().set_column(src_col.as_rowid_vector().column());
          }
          auto src_view = src_col.as_rowid_vector().view();
          dst_col_view[i] = src_view[chunck_offset];
        } else {
          auto src_view = src_col.as_flat_vector().view_as<T>();
          dst_col_view[i] = src_view[chunck_offset];
        }
      }
    };

    for (const auto &desc : joined_output_column_descs) {
      auto &dst_col = columns[desc.out_index];

      switch (desc.data_type) {
      case DataType::VARCHAR : gather_data(dst_col, dst_col.as_rowid_vector().view(), desc); break;
      case DataType::INT32   : gather_data(dst_col, dst_col.as_flat_vector().int32_view(), desc); break;
      case DataType::INT64   : gather_data(dst_col, dst_col.as_flat_vector().int64_view(), desc); break;
      case DataType::FP64    : gather_data(dst_col, dst_col.as_flat_vector().float64_view(), desc); break;
      }
    }

    data_chunck = DataChunck{std::move(columns)};
  }

private:
  std::array<Bucket, DEFAULT_NUM_BUCKETS> buckets_;
  std::atomic<size_t>                     size_{0};
  //std::array<uint64_t, DEFAULT_NUM_BUCKETS / 64> bucket_bitmap_;

  [[nodiscard]] static uint64_t key_to_hash(key_type key) noexcept {
    return hash_integer(key);
  }
};



}