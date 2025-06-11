#pragma once

#include <deque>
#include <list>

#include "data_chunck.hpp"
#include "hash_tables.hpp"
#include "page.hpp"

namespace Contest::MorselLite {

class ExecutionContext {
public:
  static constexpr size_t MAX_CHAINED_HASH_MAPS = 16;
  static constexpr size_t MAX_SINGLE_VALUE_HASH_MAPS = 16;
  static constexpr size_t MAX_HASH_MAPS = 32;

  static constexpr size_t MAX_MANAGED_DATA_CHUNCKS = static_cast<size_t>(1) << 16;

public:
  ExecutionContext() {
    hash_map_pool_.chained.resize(MAX_CHAINED_HASH_MAPS);
    for (size_t i = 0; i < hash_map_pool_.chained.size(); ++i) {
      hash_map_pool_.chained[i] = std::make_unique<ChainedHashMap>();
    }

    hash_map_pool_.single_value.resize(MAX_SINGLE_VALUE_HASH_MAPS);
    for (size_t i = 0; i < hash_map_pool_.single_value.size(); ++i) {
      hash_map_pool_.single_value[i] = std::make_unique<SingleValueHashMap>();
    }
  }

  [[nodiscard]] size_t manage_data_chunck(DataChunck &&data_chunck) {
    size_t id = managed_data_chunck_alloc_id_.fetch_add(1, std::memory_order_relaxed);
    managed_data_chuncks_[id] = std::move(data_chunck);
    return id;
  }

  [[nodiscard]] const DataChunck &get_data_chunck(size_t id) const {
    ML_ASSERT(id < managed_data_chunck_alloc_id_.load());
    return managed_data_chuncks_[id];
  }

  [[nodiscard]] Span<const DataChunck> get_data_chuncks() const {
    return managed_data_chuncks_;
  }

  [[nodiscard]] DataChunck &get_data_chunck(size_t id) {
    ML_ASSERT(id < managed_data_chunck_alloc_id_.load());
    return managed_data_chuncks_[id];
  }

  template<typename HashMapT, typename ...ArgsT, typename = std::enable_if_t<std::is_base_of_v<HashMap, HashMapT>>>
  HashMap &init_hash_table(size_t id, ArgsT &&...args) {
    HashMap *map = nullptr;
    if constexpr (std::is_same_v<HashMapT, ChainedHashMap>) {
      auto *ht = hash_map_pool_.chained[hash_map_pool_.chained_alloc_id++].get();
      ht->clear();
      hash_maps_[id] = ht;
      map = ht;
    } else {
      auto *ht = hash_map_pool_.single_value[hash_map_pool_.single_value_alloc_id++].get();
      ht->clear();
      hash_maps_[id] = ht;
      map = ht;
    }
    return *map;
  }

  [[nodiscard]] const HashMap &get_hash_table(size_t id) const {
    return *hash_maps_[id];
  }

  [[nodiscard]] HashMap &get_hash_table(size_t id) {
    return *hash_maps_[id];
  }

  const VarcharColumnPageLocator &init_varchar_column_page_locator(const Column *col) {
    auto [iter, succ] = varchar_page_locators_.try_emplace(col, *col);
    return iter->second;
  }

  [[nodiscard]] const VarcharColumnPageLocator &get_varchar_column_page_locator(const Column *col) const {
    auto iter = varchar_page_locators_.find(col);
    ML_ASSERT(iter != varchar_page_locators_.end());
    return iter->second;
  }

  [[nodiscard]] const std::unordered_map<const Column *, VarcharColumnPageLocator> &
  get_varchar_column_page_locators() const noexcept {
    return varchar_page_locators_;
  }

  const FixedColumnPageLocator &init_fixed_column_page_locator(const Column *col) {
    auto [iter, succ] = fixed_page_locators_.try_emplace(col, *col);
    return iter->second;
  }

  [[nodiscard]] const FixedColumnPageLocator &get_fixed_column_page_locator(const Column *col) const {
    auto iter = fixed_page_locators_.find(col);
    ML_ASSERT(iter != fixed_page_locators_.end());
    return iter->second;
  }

  [[nodiscard]] const std::unordered_map<const Column *, FixedColumnPageLocator> &
  get_fixed_column_page_locators() const noexcept {
    return fixed_page_locators_;
  }

  void add_result_partition(DataChunck &&data_chunck) {
    DataChunck *new_part = [&] () mutable -> DataChunck * {
      std::unique_lock<std::mutex> lk(result_partitions_mtx_);
      auto &ref = result_partitions_.emplace_back();
      return std::addressof(ref);
    }();
    *new_part = std::move(data_chunck);
  }
  
  [[nodiscard]] ColumnarTable get_result_table(Span<const DataType> out_data_types) {
    ColumnarTable result;
    result.num_rows = 0;
    result.columns.reserve(out_data_types.size());
    for (size_t i = 0; i < out_data_types.size(); ++i) {
      result.columns.emplace_back(out_data_types[i]);
    }

    size_t num_total_rows = 0;
    for (const auto &data_chunck : result_partitions_) {
      num_total_rows += data_chunck.num_rows();
    }

    if (num_total_rows == 0) {
      result.num_rows = 0;
      return result;
    }

    result.num_rows = num_total_rows;

    auto impl_fixed = [&](auto _ty, size_t col_idx) {
      using T = ML_TYPE_OF_TAG(_ty);

      auto &dst_col = result.columns[col_idx];

      auto writer = create_page_writer<T>(
        [&]() { return dst_col.new_page(); },
        [](Page *) { /* do nothing */ }
      );

      for (const auto &data_chunck : result_partitions_) {
        const auto &src_col = data_chunck.column(col_idx);
        auto src_col_view = src_col.as_flat_vector().view_as<T>();
        writer.write_values(src_col_view.begin(), src_col_view.size());
      }
      writer.flush();
    };

    auto impl_varchar = [&](size_t col_idx) {
      auto &dst_col = result.columns[col_idx];
      
      auto writer = create_page_writer<std::string>(
        [&]() { return dst_col.new_page(); },
        [](Page *) { /* do nothing */ }
      );

      const Column *base_column = result_partitions_.front().column(col_idx).as_rowid_vector().column();
      const auto &locator = get_varchar_column_page_locator(base_column);

      std::vector<std::pair<size_t, size_t>> shuffled_indices(result.num_rows);
      size_t shuffled_writer = 0;
      for (const auto &data_chunck : result_partitions_) {
        const auto &src_col = data_chunck.column(col_idx).as_rowid_vector();
        ML_ASSERT(src_col.column() == base_column);

        auto src_col_view = src_col.view();
        for (size_t i = 0; i < src_col_view.size(); ++i) {
          shuffled_indices[shuffled_writer] = {src_col_view[i], shuffled_writer};
          ++shuffled_writer;
        }
      }

      auto is_compact_page = [&](size_t page_idx) {
        return *reinterpret_cast<const uint16_t *>(base_column->pages[page_idx]->data) < 0xFFFE;
      };

      std::vector<std::string_view> shuffled_strs(result.num_rows);
      std::list<std::string> long_str_buffer;

      size_t search_hint = 0;
      for (size_t i = 0; i < shuffled_indices.size(); ++i) {
        auto [loc, new_search_hint] = locator.get_location(shuffled_indices[i].first, 0);
        search_hint = new_search_hint;
        if (loc.has_data_index()) {
          if (is_compact_page(loc.page_index())) {
            CompactStringPageReader reader(base_column->pages[loc.page_index()], loc.page_index());
            shuffled_strs[shuffled_indices[i].second] = reader.read_str(loc);
          } else {
            auto &long_str = long_str_buffer.emplace_back();
            size_t cur_page_idx = loc.page_index();
            LongStringPageReader reader(base_column->pages[cur_page_idx], cur_page_idx);
            long_str.append(reader.data());
  
            ++cur_page_idx;
            while (cur_page_idx < base_column->pages.size() && 
                   *reinterpret_cast<const uint16_t *>(base_column->pages[cur_page_idx]) == 0xFFFE) {
              LongStringPageReader reader(base_column->pages[cur_page_idx], cur_page_idx);
              long_str.append(reader.data());
              ++cur_page_idx;
            }
            shuffled_strs[shuffled_indices[i].second] = std::string_view(long_str_buffer.back());
          }
        }
      }

      writer.write_values(shuffled_strs.data(), shuffled_strs.size());
      writer.flush();
    };

    for (size_t col_idx = 0; col_idx < result.columns.size(); ++col_idx) {
      switch (result.columns[col_idx].type) {
      case DataType::INT32   : impl_fixed(type_tag<int32_t>, col_idx); break;
      case DataType::INT64   : impl_fixed(type_tag<int64_t>, col_idx); break;
      case DataType::FP64    : impl_fixed(type_tag<double>, col_idx); break;
      case DataType::VARCHAR : impl_varchar(col_idx); break;
      }
    }
    return result;
  }

  void reset() {
    hash_map_pool_.single_value_alloc_id = 0;
    hash_map_pool_.chained_alloc_id = 0;

    managed_data_chunck_alloc_id_.store(0);
    num_hash_tables_ = 0;
    result_partitions_.clear();
    varchar_page_locators_.clear();
    fixed_page_locators_.clear();
  }

private:
  struct HashMapPool {
    std::vector<std::unique_ptr<SingleValueHashMap>> single_value;
    std::vector<std::unique_ptr<ChainedHashMap>>     chained;
    size_t                                           single_value_alloc_id{0};
    size_t                                           chained_alloc_id{0};
  };

  std::array<DataChunck, MAX_MANAGED_DATA_CHUNCKS>             managed_data_chuncks_;
  HashMapPool                                                  hash_map_pool_;
  std::array<HashMap *, MAX_HASH_MAPS>                         hash_maps_;
  std::array<std::unique_ptr<HashSet>, MAX_HASH_MAPS>          hash_sets_;
  std::atomic<size_t>                                          managed_data_chunck_alloc_id_{0};
  size_t                                                       num_hash_tables_{0};
  std::mutex                                                   result_partitions_mtx_;
  std::deque<DataChunck>                                       result_partitions_;
  std::unordered_map<const Column *, VarcharColumnPageLocator> varchar_page_locators_;
  std::unordered_map<const Column *, FixedColumnPageLocator>   fixed_page_locators_;
};

}