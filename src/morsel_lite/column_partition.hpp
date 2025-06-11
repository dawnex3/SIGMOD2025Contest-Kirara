#pragma once

#include <bitset>
#include <cstdlib>
#include <vector>

#include "range/v3/view/transform.hpp"
#include "range/v3/view/iota.hpp"

#include "plan.h"

#include "page.hpp"
#include "utils.hpp"

namespace Contest::MorselLite {

class ExecutionContext;

class ColumnPartitionDesc {
public:
  ColumnPartitionDesc()
    : store_(0, 0) { }

  ColumnPartitionDesc(const ColumnPartitionDesc &) = default;
  ColumnPartitionDesc(ColumnPartitionDesc &&) = default;

  ColumnPartitionDesc &operator=(const ColumnPartitionDesc &) = default;

  [[nodiscard]] static std::vector<std::vector<ColumnPartitionDesc>>
  from_table(const ColumnarTable &table, size_t partition_size,
             const std::unordered_map<const Column *, FixedColumnPageLocator> &fixed_col_page_locator_lut,
             const std::unordered_map<const Column *, VarcharColumnPageLocator> &varchar_col_page_locator_lut,
             const std::bitset<64> &col_masks) {
    const size_t num_partitions = idiv_ceil(table.num_rows, partition_size);
    const size_t num_columns = table.columns.size();

    std::vector<std::vector<ColumnPartitionDesc>> result(num_partitions, 
      std::vector<ColumnPartitionDesc>(num_columns, ColumnPartitionDesc(0, 0)));

    auto make_col_out_view = [&](size_t col_index) {
      return ranges::views::transform(
        ranges::views::iota(static_cast<size_t>(0), num_partitions), 
        [col_index, &result](size_t i) -> ColumnPartitionDesc & {
          return result[i][col_index];
        });
    };

    for (size_t i = 0; i < num_columns; ++i) {
      if (!col_masks[i]) { continue; }

      auto out_view = make_col_out_view(i);
      const auto &column = table.columns[i];
      if (column.type != DataType::VARCHAR) {
        const auto &locator = fixed_col_page_locator_lut.at(std::addressof(column));
        from_fixed_column(column, locator, table.num_rows, partition_size, out_view.begin());
      } else {
        from_varchar_column(column, table.num_rows, partition_size, out_view.begin());
      }
    }
    return result;
  }

  template<typename T, typename OutputIterT>
  OutputIterT materialize(const Column &base_column, OutputIterT out_first) const {
    if constexpr (std::is_same_v<T, std::string>) {
      ML_ASSERT(base_column.type == DataType::VARCHAR);
      size_t row_id_offset = store_.for_varchar.first_row_id_;
      size_t num_rows = store_.for_varchar.num_rows_;
      for (size_t i = row_id_offset; i < row_id_offset + num_rows; ++i) {
        *out_first++ = i;
      }
    } else {
      const auto &info = store_.for_fixed;

      if (info.beg.page_index() == info.last_element_page_index) {
        FixedPageReader<T> reader(base_column.pages[info.beg.page_index()], info.beg.page_index());
        size_t num_rows = info.last_element_offset_in_page + 1 - info.beg.index();
        reader.read_values(info.beg, num_rows, out_first);
        return out_first;
      }

      // handle first page
      {
        FixedPageReader<T> reader(base_column.pages[info.beg.page_index()], info.beg.page_index());
        out_first = reader.read_values(info.beg, out_first);
      }

      for (size_t page_idx = info.beg.page_index() + 1; page_idx < info.last_element_page_index; ++page_idx) {
        FixedPageReader<T> reader(base_column.pages[page_idx], page_idx);
        out_first = reader.read_values(out_first);
      }

      // handle last page
      {
        FixedPageReader<T> reader(base_column.pages[info.last_element_page_index], info.last_element_page_index);
        out_first = reader.read_values(info.last_element_offset_in_page + 1, out_first);
      }
    }
    return out_first;
  }

private:
  union Store {
    struct ForFixed {
      PageDataLocation beg;
      uint32_t         last_element_page_index;
      uint32_t         last_element_offset_in_page;

      ForFixed(PageDataLocation beg, uint32_t last_element_page_index, uint32_t last_element_offset_in_page)
        : beg(beg), last_element_page_index(last_element_page_index), 
          last_element_offset_in_page(last_element_offset_in_page) { }

    } for_fixed;

    struct ForVarchar {
      size_t first_row_id_;
      size_t num_rows_;

      ForVarchar(size_t first_row_id, size_t num_rows)
        : first_row_id_(first_row_id), num_rows_(num_rows) { }
    } for_varchar;

    Store(PageDataLocation beg, uint32_t last_element_page_index, uint32_t last_element_page_offset) noexcept
      : for_fixed(beg, last_element_page_index, last_element_page_offset) { }

    Store(size_t first_row_id, size_t num) noexcept
      : for_varchar(first_row_id, num) { }

  } store_;

  ColumnPartitionDesc(size_t first_row_id, size_t num) noexcept
    : store_(first_row_id, num) { }
  
  ColumnPartitionDesc(PageDataLocation beg, uint32_t last_page_index, uint32_t last_page_offset) noexcept
    : store_(beg, last_page_index, last_page_offset) { }

  template<typename OutputIterT>
  static OutputIterT
  from_fixed_column(const Column &column, const FixedColumnPageLocator &locator, 
                    size_t num_rows, size_t partition_size, OutputIterT out_first) {
    const size_t num_full_partitions = num_rows / partition_size,
                 last_partition_size = num_rows % partition_size;

    size_t search_hint = 0;
    for (size_t i = 0; i < num_full_partitions; ++i) {
      const size_t start_idx = i * partition_size;
      auto [beg_loc, _1] = locator.get_location(start_idx, search_hint);
      auto [end_loc, _2] = locator.get_location((i + 1) * partition_size);
      *out_first++ = ColumnPartitionDesc(beg_loc, end_loc.page_index(), end_loc.index());
    }

    if (last_partition_size > 0) {
      auto [beg_loc, new_search_hint] = locator.get_location(num_full_partitions * partition_size, search_hint);
      size_t last_page_num_rows = *reinterpret_cast<const uint16_t *>(column.pages.back()->data);
      *out_first++ = ColumnPartitionDesc(beg_loc, column.pages.size() - 1, last_page_num_rows - 1);
    }
    return out_first;
  }

  template<typename OutputIterT>
  static OutputIterT
  from_varchar_column(const Column &column, size_t num_total_rows, size_t partition_size, OutputIterT out_first) {
    const size_t num_full_partitions = num_total_rows / partition_size,
                 last_partition_size = num_total_rows % partition_size;

    for (size_t i = 0; i < num_full_partitions; ++i) {
      *out_first++ = ColumnPartitionDesc(i * partition_size, partition_size);
    }

    if (last_partition_size != 0) {
      size_t last_start = num_full_partitions * partition_size;
      *out_first++ = ColumnPartitionDesc(last_start, last_partition_size);
    }

    return out_first;
  }
};
}