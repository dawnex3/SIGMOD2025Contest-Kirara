#pragma once

#include "column_vectors.hpp"

namespace Contest::MorselLite {

class DataChunck {
public:
  DataChunck() = default;

  DataChunck(const DataChunck &other) = delete;

  DataChunck(DataChunck &&other) noexcept
    : cols_(std::move(other.cols_)) { }

  DataChunck(std::vector<ColumnVector> &&cols) noexcept
    : cols_(std::move(cols)) { }

  [[nodiscard]] size_t num_columns() const { return cols_.size(); }
  [[nodiscard]] size_t num_rows() const { return cols_.empty() ? 0 : cols_.front().num_rows(); }
  [[nodiscard]] const ColumnVector &column(size_t index) const { return cols_[index]; }
  [[nodiscard]] ColumnVector &column(size_t index) { return cols_[index]; }

  DataChunck &operator=(const DataChunck &) = delete;

  DataChunck &operator=(DataChunck &&other) noexcept {
    cols_ = std::move(other.cols_);
    return *this;
  } 

  void add_column(ColumnVector &&column) {
    ML_ASSERT(validate_column(column));
    cols_.push_back(std::move(column));
  }

  void set_columns(std::vector<ColumnVector> &&columns) {
    cols_ = std::move(columns);
  }

  DataChunck &resize(size_t new_num_rows) {
    for (auto &col : cols_) {
      col.resize(new_num_rows);
    }
    return *this;
  }

  DataChunck &apply_factors(Span<const size_t> factors) {
    for (auto &col : cols_) {
      col.apply_factors(factors);
    }
    return *this;
  }

  DataChunck &select(Span<const size_t> selections) {
    for (auto &col : cols_) {
      col.select(selections);
    }
    return *this;
  }

  DataChunck &select_and_apply_factors(Span<const size_t> selections, Span<const size_t> factors) {
    for (auto &col : cols_) {
      col.select_and_apply_factors(selections, factors);
    }
    return *this;
  }

  DataChunck &project(Span<const size_t> column_indices) {
    ML_ASSERT(column_indices.size() <= cols_.size());

    std::vector<ColumnVector> new_cols(column_indices.size());
    for (size_t i = 0; i < column_indices.size(); ++i) {
      new_cols[i] = std::move(cols_[column_indices[i]]);
    }
    cols_ = std::move(new_cols);
    return *this;
  }

  DataChunck &project_select_and_apply_factors(Span<const size_t> column_indices, Span<const size_t> selections, 
                                               Span<const size_t> factors) {
    project(column_indices);
    return select_and_apply_factors(selections, factors);
  }

private:
  std::vector<ColumnVector> cols_;

  [[nodiscard]] bool validate_column(const ColumnVector &col) const {
    if (cols_.empty()) { return true; }
    return col.num_rows() == cols_.front().num_rows();
  }
};

}