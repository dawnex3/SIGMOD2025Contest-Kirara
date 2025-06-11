#pragma once

#include <algorithm>

#include "range/v3/numeric/accumulate.hpp"

#include "plan.h"

#include "buffer.hpp"
#include "defs.hpp"
#include "utils.hpp"

namespace Contest::MorselLite {

class FlatColumnVector {
  static constexpr size_t DEFAULT_ALIGNMENT = 32;
public:
  FlatColumnVector() = default;

  FlatColumnVector(const FlatColumnVector &) = delete;
  
  FlatColumnVector(FlatColumnVector &&other) noexcept
    : data_type_(std::exchange(other.data_type_, INVALID_DATA_TYPE)), buf_(std::move(other.buf_)) { }

  FlatColumnVector(DataType data_type, Buffer<uint8_t> &&buf) noexcept
    : data_type_(data_type), buf_(std::move(buf)) {
    ML_ASSERT(data_type != DataType::VARCHAR);
  }

  FlatColumnVector(DataType data_type) noexcept
    : data_type_(data_type) {
    ML_ASSERT(data_type != DataType::VARCHAR);
  }

  FlatColumnVector &operator=(const FlatColumnVector &) = delete;

  FlatColumnVector &operator=(FlatColumnVector &&other) noexcept {
    data_type_ = std::exchange(other.data_type_, INVALID_DATA_TYPE);
    buf_ = std::move(other.buf_);
    return *this;
  }

  [[nodiscard]] DataType data_type() const { return data_type_; }
  [[nodiscard]] size_t num_rows() const { return buf_.size() / element_size(); }

  [[nodiscard]] Span<const int32_t> int32_view() const {
    ML_ASSERT(check_type(DataType::INT32));
    return buf_.view_as<int32_t>();
  }

  [[nodiscard]] Span<int32_t> int32_view() {
    ML_ASSERT(check_type(DataType::INT32));
    return buf_.mut_view_as<int32_t>();
  }

  [[nodiscard]] Span<const int64_t> int64_view() const {
    ML_ASSERT(check_type(DataType::INT64));
    return buf_.view_as<int64_t>();
  }

  [[nodiscard]] Span<int64_t> int64_view() {
    ML_ASSERT(check_type(DataType::INT64));
    return buf_.mut_view_as<int64_t>();
  }

  [[nodiscard]] Span<const double> float64_view() const {
    ML_ASSERT(check_type(DataType::FP64));
    return buf_.view_as<double>();
  }

  [[nodiscard]] Span<double> float64_view() {
    ML_ASSERT(check_type(DataType::FP64));
    return buf_.mut_view_as<double>();
  }

  template<typename T>
  [[nodiscard]] Span<const T> view_as() const {
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T, int32_t>) {
      ML_ASSERT(check_type(DataType::INT32));
    } else if constexpr (std::is_same_v<T, int64_t>) {
      ML_ASSERT(check_type(DataType::INT64));
    } else {
      ML_ASSERT(check_type(DataType::FP64));
    }

    return buf_.view_as<T>();
  }

  template<typename T>
  [[nodiscard]] Span<T> view_as() {
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T, int32_t>) {
      ML_ASSERT(check_type(DataType::INT32));
    } else if constexpr (std::is_same_v<T, int64_t>) {
      ML_ASSERT(check_type(DataType::INT64));
    } else {
      ML_ASSERT(check_type(DataType::FP64));
    }

    return buf_.mut_view_as<T>();
  }

  FlatColumnVector &resize(size_t new_num_rows) {
    buf_.resize(new_num_rows * element_size());
    return *this;
  }

  FlatColumnVector &apply_factors(Span<const size_t> factors) {
    ML_ASSERT(factors.size() == num_rows());

    const size_t new_num_rows = ranges::accumulate(factors, 0, std::plus<>{});
    Buffer<uint8_t> new_buf(element_size() * new_num_rows);

    auto impl = [&](auto _ty) {
      using T = ML_TYPE_OF_TAG(_ty);
      auto in = buf_.view_as<T>();
      auto out = new_buf.mut_view_as<T>();
      for (size_t i = 0, j = 0; i < factors.size(); ++i) {
        std::uninitialized_fill_n(out.data() + j, factors[i], in[i]);
        j += factors[i];
      }
    };

    switch (data_type_) {
    default              : unreachable_branch();
    case DataType::INT32 : impl(type_tag<int32_t>); break;
    case DataType::INT64 : impl(type_tag<int64_t>); break;
    case DataType::FP64  : impl(type_tag<double>); break;
    }
    buf_ = std::move(new_buf);
    return *this;
  }

  FlatColumnVector &select(Span<const size_t> selections) {
    ML_ASSERT(validate_selection_indices(selections, num_rows()));

    const size_t new_num_rows = selections.size();
    Buffer<uint8_t> new_buf(element_size() * new_num_rows);

    auto impl = [&](auto _ty) {
      using T = ML_TYPE_OF_TAG(_ty);
      auto in = buf_.view_as<T>();
      auto out = new_buf.mut_view_as<T>();
      for (size_t i = 0; i < selections.size(); ++i) {
        out[i] = in[selections[i]];
      }
    };

    switch (data_type_) {
    default              : unreachable_branch();
    case DataType::INT32 : impl(type_tag<int32_t>); break;
    case DataType::INT64 : impl(type_tag<int64_t>); break;
    case DataType::FP64  : impl(type_tag<double>); break;
    }
    buf_ = std::move(new_buf);
    return *this;
  }

  FlatColumnVector &select_and_apply_factors(Span<const size_t> selections, Span<const size_t> factors) {
    ML_ASSERT(selections.size() == factors.size());
    ML_ASSERT(validate_selection_indices(selections, num_rows()));

    const size_t new_num_rows = ranges::accumulate(factors, 0, std::plus<>{});
    Buffer<uint8_t> new_buf(element_size() * new_num_rows);

    auto impl = [&](auto _ty) {
      using T = ML_TYPE_OF_TAG(_ty);
      auto in = buf_.view_as<T>();
      auto out = new_buf.mut_view_as<T>();
      for (size_t i = 0, j = 0; i < factors.size(); ++i) {
        std::uninitialized_fill_n(out.data() + j, factors[i], in[selections[i]]);
        j += factors[i];
      }
    };

    switch (data_type_) {
    default              : unreachable_branch();
    case DataType::INT32 : impl(type_tag<int32_t>); break;
    case DataType::INT64 : impl(type_tag<int64_t>); break;
    case DataType::FP64  : impl(type_tag<double>); break;
    }
    buf_ = std::move(new_buf);

    return *this;
  }

  [[nodiscard]] FlatColumnVector clone() const {
    Buffer<uint8_t> buf(buf_.size());
    std::copy_n(buf_.data(), buf_.size(), buf.data());
    return FlatColumnVector(data_type_, std::move(buf));
  }

  void swap(FlatColumnVector &other) noexcept {
    std::swap(data_type_, other.data_type_);
    buf_.swap(other.buf_);
  }

private:
  DataType        data_type_{INVALID_DATA_TYPE};
  Buffer<uint8_t> buf_;

  [[nodiscard]] bool check_type(DataType expected) const { return data_type_ == expected; }
  [[nodiscard]] size_t element_size() const noexcept { return data_type_ == DataType::INT32 ? 4 : 8; }
};

class RowIdColumnVector {
  static constexpr size_t DEFAULT_ALIGNMENT = 32;
public:
  RowIdColumnVector() = default;

  RowIdColumnVector(const RowIdColumnVector &) = delete;

  RowIdColumnVector(RowIdColumnVector &&other) noexcept
    : column_(std::exchange(other.column_, nullptr)),
      buf_(std::move(other.buf_)) { }

  RowIdColumnVector(const Column &column, Buffer<size_t> &&buf) noexcept
    : column_(std::addressof(column)), buf_(std::move(buf)) { }
  
  RowIdColumnVector(const Column &column) noexcept
    : column_(std::addressof(column)) { }
  
  RowIdColumnVector &operator=(const RowIdColumnVector &) = delete;

  RowIdColumnVector &operator=(RowIdColumnVector &&other) noexcept {
    column_ = std::exchange(other.column_, nullptr);
    buf_ = std::move(other.buf_);
    return *this;
  }
  
  [[nodiscard]] Span<const size_t> view() const { return buf_.view(); }
  [[nodiscard]] Span<size_t> view() { return buf_.mut_view(); }

  [[nodiscard]] DataType data_type() const { return column_->type; }
  [[nodiscard]] size_t num_rows() const { return buf_.size(); }

  [[nodiscard]] const Column *column() const noexcept { return column_; }
  void set_column(const Column *col) { column_ = col; }

  RowIdColumnVector &resize(size_t new_num_rows) {
    buf_.resize(new_num_rows);
    return *this;
  }

  RowIdColumnVector &apply_factors(Span<const size_t> factors) {
    ML_ASSERT(factors.size() == num_rows());
    
    const size_t new_num_rows = ranges::accumulate(factors, 0, std::plus<>{});
    Buffer<size_t> new_buf(new_num_rows);
    for (size_t i = 0, j = 0; i < factors.size(); ++i) {
      std::uninitialized_fill_n(new_buf.data() + j, factors[i], buf_[i]);
      j += factors[i];
    }

    buf_ = std::move(new_buf);
    return *this;
  }

  RowIdColumnVector &select(Span<const size_t> selections) {
    ML_ASSERT(validate_selection_indices(selections, num_rows()));

    const size_t new_num_rows = selections.size();
    Buffer<size_t> new_buf(new_num_rows);
    for (size_t i = 0; i < selections.size(); ++i) {
      new_buf[i] = buf_[selections[i]];
    }
    buf_ = std::move(new_buf);
    return *this;
  }

  RowIdColumnVector &select_and_apply_factors(Span<const size_t> selections, Span<const size_t> factors) {
    ML_ASSERT(selections.size() == factors.size());
    ML_ASSERT(validate_selection_indices(selections, num_rows()));

    const size_t new_num_rows = ranges::accumulate(factors, 0, std::plus<>{});
    Buffer<size_t> new_buf(new_num_rows);
    for (size_t i = 0, j = 0; i < factors.size(); ++i) {
      std::uninitialized_fill_n(new_buf.data() + j, factors[i], buf_[selections[i]]);
      j += factors[i];
    }

    buf_ = std::move(new_buf);
    return *this;
  }

  [[nodiscard]] RowIdColumnVector clone() const {
    Buffer<size_t> buf(buf_.size());
    std::copy_n(buf_.data(), buf_.size(), buf.data());
    return RowIdColumnVector(*column_, std::move(buf));
  }

  void swap(RowIdColumnVector &other) noexcept {
    std::swap(column_, other.column_);
    buf_.swap(other.buf_);
  }

private:
  const Column   *column_{nullptr};
  Buffer<size_t>  buf_;
};


class ColumnVector {
  static constexpr size_t FLAT_VECTOR_TYPE_ID = 1;
  static constexpr size_t ROWID_VECTOR_TYPE_ID = 2;

  template<typename Fn>
  decltype(auto) dispatch_call(Fn &&fn) const {
    ML_ASSERT(payload_.index() != 0);

    switch (payload_.index()) {
    default : unreachable_branch();
    case FLAT_VECTOR_TYPE_ID : return std::invoke(std::forward<Fn>(fn), std::get<FLAT_VECTOR_TYPE_ID>(payload_));
    case ROWID_VECTOR_TYPE_ID : return std::invoke(std::forward<Fn>(fn), std::get<ROWID_VECTOR_TYPE_ID>(payload_));
    }
  }

  template<typename Fn>
  decltype(auto) dispatch_call(Fn &&fn) {
    ML_ASSERT(payload_.index() != 0);

    switch (payload_.index()) {
    default : unreachable_branch();
    case FLAT_VECTOR_TYPE_ID : return std::invoke(std::forward<Fn>(fn), std::get<FLAT_VECTOR_TYPE_ID>(payload_));
    case ROWID_VECTOR_TYPE_ID : return std::invoke(std::forward<Fn>(fn), std::get<ROWID_VECTOR_TYPE_ID>(payload_));
    }
  }

public:
  ColumnVector()
    : payload_(std::monostate{}) { }
  
  ColumnVector(const ColumnVector &) = delete;
  
  ColumnVector(ColumnVector &&other) noexcept
    : payload_(std::move(other.payload_)) { }
  
  ColumnVector(FlatColumnVector &&vec) noexcept
    : payload_(std::in_place_index<1>, std::move(vec)) { }

  ColumnVector(RowIdColumnVector &&vec) noexcept
    : payload_(std::in_place_index<2>, std::move(vec)) { }
  
  ColumnVector &operator=(const ColumnVector &) = delete;

  ColumnVector &operator=(ColumnVector &&other) {
    payload_ = std::move(other.payload_);
    return *this;
  }

  [[nodiscard]] bool is_uninit() const { return payload_.index() == 0; }
  [[nodiscard]] bool is_flat_vector() const { return payload_.index() == FLAT_VECTOR_TYPE_ID; }
  [[nodiscard]] bool is_row_id_vector() const { return payload_.index() == ROWID_VECTOR_TYPE_ID; }

  [[nodiscard]] const FlatColumnVector &as_flat_vector() const {
    const auto *ptr = std::get_if<FLAT_VECTOR_TYPE_ID>(&payload_);
    ML_ASSERT(ptr != nullptr);
    return *ptr;
  }

  [[nodiscard]] FlatColumnVector &as_flat_vector() {
    auto *ptr = std::get_if<FLAT_VECTOR_TYPE_ID>(&payload_);
    ML_ASSERT(ptr != nullptr);
    return *ptr;
  }

  [[nodiscard]] const RowIdColumnVector &as_rowid_vector() const {
    const auto *ptr = std::get_if<ROWID_VECTOR_TYPE_ID>(&payload_);
    ML_ASSERT(ptr != nullptr);
    return *ptr;
  }

  [[nodiscard]] RowIdColumnVector &as_rowid_vector() {
    auto *ptr = std::get_if<ROWID_VECTOR_TYPE_ID>(&payload_);
    ML_ASSERT(ptr != nullptr);
    return *ptr;
  }

  [[nodiscard]] DataType data_type() const {
    return dispatch_call([](const auto &x) { return x.data_type(); });
  }

  [[nodiscard]] size_t num_rows() const {
    return dispatch_call([](const auto &x) { return x.num_rows(); });
  }

  ColumnVector &resize(size_t new_num_rows) {
    dispatch_call([new_num_rows](auto &x) { x.resize(new_num_rows); });
    return *this;
  }

  ColumnVector &apply_factors(Span<const size_t> factors) {
    dispatch_call([&factors](auto &x) { x.apply_factors(factors); });
    return *this;
  }

  ColumnVector &select(Span<const size_t> indices) {
    dispatch_call([&indices](auto &x) { x.select(indices); });
    return *this;
  }

  ColumnVector &select_and_apply_factors(Span<const size_t> selections, Span<const size_t> factors) {
    dispatch_call([&selections, &factors](auto &x) { x.select_and_apply_factors(selections, factors); });
    return *this;
  }

  [[nodiscard]] ColumnVector clone() const {
    ColumnVector result;
    if (payload_.index() == FLAT_VECTOR_TYPE_ID) {
      result = std::get<FLAT_VECTOR_TYPE_ID>(payload_).clone();
    } else if (payload_.index() == ROWID_VECTOR_TYPE_ID) {
      result = std::get<ROWID_VECTOR_TYPE_ID>(payload_).clone();
    } else {
      return ColumnVector();
    }
    return result;
  }

private:
  std::variant<std::monostate, FlatColumnVector, RowIdColumnVector> payload_;
};

}