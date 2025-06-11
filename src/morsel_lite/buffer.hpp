#pragma once

#include <memory>
#include <utility>

#include "utils.hpp"

namespace Contest::MorselLite {

template<typename T>
class Buffer {
public:
  Buffer() = default;

  Buffer(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) { }

  Buffer(size_t init_size)
    : size_(init_size) {
    data_ = allocate(init_size);
  }
  
  ~Buffer() {
    if (data_) {
      deallocate(data_);
    }
  }

  Buffer &operator=(const Buffer &) = delete;

  Buffer &operator=(Buffer &&other) noexcept {
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    return *this;
  }

  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] const T &operator[](size_t index) const noexcept {
    ML_ASSERT(index < size_);
    return data_[index];
  }

  [[nodiscard]] T &operator[](size_t index) noexcept {
    ML_ASSERT(index < size_);
    return data_[index];
  }

  [[nodiscard]] const T &front() const noexcept {
    ML_ASSERT(!empty());
    return data_[0];
  }

  [[nodiscard]] T &front() noexcept {
    ML_ASSERT(!empty());
    return data_[0];
  }

  [[nodiscard]] const T &back() const noexcept {
    ML_ASSERT(!empty());
    return data_[size_ - 1];
  }

  [[nodiscard]] T &back() noexcept {
    ML_ASSERT(!empty());
    return data_[size_ - 1];
  }

  [[nodiscard]] T *begin() noexcept { return data_; }
  [[nodiscard]] const T *begin() const noexcept { return cbegin(); }
  [[nodiscard]] const T *cbegin() const noexcept { return data_; }
  [[nodiscard]] T *end() noexcept { return data_ + size_; }
  [[nodiscard]] const T *end() const noexcept { return cend() + size_; }
  [[nodiscard]] const T *cend() const noexcept { return data_ + size_; }

  [[nodiscard]] T *data() noexcept { return data_; }
  [[nodiscard]] const T *data() const noexcept { return data_; }

  [[nodiscard]] Span<T> mut_view() noexcept { return Span<T>{data_, size_}; }
  [[nodiscard]] Span<const T> view() const noexcept { return Span<const T>{data_, size_}; }

  template<typename U>
  [[nodiscard]] std::enable_if_t<std::is_same_v<T, uint8_t>, Span<U>>
  mut_view_as() noexcept {
    return Span<U>(reinterpret_cast<U *>(data_), size_ / (sizeof(U) / sizeof(T)));
  }

  template<typename U>
  [[nodiscard]] std::enable_if_t<std::is_same_v<T, uint8_t>, Span<const U>>
  view_as() const noexcept {
    return Span<const U>(reinterpret_cast<const U *>(data_), size_ / (sizeof(U) / sizeof(T)));
  }

  void resize(size_t new_size) {
    if (size_ == new_size) { return; }

    T *new_data = allocate(new_size);
    std::uninitialized_copy_n(data_, std::min(size_, new_size), new_data);

    deallocate(data_);
    data_ = new_data;
    size_ = new_size;
  }

  void resize(size_t new_size, const T &fill_val) {
    if (size_ == new_size) { return; }

    T *new_data = allocate(new_size);

    std::uninitialized_copy_n(data_, std::min(size_, new_size), new_data);
    if (new_size > size_) {
      std::uninitialized_fill_n(new_data + size_, new_size - size_, fill_val);
    }

    deallocate(data_);
    data_ = new_data;
    size_ = new_size;
  }

  void fill(const T &val) {
    std::fill_n(data_, size_, val);
  }

  [[nodiscard]] std::pair<T *, size_t> release_ptr() && noexcept {
    return {std::exchange(data_, nullptr), std::exchange(size_, 0)};
  }

  void swap(Buffer &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

private:
  T      *data_{nullptr};
  size_t  size_{0};

  [[nodiscard]] static T *allocate(size_t num) {
    auto ceil_32 = [](size_t x) {
      return (x + 31) & ~uint32_t(31);
    };

    return reinterpret_cast<T *>(::operator new(ceil_32(num * sizeof(T)), std::align_val_t{32}));
  }
  static void deallocate(T *ptr) {
    ::operator delete[](static_cast<void *>(ptr), std::align_val_t{32});
  }
};

}