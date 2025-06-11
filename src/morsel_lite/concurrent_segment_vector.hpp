#pragma once

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include "buffer.hpp"
#include "utils.hpp"

namespace Contest::MorselLite {

template<typename T, typename AllocT = std::allocator<T>>
class ConcurrentSegmentVector {
  static_assert(std::is_trivial_v<T> && std::is_trivially_copyable_v<T>);
  
  struct Segment {
    static constexpr size_t SIZE_IN_BYTES = 4096 * 8;
    static_assert(SIZE_IN_BYTES % sizeof(T) == 0);
    static constexpr size_t NUM_ELEMENTS = SIZE_IN_BYTES / sizeof(T);
    T payload[NUM_ELEMENTS];
  };

public:
  using value_type             = T;
  using size_type              = size_t;
  using difference_type        = ptrdiff_t;
  using const_reference        = const value_type &;
  using reference              = value_type &;
  using const_pointer          = const value_type *;
  using pointer                = value_type *;
  using allocator_type         = AllocT;
  using segment_allocator_type = typename std::allocator_traits<AllocT>::template rebind_alloc<Segment>;

public:
  ConcurrentSegmentVector() {
    segments_.resize(16);
    segments_.fill(nullptr);
    unsafe_ensure_segments(0, 4);
  }

  ConcurrentSegmentVector(const ConcurrentSegmentVector &) = delete;

  ConcurrentSegmentVector(ConcurrentSegmentVector &&other) noexcept
    : segment_lut_mtx_(std::move(other.segment_lut_mtx_)),
      segments_(std::move(other.segments_)),
      size_(std::move(other.size_)) { }

  ConcurrentSegmentVector &operator=(const ConcurrentSegmentVector &) = delete;

  ~ConcurrentSegmentVector() {
    for (size_t i = 0; i < segments_.size(); ++i) {
      if (segments_[i] != nullptr) {
        std::allocator_traits<segment_allocator_type>::deallocate(segment_alloc_, segments_[i], 1);
      }
    }
  }

  size_type push_back(const value_type &val) {
    size_type index = size_.fetch_add(1);

    const size_type segment_index = index / Segment::NUM_ELEMENTS,
                    segment_offset = index % Segment::NUM_ELEMENTS;
    
    Segment *segment = ensure_segment(segment_index);
    segment->payload[segment_offset] = val;
    return index;
  }

  template<typename InputIterT>
  size_type append(InputIterT first, size_type num) {
    size_type index = size_.fetch_add(num, std::memory_order_relaxed);

    const size_type first_segment_index = index / Segment::NUM_ELEMENTS,
                    first_segment_offset = index % Segment::NUM_ELEMENTS;
    const size_type last_segment_index = (index + num - 1) / Segment::NUM_ELEMENTS,
                    last_segment_offset = (index + num - 1) % Segment::NUM_ELEMENTS;

    auto required_segments = ensure_segments(first_segment_index, last_segment_index - first_segment_index + 1);

    if (required_segments.size() == 1) {
      std::copy_n(first, num, required_segments[0]->payload + first_segment_offset);
    } else {
      std::copy_n(first, Segment::NUM_ELEMENTS - first_segment_offset, 
                  required_segments.front()->payload + first_segment_offset);
      first += Segment::NUM_ELEMENTS - first_segment_offset;
      for (size_type i = 1; i < required_segments.size() - 1; ++i) {
        std::copy_n(first, Segment::NUM_ELEMENTS, required_segments[i]->payload);
        first += Segment::NUM_ELEMENTS;
      }
      std::copy_n(first, last_segment_offset + 1, required_segments.back()->payload);
    }
    return index;
  }

  void clear() {
    size_.store(0);
  }

  [[nodiscard]] size_type size() const noexcept {
    return size_.load();
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  [[nodiscard]] const_reference at(size_type index) const {
    ML_ASSERT(index < size_.load());
    const size_type segment_index = index / Segment::NUM_ELEMENTS,
                    segment_offset = index % Segment::NUM_ELEMENTS;
    return segments_[segment_index]->payload[segment_offset];
  }
  
  [[nodiscard]] reference at(size_type index) {
    ML_ASSERT(index < size_.load());
    const size_type segment_index = index / Segment::NUM_ELEMENTS,
                    segment_offset = index % Segment::NUM_ELEMENTS;
    return segments_[segment_index]->payload[segment_offset];
  }

  [[nodiscard]] const_reference operator[](size_type index) const { return at(index); }

  [[nodiscard]] reference operator[](size_type index) { return at(index); }

  [[nodiscard]] Span<value_type> segment_mut_view(size_type index) {
    ML_ASSERT(index < segments_.size());
    return Span<value_type>{segments_[index]->payload, Segment::NUM_ELEMENTS};
  }

  [[nodiscard]] Span<const value_type> segment_view(size_type index) const {
    ML_ASSERT(index < segments_.size());
    return Span<const value_type>{segments_[index]->payload, Segment::NUM_ELEMENTS};
  }

  [[nodiscard]] std::vector<value_type> to_vector() const {
    if (empty()) {
      return std::vector<value_type>{};
    }

    std::vector<value_type> res(size_.load());

    const size_t num_full_used_segments = res.size() / Segment::NUM_ELEMENTS,
                 partial_used_segment_size = res.size() % Segment::NUM_ELEMENTS;

    for (size_t i = 0; i < num_full_used_segments; ++i) {
      std::copy_n(segments_[i]->payload, Segment::NUM_ELEMENTS, res.data() + i * Segment::NUM_ELEMENTS);
    }

    if (partial_used_segment_size != 0) {
      std::copy_n(segments_[num_full_used_segments]->payload, 
                  partial_used_segment_size, res.data() + num_full_used_segments * Segment::NUM_ELEMENTS);
    }
    return res;
  }

private:
  segment_allocator_type segment_alloc_;
  std::mutex             segment_lut_mtx_;
  Buffer<Segment *>      segments_;
  alignas(64) std::atomic<size_type> size_{0};

  [[nodiscard]] Segment *ensure_segment(size_type segment_index) {
    std::unique_lock<std::mutex> lk{segment_lut_mtx_};
    if (segment_index < segments_.size()) {
      Segment *segment = segments_[segment_index];
      if (!segment) {
        segments_[segment_index] = std::allocator_traits<segment_allocator_type>::allocate(segment_alloc_, 1);
        return segments_[segment_index];
      }
      return segment;
    }

    segments_.resize(segments_.size() * 2, nullptr);
    segments_[segment_index] = std::allocator_traits<segment_allocator_type>::allocate(segment_alloc_, 1);
    return segments_[segment_index];
  }

  struct SegmentList {
    using SegmentPtr = Segment *;
    static constexpr size_t SMALL_BUF_SIZE = 16;
    
    SegmentList(size_type num) {
      if (num <= SMALL_BUF_SIZE) {
        data_ = small_buf_.data();
      } else {
        data_ = new SegmentPtr[num];
      }
      size_ = num;
    }

    ~SegmentList() {
      if (data_ != small_buf_.data()) {
        delete[] data_;
      }
    }

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] SegmentPtr &front() noexcept { return data_[0]; }
    [[nodiscard]] const SegmentPtr &front() const noexcept { return data_[0]; }
    [[nodiscard]] SegmentPtr &back() noexcept { return data_[size_ - 1]; }
    [[nodiscard]] const SegmentPtr &back() const noexcept { return data_[size_ - 1]; }
    [[nodiscard]] SegmentPtr &operator[](size_type index) noexcept { return data_[index]; }
    [[nodiscard]] const SegmentPtr &operator[](size_type index) const noexcept { return data_[index]; }
  
  private:
    std::array<SegmentPtr, SMALL_BUF_SIZE>  small_buf_;
    SegmentPtr                             *data_{nullptr};
    size_type                               size_{0};
  };

  void unsafe_ensure_segments(size_type first_index, size_type num) {
    auto alloc_if_null = [&]() {
      for (size_t i = first_index; i < first_index + num; ++i) {
        if (!segments_[i]) {
          segments_[i] = std::allocator_traits<segment_allocator_type>::allocate(segment_alloc_, 1);
        }
      }
    };

    if (first_index + num <= segments_.size()) {
      alloc_if_null();
      return;
    }

    segments_.resize(std::max(first_index + num, segments_.size() * 2), nullptr);
    alloc_if_null();
  }

  [[nodiscard]] std::vector<Segment *> ensure_segments(size_type first_index, size_type num) {
    ML_ASSERT(num != 0);
    std::unique_lock<std::mutex> lk{segment_lut_mtx_};

    auto collect_segments = [&]() {
      std::vector<Segment *> result(num);

      for (size_t i = first_index; i < first_index + num; ++i) {
        if (segments_[i] != nullptr) {
          result[i - first_index] = segments_[i];
        } else {
          segments_[i] = std::allocator_traits<segment_allocator_type>::allocate(segment_alloc_, 1);
          result[i - first_index] = segments_[i];
        }
      }
      return result;
    };

    if (first_index + num - 1 < segments_.size()) {
      return collect_segments();
    }

    segments_.resize(std::max(first_index + num, segments_.size() * 2), nullptr);
    return collect_segments();
  }
};

}