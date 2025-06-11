#pragma once

#include "buffer.hpp"
#include "plan.h"

#include "defs.hpp"
#include "intrinsics.hpp"
#include "utils.hpp"

namespace Contest::MorselLite {

class PageDataLocation {
public:
  PageDataLocation() = default;
  PageDataLocation(const PageDataLocation &) = default;

  explicit PageDataLocation(uint32_t page_index, uint16_t index, uint16_t data_index) noexcept
    : page_index_(page_index), index_(index), data_index_(data_index) { }
  
  explicit PageDataLocation(uint32_t page_index, uint16_t index) noexcept
    : page_index_(page_index), index_(index) { }
  
  PageDataLocation &operator=(const PageDataLocation &) = default;

  [[nodiscard]] uint32_t page_index() const noexcept { return page_index_; }
  [[nodiscard]] uint16_t index() const noexcept { return index_; }
  [[nodiscard]] uint16_t data_index() const noexcept { return data_index_; }

  [[nodiscard]] bool has_data_index() const noexcept { return data_index_ != INVALID_INDEX_OF<uint16_t>; }

  void set_page_index(uint32_t page_index) { page_index_ = page_index; }
  void set_index(uint16_t index) { index_ = index; }
  void set_data_index(uint16_t data_index) { data_index_ = data_index; }

  friend bool operator==(PageDataLocation lhs, PageDataLocation rhs) noexcept {
    return lhs.page_index_ == rhs.page_index_ && lhs.index_ == rhs.index_ &&
           lhs.data_index_ == rhs.data_index_;
  }

  friend bool operator<(PageDataLocation lhs, PageDataLocation rhs) noexcept {
    if (lhs.page_index_ == rhs.page_index_) {
      return lhs.index_ < rhs.index_;
    }
    return lhs.page_index_ < rhs.page_index_;
  }

  friend bool operator>(PageDataLocation lhs, PageDataLocation rhs) noexcept {
    return rhs < lhs;
  }

  friend bool operator<=(PageDataLocation lhs, PageDataLocation rhs) noexcept {
    return !(rhs < lhs);
  }

  friend bool operator>=(PageDataLocation lhs, PageDataLocation rhs) noexcept {
    return !(lhs < rhs);
  }

private:
  uint32_t page_index_{INVALID_INDEX_OF<uint32_t>};
  uint16_t index_{INVALID_INDEX_OF<uint16_t>}; // logical index
  uint16_t data_index_{INVALID_INDEX_OF<uint16_t>}; // physical index in data
};

template<typename T>
struct FixedPageTraits;

template<>
struct FixedPageTraits<int32_t> {
  static constexpr int32_t NULL_VALUE = std::numeric_limits<int32_t>::min();
};

template<>
struct FixedPageTraits<int64_t> {
  static constexpr int64_t NULL_VALUE = std::numeric_limits<int64_t>::min();
};

template<>
struct FixedPageTraits<size_t> {
  static constexpr size_t NULL_VALUE = std::numeric_limits<size_t>::max();
};

template<>
struct FixedPageTraits<double> {
  static constexpr double NULL_VALUE = std::numeric_limits<double>::quiet_NaN();
};

template<typename T>
class FixedPageReader {
public:
  static constexpr T NULL_VALUE = FixedPageTraits<T>::NULL_VALUE;

public:
  FixedPageReader(const Page *page, size_t page_index) noexcept
    : page_(page), page_index_(page_index) {
    num_rows_ = *reinterpret_cast<const uint16_t*>(page_);
  }

  [[nodiscard]] size_t num_rows() const noexcept { return num_rows_; }
  [[nodiscard]] size_t page_index() const noexcept { return page_index_; }

  [[nodiscard]] const uint8_t *notnull_bitmap() const noexcept {
    return reinterpret_cast<const uint8_t*>(page_->data + PAGE_SIZE - (num_rows() + 7) / 8);
  }

  [[nodiscard]] const uint8_t *data() const noexcept {
    constexpr size_t DATA_OFFSET = std::is_same_v<T, int32_t> ? 4 : 8;
    return reinterpret_cast<const uint8_t*>(page_->data + DATA_OFFSET);
  }

  [[nodiscard]] bool is_null(size_t index) const noexcept {
    return !get_bit(notnull_bitmap(), index);
  }

  [[nodiscard]] PageDataLocation first_location() const {
    if (is_null(0)) { return PageDataLocation(page_index_, 0); }
    auto bitmap = notnull_bitmap();
    size_t byte_index = 0;
    while (bitmap[byte_index] == 0) { ++byte_index; }
    for (int i = 0; i < 8; ++i) {
      if (bitmap[byte_index] & (static_cast<uint32_t>(1) << i)) {
        return PageDataLocation(page_index_, 0, byte_index * 8 + i);
      }
    }
    return PageDataLocation{};
  }

  [[nodiscard]] PageDataLocation index_to_location(size_t index) const {
    ML_ASSERT(index < num_rows());
    size_t data_index = rank_bit(index);
    return PageDataLocation(page_index_, index, data_index);
  }

  template<typename OutputIterT>
  OutputIterT read_values(PageDataLocation first, size_t num, OutputIterT out_first) {
    const uint8_t *bitmap = notnull_bitmap();
    const uint8_t *data_reader_ptr = data() + first.data_index() * sizeof(T);
    for (size_t i = first.index(); i < first.index() + num; ++i) {
      if (get_bit(bitmap, i)) {
        *out_first++ = *reinterpret_cast<const T *>(data_reader_ptr);
        data_reader_ptr += sizeof(T);
      } else {
        *out_first++ = NULL_VALUE;
      }
    }
    return out_first;
  }

  template<typename OutputIterT>
  OutputIterT read_values(PageDataLocation first, OutputIterT out_first) {
    const uint8_t *bitmap = notnull_bitmap();
    const uint8_t *data_reader_ptr = data() + first.data_index() * sizeof(T);
    for (size_t i = first.index(); i < num_rows_; ++i) {
      if (get_bit(bitmap, i)) {
        *out_first++ = *reinterpret_cast<const T *>(data_reader_ptr);
        data_reader_ptr += sizeof(T);
      } else {
        *out_first++ = NULL_VALUE;
      }
    }
    return out_first;
  }

  template<typename OutputIterT>
  OutputIterT read_values(OutputIterT out_first) const {
    const uint8_t *bitmap = notnull_bitmap();
    const uint8_t *data_reader_ptr = data();
    for (size_t i = 0; i < num_rows_; ++i) {
      if (get_bit(bitmap, i)) {
        *out_first++ = *reinterpret_cast<const T *>(data_reader_ptr);
        data_reader_ptr += sizeof(T);
      } else {
        *out_first++ = NULL_VALUE;
      }
    }
    return out_first;
  }

  template<typename OutputIterT>
  OutputIterT read_values(size_t num, OutputIterT out_first) const {
    const uint8_t *bitmap = notnull_bitmap();
    const uint8_t *data_reader_ptr = data();
    for (size_t i = 0; i < num; ++i) {
      if (get_bit(bitmap, i)) {
        *out_first++ = *reinterpret_cast<const T *>(data_reader_ptr);
        data_reader_ptr += sizeof(T);
      } else {
        *out_first++ = NULL_VALUE;
      }
    }
    return out_first;
  }

private:
  const Page *page_{nullptr};
  size_t      page_index_{INVALID_INDEX};
  size_t      num_rows_{0};

  // return how many 1 bits before index (the bit in index is excluding)
  [[nodiscard]] size_t rank_bit(size_t index) const noexcept {
    const uint8_t *bitmap = notnull_bitmap();
    const size_t block_index = index / 8,
                 block_offset = index % 8;
    size_t rank = 0;
    for (size_t i = 0; i < block_index; ++i) {
      rank += popcount(bitmap[i]); 
    }
    rank += popcount(bitmap[block_index] & ((1U << block_offset) - 1));
    return rank;
  }
};

class LongStringPageReader {
public:
  LongStringPageReader(const Page *page, size_t page_index)
    : page_(page), page_index_(page_index) {
    uint16_t check_bits = *reinterpret_cast<const uint16_t *>(page->data);
    is_header_ = check_bits == 0xFFFF;
    num_chars = *reinterpret_cast<const uint16_t *>(page->data + 2);
  }

  [[nodiscard]] size_t page_index() const noexcept { return page_index_; }

  [[nodiscard]] bool is_header_page() const { return is_header_; }
  [[nodiscard]] bool is_overflow_page() const { return !is_header_; }

  [[nodiscard]] std::string_view data() const noexcept {
    const char *str_data = reinterpret_cast<const char *>(page_->data + 4);
    return std::string_view{str_data, num_chars};
  }

private:
  const Page *page_{nullptr};
  size_t      page_index_{INVALID_INDEX};
  uint32_t    is_header_ : 1;
  uint32_t    num_chars : 31;
};

class CompactStringPageReader {
public:
  static constexpr std::string_view NULL_VALUE{};
public:
  CompactStringPageReader(const Page *page, size_t page_index) noexcept
    : page_(page), page_index_(page_index) {
    num_rows_ = *reinterpret_cast<const uint16_t*>(page_->data);
  }

  [[nodiscard]] size_t page_index() const noexcept { return page_index_; }
  [[nodiscard]] size_t num_rows() const noexcept { return num_rows_; }

  [[nodiscard]] const uint8_t *notnull_bitmap() const noexcept {
    return reinterpret_cast<const uint8_t*>(page_->data + PAGE_SIZE - (num_rows() + 7) / 8);
  }

  [[nodiscard]] size_t num_notnull() const noexcept {
    return *reinterpret_cast<const uint16_t *>(page_->data + 2);
  }

  [[nodiscard]] const uint16_t *offsets() const noexcept {
    return reinterpret_cast<const uint16_t *>(page_->data + 4);
  }

  [[nodiscard]] const uint8_t *data() const noexcept {
      return reinterpret_cast<const uint8_t *>(page_->data + 4 + num_notnull() * sizeof(uint16_t));
  }

  [[nodiscard]] bool is_null(size_t index) const noexcept {
    return !get_bit(notnull_bitmap(), index);
  }

  [[nodiscard]] PageDataLocation first_location() const {
    if (is_null(0)) { return PageDataLocation(page_index_, 0); }
    auto bitmap = notnull_bitmap();
    size_t byte_index = 0;
    while (bitmap[byte_index] == 0) { ++byte_index; }
    for (int i = 0; i < 8; ++i) {
      if (bitmap[byte_index] & (static_cast<uint32_t>(1) << i)) {
        return PageDataLocation(page_index_, 0, byte_index * 8 + i);
      }
    }
    return PageDataLocation{};
  }

  [[nodiscard]] PageDataLocation index_to_location(size_t index) const noexcept {
    ML_ASSERT(index < num_rows());
    if (is_null(index)) {
      return PageDataLocation(page_index_, index);
    }
    size_t data_index = rank_bit(index);
    return PageDataLocation(page_index_, index, data_index);
  }

  [[nodiscard]] std::string_view read_str(PageDataLocation loc) const {
    if (!loc.has_data_index()) {
      return NULL_VALUE;
    }
    const char *str_data = reinterpret_cast<const char *>(data());
    size_t end_pos = offsets()[loc.data_index()];
    size_t start_pos = loc.data_index() == 0 ? 0 : offsets()[loc.data_index() - 1];
    return std::string_view{str_data + start_pos, end_pos - start_pos};
  }

  template<typename OutputIterT>
  OutputIterT read_values(OutputIterT out_first) const {
    const char *str_data = reinterpret_cast<const char *>(data());
    const auto *bitmap = notnull_bitmap();

    size_t data_index = 0;
    const char *str_begin = str_data;

    for (size_t i = 0; i < num_rows(); ++i) {
      if (get_bit(bitmap, i)) {
        size_t end_pos = offsets()[data_index++];
        *out_first++ = std::string_view(str_begin, str_data + end_pos - str_begin);
        str_begin = str_data + end_pos;
      } else {
        *out_first++ = std::string_view{};
      }
    }
    return out_first;
  }

private:
  const Page *page_{nullptr};
  size_t      page_index_{INVALID_INDEX};
  size_t      num_rows_;

  // return how many 1 bits before index (the bit in index is excluding)
  [[nodiscard]] size_t rank_bit(size_t index) const noexcept {
    const uint8_t *bitmap = notnull_bitmap();
    const size_t block_index = index / 8,
                 block_offset = index % 8;
    size_t rank = 0;
    for (size_t i = 0; i < block_index; ++i) {
      rank += popcount(bitmap[i]); 
    }
    rank += popcount(bitmap[block_index] & ((1U << block_offset) - 1));
    return rank;
  }
};

class FixedColumnPageLocator {
public:
  FixedColumnPageLocator(const Column &column)
    : column_(std::addressof(column)) {
    const auto &pages = column.pages;
    samples_.reserve(pages.size() + 1);

    size_t offset = 0;
    for (size_t i = 0; i < pages.size(); ++i) {
      samples_.push_back(offset);
      size_t num_rows = *reinterpret_cast<const uint16_t *>(pages[i]->data);
      offset += num_rows;
    }
    samples_.push_back(offset);
  }

  [[nodiscard]] std::pair<PageDataLocation, size_t> get_location(size_t index, size_t search_hint = 0) const noexcept {
    auto iter = branchless_upper_bound(samples_.begin() + search_hint, 
                                       samples_.end() - samples_.begin() - search_hint, index) - 1;

    size_t page_idx = std::distance(samples_.begin(), iter);

    const Page *page = column_->pages[page_idx];
    if (column_->type == DataType::INT32) {
      FixedPageReader<int32_t> reader(page, page_idx);
      return {reader.index_to_location(index - *iter), page_idx};
    } else if (column_->type == DataType::INT64) {
      FixedPageReader<int64_t> reader(page, page_idx);
      return {reader.index_to_location(index - *iter), page_idx};
    } else if (column_->type == DataType::FP64) {
      FixedPageReader<double> reader(page, page_idx);
      return {reader.index_to_location(index - *iter), page_idx};
    }

    unreachable_branch();
    return {{}, {}};
  }

private:
  const Column          *column_;
  std::vector<size_t>    samples_;
};

[[nodiscard]] inline bool is_compact_varchar_page(Page *p) {
  return *reinterpret_cast<const uint16_t *>(p->data) < 0xFFFE;
}

[[nodiscard]] inline bool is_long_varchar_head_page(Page *p) {
  return *reinterpret_cast<const uint16_t *>(p->data) == 0xFFFF;
}

[[nodiscard]] inline bool is_long_varchar_overflow_page(Page *p) {
  return *reinterpret_cast<const uint16_t *>(p->data) == 0xFFFE;
}

class VarcharColumnPageLocator {
public:
  VarcharColumnPageLocator() = default;

  VarcharColumnPageLocator(const Column &column)
    : column_(std::addressof(column)) {
    const auto &pages = column.pages;
    samples_.reserve(pages.size() + 1);
    entries_.reserve(pages.size() + 1);

    size_t offset = 0;
    for (size_t i = 0; i < pages.size(); ++i) {
      if (is_compact_varchar_page(pages[i])) {
        samples_.push_back(offset);
        size_t num_rows = *reinterpret_cast<const uint16_t *>(pages[i]->data);
        entries_.push_back(Entry::make(i, false, num_rows));
        offset += num_rows;
      } else if (is_long_varchar_head_page(pages[i])) {
        samples_.push_back(offset);
        entries_.push_back(Entry::make(i, true, 1));
        ++offset;
      } else if (is_long_varchar_overflow_page(pages[i])) {
        entries_.back().num++;
      }
    }
    samples_.push_back(offset);
  }

  [[nodiscard]] std::pair<PageDataLocation, size_t> get_location(size_t index, size_t search_hint = 0) const noexcept {
    auto iter = branchless_upper_bound(samples_.begin() + search_hint, 
                                       samples_.end() - samples_.begin() - search_hint, index) - 1;
    size_t entry_idx = std::distance(samples_.begin(), iter);
    Entry entry = entries_[entry_idx];
    const Page *page = column_->pages[entry.page_index];
    if (!entry.is_long_string_page) {
      CompactStringPageReader reader(page, entry.page_index);
      return {reader.index_to_location(index - *iter), entry_idx};
    } else {
      return {PageDataLocation(entry.page_index, 0, 0), entry_idx};
    }
  }

private:
  struct Entry {
    uint32_t page_index;
    uint32_t is_long_string_page : 1;
    uint32_t num : 31;

    [[nodiscard]] static Entry make(uint32_t page_index, bool is_long_string_page, uint32_t num) {
      Entry e;
      e.page_index = page_index;
      e.is_long_string_page = is_long_string_page;
      e.num = num;
      return e;
    }
  };

  const Column        *column_{nullptr};
  std::vector<size_t>  samples_;
  std::vector<Entry>   entries_;
};

namespace Details {

template<typename AllocPageCallbackT, typename FlushPageCallbackT>
class PageWriterBase {
public:
  PageWriterBase(AllocPageCallbackT alloc_page_cb, FlushPageCallbackT flush_page_cb)
    : alloc_page_cb_(std::move(alloc_page_cb)), flush_page_cb_(std::move(flush_page_cb)) {
    bitmap_buf_.reserve(256);
  }

protected:
  AllocPageCallbackT    alloc_page_cb_;
  FlushPageCallbackT    flush_page_cb_;
  Page                 *cur_page_{nullptr};
  uint8_t              *writer_ptr_{nullptr};
  std::vector<uint8_t>  bitmap_buf_;
  size_t                num_rows_{0};

  void set_bitmap(size_t index) {
    while (bitmap_buf_.size() < index / 8 + 1) {
      bitmap_buf_.push_back(0);
    }
    set_bit(bitmap_buf_.data(), index);
  }

  void unset_bitmap(size_t index) {
    while (bitmap_buf_.size() < index / 8 + 1) {
      bitmap_buf_.push_back(0);
    }
    unset_bit(bitmap_buf_.data(), index);
  }

  [[nodiscard]] uint8_t *page_raw_data() {
    return cur_page_ ? reinterpret_cast<uint8_t *>(cur_page_->data) : nullptr;
  }

  void force_flush() {
    flush_page_cb_(this->cur_page_);
    cur_page_ = nullptr;
    writer_ptr_ = nullptr;
    bitmap_buf_.clear();
    num_rows_ = 0;
  }
};

}

template<typename T, typename AllocPageCallbackT, typename FlushPageCallbackT>
class PageWriter : public Details::PageWriterBase<AllocPageCallbackT, FlushPageCallbackT> {
  using Base = Details::PageWriterBase<AllocPageCallbackT, FlushPageCallbackT>;
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
public:
  PageWriter(AllocPageCallbackT alloc_page_cb, FlushPageCallbackT flush_page_cb)
    : Base(std::move(alloc_page_cb), std::move(flush_page_cb)) { }

  PageWriter(Page *cur_page, AllocPageCallbackT alloc_page_cb, FlushPageCallbackT flush_page_cb)
    : Base(std::move(alloc_page_cb), std::move(flush_page_cb)) {
    this->cur_page_ = cur_page;
  }

  template<typename InputIterT>
  void write_values(InputIterT first, size_t num) {
    constexpr T NULL_VALUE = FixedPageTraits<T>::NULL_VALUE;
    constexpr size_t ELEMENT_SIZE = sizeof(T);

    if (!this->cur_page_) {
      init_new_page();
    }

    for (size_t i = 0; i < num; ++i) {
      if (ELEMENT_SIZE + (this->num_rows_ + 1) * ELEMENT_SIZE + (this->num_rows_ / 8 + 1) > PAGE_SIZE) {
        flush();
        init_new_page();
      }
      T val = *first++;
      if (val == NULL_VALUE) {
        this->unset_bitmap(this->num_rows_);
      } else {
        this->set_bitmap(this->num_rows_);
        *reinterpret_cast<T *>(this->writer_ptr_) = val;
        this->writer_ptr_ += ELEMENT_SIZE;
      }
      ++this->num_rows_;
    }
  }

  void flush() {
    uint8_t *page_data = this->page_raw_data();
    *reinterpret_cast<uint16_t *>(page_data) = this->num_rows_;
    uint8_t *bitmap_addr = page_data + PAGE_SIZE - (this->num_rows_ + 7) / 8;
    memcpy(bitmap_addr, this->bitmap_buf_.data(), (this->num_rows_ + 7) / 8);
    this->force_flush();
  }

private:
  void init_new_page() {
    this->cur_page_ = this->alloc_page_cb_();
    this->writer_ptr_ = reinterpret_cast<uint8_t*>(this->cur_page_->data + sizeof(T));
  }
};

template<typename AllocPageCallbackT, typename FlushPageCallbackT>
class PageWriter<std::string, AllocPageCallbackT, FlushPageCallbackT>
  : public Details::PageWriterBase<AllocPageCallbackT, FlushPageCallbackT>  {
  using Base = Details::PageWriterBase<AllocPageCallbackT, FlushPageCallbackT>;
public:
  PageWriter(AllocPageCallbackT alloc_page_cb, FlushPageCallbackT flush_page_cb)
    : Base(std::move(alloc_page_cb), std::move(flush_page_cb)) {
    short_str_buf_.resize(PAGE_SIZE);
  }

  template<typename InputIterT>
  void write_values(InputIterT first, size_t num) {
    for (size_t i = 0; i < num; ++i) {
      if (!this->cur_page_) {
        init_new_page();
      }

      std::string_view val = *first++;
      if (val.empty()) {
        this->unset_bitmap(this->num_rows_);
        ++this->num_rows_;
      } else if (val.size() > PAGE_SIZE - 7) { // long string
        write_long_string(val);
      } else {
        bool prv_is_long = i == 0 ? false : ((first - 1)->size() > PAGE_SIZE - 7);
        write_short_string(val, prv_is_long);
      }
    }
  }

  void write_chars(std::string_view val) {
    if (!this->cur_page_) {
      init_new_page();
    }
    if (val.empty()) {
      this->unset_bitmap(this->num_rows_);
    } else if (val.size() > PAGE_SIZE - 7) {
      write_long_string(val);
    } else {
      write_short_string(val, false);
    }

    ++this->num_rows_;
  }

  void flush() {
    if (!this->cur_page_) { return; }

    uint8_t *page_data = this->page_raw_data();
    *reinterpret_cast<uint16_t *>(page_data) = this->num_rows_;
    *reinterpret_cast<uint16_t *>(page_data + 2) = static_cast<uint16_t>(this->offsets_buf_.size());
    memcpy(page_data + 4, offsets_buf_.data(), offsets_buf_.size() * sizeof(uint16_t));

    memcpy(page_data + 4 + offsets_buf_.size() * sizeof(uint16_t), short_str_buf_.data(), data_size_);

    uint8_t *bitmap_addr = page_data + PAGE_SIZE - this->bitmap_buf_.size();
    memcpy(bitmap_addr, this->bitmap_buf_.data(), this->bitmap_buf_.size());

    this->force_flush();
    offsets_buf_.clear();
    data_size_ = 0;
  }

private:
  std::vector<uint16_t> offsets_buf_;
  Buffer<char>          short_str_buf_;
  size_t                data_size_{0};

  void init_new_page() {
    this->cur_page_ = this->alloc_page_cb_();
    this->writer_ptr_ = reinterpret_cast<uint8_t*>(this->cur_page_->data + 4);
  }

  void write_long_string(std::string_view val) {
    if (this->num_rows_ > 0) {
      flush();
      init_new_page();
    }

    size_t writer_offset = 0;
    
    auto write_str_data = [&]() {
      size_t page_data_len = std::min(val.size() - writer_offset, PAGE_SIZE - 4);
      *reinterpret_cast<uint16_t *>(this->page_raw_data() + 2) = page_data_len;
      memcpy(this->page_raw_data() + 4, val.data() + writer_offset, page_data_len);
      writer_offset += page_data_len;
    };

    auto flush_long_string_page = [&]() {
      this->flush_page_cb_(this->cur_page_);
      this->cur_page_ = nullptr;
    };

    *reinterpret_cast<uint16_t *>(this->page_raw_data()) = 0xFFFF;
    write_str_data();
    flush_long_string_page();

    while (writer_offset < val.size()) {
      if (!this->cur_page_) {
        init_new_page();
      }

      *reinterpret_cast<uint16_t *>(this->page_raw_data()) = 0xFFFE;
      write_str_data();
      flush_long_string_page();
    }
  }

  void write_short_string(std::string_view val, bool prv_is_long) {
    if (4 + (offsets_buf_.size() + 1) * 2 + (data_size_ + val.size()) + (this->num_rows_ / 8 + 1) > PAGE_SIZE) {
      flush();
      init_new_page();
    }

    this->set_bitmap(this->num_rows_);

    memcpy(short_str_buf_.data() + data_size_, val.data(), val.size());
    data_size_ += val.size();

    offsets_buf_.push_back(data_size_);
    ++this->num_rows_;
  }
};

template<typename T, typename AllocPageCallbackT, typename FlushPageCallbackT>
inline PageWriter<T, std::decay_t<AllocPageCallbackT>, std::decay_t<FlushPageCallbackT>>
create_page_writer(AllocPageCallbackT &&alloc_page_cb, FlushPageCallbackT &&flush_page_cb) {
  using WriterType = PageWriter<T, std::decay_t<AllocPageCallbackT>, std::decay_t<FlushPageCallbackT>>;
  return WriterType{std::forward<AllocPageCallbackT>(alloc_page_cb),
                    std::forward<FlushPageCallbackT>(flush_page_cb)};
}

}