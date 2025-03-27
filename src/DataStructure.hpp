#pragma once

#include "plan.h"
#include "iostream"
#include "sstream"
#include "MemoryPool.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Contest {
#define FULL_INT32_PAGE (1984)

#define NULL_INT32 (-2147483648)

#define NULL_VARCHAR 0

#define LONG_STRING_START (0xffff)

#define LOGNG_STRING_FOLLOW (0xfffe)

struct varchar_ptr {
    uint64_t ptr_ = {0};

    varchar_ptr(const char* str, uint16_t length) { set(str, length); }

    varchar_ptr(Page* const* page, uint16_t num_page) { set(page, num_page); }

    inline varchar_ptr() = default;

    inline void set(const char* str, uint16_t length) {
        ptr_ = ((uint64_t)length << 48) | (uint64_t)(str);
    }

    inline void set(Page* const* page, uint16_t num_page) { // 最高位设为1表示Long String，其余高15位表示页数，低48位指向指向Column.pages的存储着该Long String的起始页。
        ptr_ = ((uint64_t)num_page << 48) | (uint64_t)(page) | 0x8000000000000000;
    }

    inline const char* string() const { return (const char*)(ptr_ & 0x0000FFFFFFFFFFFF); }

    inline uint16_t length() const { return (uint16_t)(ptr_ >> 48); }

    inline bool isNull() const { return ptr_ == NULL_VARCHAR; }

    inline bool isLongString() const { return ptr_ >> 63; }

    inline Page* const* longStringPage() const {
        return (Page* const*)(ptr_ & 0x0000FFFFFFFFFFFF);
    }

    inline uint16_t longStringPageNum() const { return (uint16_t)(ptr_ >> 48) & 0x7FFF; }
};

// 一些辅助处理Page信息的函数
template<typename Alloc>
void set_bitmap(std::vector<uint8_t, Alloc>& bitmap, uint16_t idx) {
    while (bitmap.size() < idx / 8 + 1) {
        bitmap.emplace_back(0);
    }
    auto byte_idx     = idx / 8;
    auto bit          = idx % 8;
    bitmap[byte_idx] |= (1u << bit);
}

template<typename Alloc>
void unset_bitmap(std::vector<uint8_t, Alloc>& bitmap, uint16_t idx) {
    while (bitmap.size() < idx / 8 + 1) {
        bitmap.emplace_back(0);
    }
    auto byte_idx     = idx / 8;
    auto bit          = idx % 8;
    bitmap[byte_idx] &= ~(1u << bit);
}

class TempPage {
public:
    virtual Page *dump_page() = 0;
    virtual bool is_empty() = 0;
    virtual ~TempPage() = default;
};

class TempIntPage : public TempPage {
private:
    LocalVector<uint8_t> bitmap_;
    Page *page_ = nullptr;
    void alloc_page() {
#ifdef PROFILER
        if (page_ != nullptr) {
            throw std::runtime_error("page is not null");
        }
#endif
        page_ = new Page;
        memset(page_->data, 0, PAGE_SIZE);
    }
public:
    TempIntPage() {
        bitmap_.reserve(256);
    }

    uint16_t &num_rows() {
        return *(uint16_t *)(page_->data);
    }

    uint16_t &num_values() {
        return *(uint16_t *)(page_->data + 2);
    }

    int32_t *data() {
        return (int32_t *)(page_->data + 4);
    }

    bool is_full() {
        if (page_ == nullptr) {
            return false;
        } else {
            return 4 + (num_values() + 1) * 4 + (num_rows() / 8 + 1) > PAGE_SIZE;
        }
    }

    bool is_empty() override {
        return page_ == nullptr;
    }

    void add_row(int value) {
#ifdef PROFILER
        if (is_full()) {
            throw std::runtime_error("page is full");
        }
#endif
        if (is_empty()) {
            alloc_page();
        }
        if (value != NULL_INT32) {
            set_bitmap(bitmap_, num_rows());
            data()[num_values()] = value;
            num_values() ++;
        } else {
            unset_bitmap(bitmap_, num_rows());
        }
        num_rows() ++;
    }

    Page *dump_page() override {
        Page *result_page = page_;
        memcpy(page_->data + PAGE_SIZE - bitmap_.size(), bitmap_.data(), bitmap_.size());
        bitmap_.clear();
        page_ = nullptr;
        return result_page;
    }
};

class TempStringPage : public TempPage {
private:
  uint16_t num_rows = 0;
  uint16_t data_len = 0;
  LocalVector<varchar_ptr> strings_;
  LocalVector<uint16_t> offsets_;
  LocalVector<uint8_t> bitmap_;

public:
    TempStringPage() {
        strings_.reserve(256);
        offsets_.reserve(4096);
        bitmap_.reserve(512);
    }

    bool is_empty() override {
        return num_rows == 0;
    }

    bool can_store_string(const varchar_ptr value) {
        if(value.isLongString()){
            return num_rows==0;
        } else if (value.isNull()) {
            return 4 + offsets_.size() * 2 + data_len + (num_rows / 8 + 1) <= PAGE_SIZE;
        } else {
            return 4 + (offsets_.size() + 1) * 2 + (data_len + value.length()) + (num_rows / 8 + 1) <= PAGE_SIZE;
        }
    }

    void add_row(const varchar_ptr value) {
#ifdef PROFILER
        if (!can_store_string(value)) {
            throw std::runtime_error("page is full");
        }
        if (value.length() > PAGE_SIZE - 7) {
            throw std::runtime_error("long string is not support");
        }
#endif
        if (value.isNull()) {
            unset_bitmap(bitmap_, num_rows);
            ++num_rows;
        } else {
            set_bitmap(bitmap_, num_rows);
            data_len += value.length();
            strings_.emplace_back(value);
            offsets_.emplace_back(data_len);
            ++num_rows;
        }
    }

    Page *dump_page() override {
        auto* page                             = new Page;
        *reinterpret_cast<uint16_t*>(page->data)     = num_rows;
        *reinterpret_cast<uint16_t*>(page->data + 2) = static_cast<uint16_t>(offsets_.size());
        memcpy(page->data + 4, offsets_.data(), offsets_.size() * sizeof(uint16_t));
        memcpy(page->data + PAGE_SIZE - bitmap_.size(), bitmap_.data(), bitmap_.size());

        std::byte *current = page->data + 4 + offsets_.size() * sizeof(uint16_t);
        for (size_t i = 0; i < strings_.size(); i ++) {
            memcpy(current, strings_[i].string(), strings_[i].length());
            current += strings_[i].length();
        }
        
        num_rows = 0;
        data_len = 0;
        strings_.clear();
        offsets_.clear();
        bitmap_.clear();
        return page;
    }

};

// 调用算子next得到的结果表。每一列可能是已经实例化的，也有可能只是给出了在原表中的行号。
// SCAN算子给出的结果都是未实例化的，并且是一段连续的行号。
// JOIN算子会将链接键值实例化。
class OperatorResultTable {
public:
    // 总共的行数
    size_t num_rows_{0};

    // 定义三种列类型：
    // 1. 已实例化的列，以 std::pair<DataType, void*> 表示，
    //    表示列的数据已经实例化，并提供该列的数据指针。
    using InstantiatedColumn = std::pair<DataType, void*>;

    // 2. 未实例化且行号连续的列，以 std::pair<Column *, uint32_t> 表示，
    //    表示引用的原始列，以及起始行所在的页码，以及起始行在该页之内的位置。
    using ContinuousColumn = std::tuple<const Column*, uint32_t, uint32_t>;

    // 3. 未实例化且行号不连续的列，以 std::pair<Column *, uint32_t *> 表示，
    //    表示引用的原始列，以及存储这些不连续行号的数组（多个未实例化列的行号可能完全相同，因而指向同一个行号数组）。
    //    这个类被弃用了，因为从NonContinuousColumn读取实际值的代价很大，有很多的随机访问，还需要计算行号。
    //    using NonContinuousColumn = std::pair<Column *, uint32_t *>;

    // 将以上三种列类型组合成一个 std::variant 类型
    using ColumnVariant = std::variant<InstantiatedColumn, ContinuousColumn>;

    // 存储所有列的变体集合
    LocalVector<ColumnVariant> columns_;

    inline bool isEmpty() const { return num_rows_ == 0; }

    // 打印出该表。目前只支持列全部为InstantiatedColumn的情况
    void print() const {
        std::cout << toString(); // 直接输出 toString 的结果
    }

    std::string toString(bool head = true) const {
        std::ostringstream oss;

        // 添加表头信息
        if (head) {
            oss << "table size: " << num_rows_ << " rows * " << columns_.size() << " cols\n";
        }

        // 构建列信息
        std::vector<std::variant<const int32_t*, const varchar_ptr*>> cols;
        for (const ColumnVariant& column_variant: columns_) {
            auto col = std::get<InstantiatedColumn>(column_variant);
            if (col.first == DataType::INT32) {
                cols.emplace_back(reinterpret_cast<const int32_t*>(col.second));
            } else if (col.first == DataType::VARCHAR) {
                cols.emplace_back(reinterpret_cast<const varchar_ptr*>(col.second));
            } else {
                throw std::runtime_error("Unsupported data type");
            }
        }

        // 构建行信息
        for (size_t i = 0; i < num_rows_; i++) {
            for (auto col_ptr: cols) {
                std::visit(
                    [&](auto&& ptr) {
                        using T = std::decay_t<decltype(ptr)>;
                        if constexpr (std::is_same_v<T, const int32_t*>) {
                            if (ptr[i] != NULL_INT32) {
                                oss << ptr[i] << "\t\t";
                            } else {
                                oss << "NULL" << "\t\t";
                            }
                        } else if constexpr (std::is_same_v<T, const varchar_ptr*>){
                            if(ptr[i].isLongString()){
                                std::string long_str;
                                uint16_t page_num = ptr[i].longStringPageNum();  // 该Long String有多少页
                                Page* const* page_start = ptr[i].longStringPage();
                                for(size_t k=0; k<page_num; k++){
                                    long_str.append((const char*)(page_start[k]->data + 4), *reinterpret_cast<const uint16_t*>(page_start[k]->data + 2));
                                }
                                oss << long_str << "\t\t";
                            } else if (!ptr[i].isNull()) {
                                oss << std::string(ptr[i].string(), ptr[i].length()) << "\t\t";
                            } else {
                                oss << "NULL" << "\t\t";
                            }
                        }
                    },
                    col_ptr);
            }
            oss << "\n"; // 换行
        }

        return oss.str(); // 返回构建的字符串
    }
};



// 获取页面的总行数（含NULL值）
inline uint16_t getRowCount(const Page* page) {
    return *reinterpret_cast<const uint16_t*>(page->data);
}

// 获取页面的非NULL值数量
inline uint16_t getNonNullCount(const Page* page) {
    return *reinterpret_cast<const uint16_t*>(page->data + 2);
}

// 一个页中数据区域起始指针
template <typename T>
inline const T* getPageData(const Page* page) {
    if constexpr (std::is_same_v<T, int32_t>) {
        return reinterpret_cast<const int32_t*>(page->data + 4);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return reinterpret_cast<const int64_t*>(page->data + 8);
    } else if constexpr (std::is_same_v<T, double>) {
        return reinterpret_cast<const double*>(page->data + 8);
    } else if constexpr (std::is_same_v<T, char>) {
        // 检查是否为Long String。Long String的开头16字节为0xffff
        assert(getRowCount(page) != 0xffff);
        return reinterpret_cast<const char*>(page->data + 4 + getNonNullCount(page) * 2);
    } else {
        // 编译时静态断言：不支持的类型
        static_assert(std::is_same_v<T, void>, "Unsupported type: T must be int32_t or char");
        return nullptr;
    }
}

// 获取位图
inline const uint8_t* getBitmap(const Page* page) {
    return reinterpret_cast<const uint8_t*>(
        page->data + PAGE_SIZE - (getRowCount(page) + 7) / 8);
}

// 获取VARCHAR页面中的偏移数组的指针
inline const uint16_t* getVarcharOffset(const Page* page) {
    return reinterpret_cast<const uint16_t*>(page->data + 4);
}

// 位图中idx号元素是否为NULL
inline bool isNotNull(const uint8_t* bitmap, uint16_t idx) {
    auto byte_idx = idx / 8;
    auto bit      = idx % 8;
    return bitmap[byte_idx] & (1u << bit);
}

// 计算位图的前n个元素有多少个不为NULL
size_t getNonNullCount(const uint8_t* bitmap, uint16_t n) {
    if(n==0) return 0;

    int      count          = 0;
    uint16_t full_bytes     = n / 8;
    uint8_t  remaining_bits = n % 8;

    // 统计完整字节中的 1 的位数
    for (uint16_t i = 0; i < full_bytes; ++i) {
        count += __builtin_popcount(bitmap[i]);
    }

    // 处理剩余位
    if (remaining_bits > 0) {
        uint8_t last_byte  = bitmap[full_bytes];
        uint8_t mask       = (1 << remaining_bits) - 1; // 保留低 remaining_bits 位
        count             += __builtin_popcount(last_byte & mask);
    }

    return count;
}

// 判断一个页是否是Long String的起始页
inline bool isLongStringStart(const Page* page) {
    return getRowCount(page) == LONG_STRING_START;
}

// 判断一个页是否是Long String的后续页
inline bool isLongStringFollow(const Page* page) {
    return getRowCount(page) == LOGNG_STRING_FOLLOW;
}

// 获取VARCHAR页面含有的字符串数
inline uint16_t getStringCount(const Page* page) {
    uint16_t first16 = getRowCount(page);
    if(first16==LONG_STRING_START){
        return 1;
    } else if(first16==LOGNG_STRING_FOLLOW){
        return 0;
    } else{
        return first16;
    }
}


// 将InstantiatedColumn的值读取出来，并存储到指定位置。
// input_column: 要读取的ColumnVariant
// n: 读取的行数
// col_target: 存储到的位置
// col_step: col_target每次移动的步长
// SpecifiedIndex: 从指定的下标数组中读取行号，而不是连续的n行
// idx: 指定行的下标数组。下标数组必须递增。下标数组大小必须大于等于n+1!!!
template <bool SpecifiedIndex>
void gatherInstantiatedCol(OperatorResultTable::InstantiatedColumn input_column, size_t n, uint8_t* col_target, size_t col_step, const uint32_t* idx=nullptr){
    // 已实例化的列为 std::pair<DataType, void*>。
    if(input_column.first==DataType::INT32){
        const int32_t* base = (int32_t*)input_column.second;
        for (size_t i = 0; i < n; ++i) {
            if constexpr (SpecifiedIndex){
                *(int32_t*)(col_target) = base[idx[i]];
            } else {
                *(int32_t*)(col_target) = base[i];
            }
            col_target += col_step;
        }
    } else if(input_column.first==DataType::VARCHAR){
        const uint64_t* base = (uint64_t*)input_column.second;
        for (size_t i = 0; i < n; ++i) {
            if constexpr (SpecifiedIndex){
                *(uint64_t*)(col_target) = base[idx[i]];
            } else {
                *(uint64_t*)(col_target) = base[i];
            }
            col_target += col_step;
        }
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}


// 将ContinuousColumn的连续n行的值读取出来，并存储到指定位置。
// input_column: 要读取的ColumnVariant
// n: 读取的行数
// col_target: 存储到的位置
// col_step: col_target每次移动的步长
void gatherContinuousCol(OperatorResultTable::ContinuousColumn input_column, size_t n, uint8_t* col_target, size_t col_step){
    // 连续未实例化的列为 std::tuple<Column*, uint32_t, uint32_t>
    const Column* col = std::get<0>(input_column);
    if(col->type==DataType::INT32) {
        for(size_t i=std::get<1>(input_column); i<col->pages.size(); i++){
            const Page* current_page = col->pages[i];                       // 要读取的页面
            const uint8_t* bitmap = getBitmap(current_page);                // 要读取页面的位图
            size_t start_row = i==std::get<1>(input_column) ? std::get<2>(input_column) : 0;  // 本页的起始行
            size_t end_row = std::min((size_t)getRowCount(current_page), n + start_row);  // 本页的终止行
            const int32_t* base = getPageData<int32_t>(current_page) + getNonNullCount(bitmap, start_row);

            for (size_t j=start_row; j<end_row; j++) {
                if (isNotNull(bitmap, j)) {
                    *(int32_t*)(col_target) = *(base++);
                } else {
                    *(int32_t*)(col_target) = NULL_INT32;
                }
                col_target += col_step;
            }

            n -= end_row-start_row;
            if(n<=0) break;
        }

    } else if(col->type==DataType::VARCHAR){
        for(size_t i=std::get<1>(input_column); i<col->pages.size(); i++){
            const Page* current_page = col->pages[i];                       // 要读取的页面
            assert(!isLongStringFollow(current_page));

            if(isLongStringStart(current_page)){    // 如果当前页面是Long String起始页
                // 寻找Long String一共有多少页
                size_t end_page = i+1;
                while(end_page<col->pages.size() && isLongStringFollow(col->pages[end_page])){
                    end_page++;
                }
                varchar_ptr ptr(&col->pages[i],end_page-i);
                *(uint64_t *)(col_target) = ptr.ptr_;
                col_target += col_step;
                n -= 1;
                i = end_page-1;
            } else {    // 否则该页是普通页
                const uint8_t* bitmap = getBitmap(current_page);                // 要读取页面的位图
                size_t start_row = i==std::get<1>(input_column) ? std::get<2>(input_column) : 0;  // 本页的起始行
                size_t end_row = std::min((size_t)getRowCount(current_page), n + start_row);  // 本页的终止行
                const char* base = getPageData<char>(current_page);             // 本页数据的起始指针
                const uint16_t* current_offset = getVarcharOffset(current_page) + getNonNullCount(bitmap, start_row);  // 当前字符串的结尾的偏移量
                uint16_t last_offset = getNonNullCount(bitmap, start_row)==0 ? 0 : *(current_offset-1);                // 当前字符串的开头的偏移量

                for (size_t j=start_row; j<end_row; j++) {
                    if (isNotNull(bitmap, j)) {
                        varchar_ptr ptr(base + last_offset,*current_offset - last_offset);
                        *(uint64_t *)(col_target) = ptr.ptr_;
                        last_offset = *current_offset;
                        current_offset ++;
                    } else {
                        *(uint64_t *)(col_target) = NULL_VARCHAR;
                    }
                    col_target += col_step;
                }

                n -= end_row-start_row;
            }
            if(n<=0) break;
        }
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}


// 将InstantiatedColumn的由idx指定的n行的值读取出来，并存储到指定位置。
// input_column: 要读取的ColumnVariant
// n: 读取的行数
// col_target: 存储到的位置
// col_step: col_target每次移动的步长
// idx: 指定行的下标数组。下标数组必须递增。下标数组大小必须大于等于n+1!!!
void gatherContinuousColWithIndex(OperatorResultTable::ContinuousColumn input_column, size_t n, uint8_t* col_target, size_t col_step, const uint32_t* idx){
    // 假设下标数组idx是递增的
    const Column* col = std::get<0>(input_column);
    Page *const * current_page = col->pages.data() + std::get<1>(input_column);
    uint32_t offset = idx[0] + std::get<2>(input_column);
    if(col->type==DataType::INT32) {
        for(uint32_t i=0; i<n; i++){
            // 定位到idx[i]所在的Page，和页内偏移
            while(offset >= getRowCount(*current_page)){
                offset -= getRowCount(*current_page);
                current_page++;
            }
            const uint8_t* bitmap = getBitmap(*current_page);

            // 取出数据
            if(isNotNull(bitmap, offset)){
                const int32_t* data_ptr = getPageData<int32_t>(*current_page) + getNonNullCount(bitmap, offset);
                *(int32_t*)(col_target) = *data_ptr;
            } else {
                *(int32_t*)(col_target) = NULL_INT32;
            }

            col_target += col_step;
            offset += idx[i+1] - idx[i]; // 假设下标数组idx是递增的
        }
    } else if(col->type==DataType::VARCHAR){
        for(uint32_t i=0; i<n; i++){
            // 定位到idx[i]所在的Page，和页内偏移
            while(offset >= getStringCount(*current_page)){
                offset -= getStringCount(*current_page);
                current_page++;
            }

            if(isLongStringStart(*current_page)){
                Page *const * end_page = current_page+1;
                while(end_page!=col->pages.data()+col->pages.size() && isLongStringFollow(*end_page)){
                    end_page++;
                }
                varchar_ptr ptr(current_page,end_page-current_page);
                *(uint64_t *)(col_target) = ptr.ptr_;
                current_page = end_page;
                offset = 0; // offset强制设为0，因为下一个条目必然从新页开始
            } else {
                const uint8_t* bitmap = getBitmap(*current_page);
                if(isNotNull(bitmap, offset)){
                    // 获取varchar的起始和终止位置
                    size_t non_null = getNonNullCount(bitmap,offset);
                    const uint16_t* str_end = getVarcharOffset(*current_page) + non_null;
                    uint16_t str_begin = non_null==0 ? 0 : *(str_end-1);
                    varchar_ptr ptr(getPageData<char>(*current_page) + str_begin,*str_end - str_begin);
                    *(uint64_t *)(col_target) = ptr.ptr_;
                } else {
                    *(uint64_t*)(col_target) = NULL_VARCHAR;
                }
                offset += idx[i+1] - idx[i]; // 假设下标数组idx是递增的
            }
            col_target += col_step;

        }
    } else {
        throw std::runtime_error("Unsupported data type");
    }
}


// 将ColumnVariant的值读取出来，并存储到指定位置
// input_column: 要读取的ColumnVariant
// n: 读取的行数
// col_target: 存储到的位置
// col_step: col_target每次移动的步长
// SpecifiedIndex: 从指定的下标数组中读取行号，而不是连续的n行
// idx: 指定行的下标数组。下标数组必须递增。下标数组大小必须大于等于n+1!!!
template <bool SpecifiedIndex>
void gatherCol(OperatorResultTable::ColumnVariant input_column, size_t n, uint8_t* col_target, size_t col_step, const uint32_t* idx=nullptr){
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
            gatherInstantiatedCol<SpecifiedIndex>(arg, n, col_target, col_step, idx);
        } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn> & !SpecifiedIndex) {
            gatherContinuousCol(arg, n, col_target, col_step);
        } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn> & SpecifiedIndex){
            gatherContinuousColWithIndex(arg, n, col_target, col_step, idx);
        }
    }, input_column);
}

}