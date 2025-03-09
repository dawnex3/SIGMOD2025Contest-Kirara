#pragma once

#include "plan.h"
#include "iostream"
#include "sstream"

namespace Contest {

#define FULL_INT32_PAGE (1984)

#define NULL_INT32 (-2147483648)

#define NULL_VARCHAR 0

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
    std::vector<ColumnVariant> columns_;

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
                        } else {
                            if (!ptr[i].isNull()) {
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

// 一些辅助处理Page信息的函数

void set_bitmap(std::vector<uint8_t>& bitmap, uint16_t idx) {
    while (bitmap.size() < idx / 8 + 1) {
        bitmap.emplace_back(0);
    }
    auto byte_idx     = idx / 8;
    auto bit          = idx % 8;
    bitmap[byte_idx] |= (1u << bit);
}

void unset_bitmap(std::vector<uint8_t>& bitmap, uint16_t idx) {
    while (bitmap.size() < idx / 8 + 1) {
        bitmap.emplace_back(0);
    }
    auto byte_idx     = idx / 8;
    auto bit          = idx % 8;
    bitmap[byte_idx] &= ~(1u << bit);
}

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
    return getRowCount(page) == 0xffff;
}

// 判断一个页是否是Long String的后续页
inline bool isLongStringFollowing(const Page* page) {
    return getRowCount(page) == 0xfffe;
}
}