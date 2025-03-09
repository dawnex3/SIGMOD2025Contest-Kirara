#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <iostream>
#include "SharedState.hpp"
#include "attribute.h"
#include "statement.h"
#include "plan.h"
#include "HashMap.hpp"
#include "Barrier.hpp"

namespace Contest {
#define DEBUG_LOG

#define FULL_INT32_PAGE 1984

#define NULL_INT32 -2147483648

#define NULL_VARCHAR 0

struct varchar_ptr {
    uint64_t ptr = {0};

    varchar_ptr(const char *str, uint16_t length) {
      set(str, length);
    }
    inline varchar_ptr() = default;
    inline void set(const char *str, uint16_t length) {
        ptr = ((uint64_t)length << 48) | (uint64_t)(str);
    }
    inline const char *string() const{
        return (const char *)(ptr & 0x0000FFFFFFFFFFFF);
    }
    inline uint16_t length() const{
        return (uint16_t)(ptr >> 48);
    }
    inline bool is_null() const{
        return ptr == NULL_VARCHAR;
    }
};

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
    using ContinuousColumn = std::tuple<const Column *, uint32_t, uint32_t>;

    // 3. 未实例化且行号不连续的列，以 std::pair<Column *, uint32_t *> 表示，
    //    表示引用的原始列，以及存储这些不连续行号的数组（多个未实例化列的行号可能完全相同，因而指向同一个行号数组）。
    //    这个类被弃用了，因为从NonContinuousColumn读取实际值的代价很大，有很多的随机访问，还需要计算行号。
    //    using NonContinuousColumn = std::pair<Column *, uint32_t *>;

    // 将以上三种列类型组合成一个 std::variant 类型
    using ColumnVariant = std::variant<InstantiatedColumn, ContinuousColumn>;

    // 存储所有列的变体集合
    std::vector<ColumnVariant> columns_;

    inline bool isEmpty() const{
        return num_rows_==0;
    };

    // 打印出该表。目前只支持列全部为InstantiatedColumn的情况
    void print() const {
        std::cout << toString(); // 直接输出 toString 的结果
    }

    std::string toString(bool head=true) const {
        std::ostringstream oss;

        // 添加表头信息
        if(head){
            oss << "table size: " << num_rows_ << " rows * " << columns_.size() << " cols\n";
        }

        // 构建列信息
        std::vector<std::variant<const int32_t*, const varchar_ptr*>> cols;
        for (const ColumnVariant& column_variant : columns_) {
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
            for (auto col_ptr : cols) {
                std::visit([&](auto&& ptr) {
                    using T = std::decay_t<decltype(ptr)>;
                    if constexpr (std::is_same_v<T, const int32_t*>) {
                        if(ptr[i]!=NULL_INT32){
                            oss << ptr[i] << "\t\t";
                        } else {
                            oss << "NULL" << "\t\t";
                        }
                    } else {
                        if(!ptr[i].is_null()){
                            oss << std::string(ptr[i].string(), ptr[i].length()) << "\t\t";
                        } else {
                            oss << "NULL" << "\t\t";
                        }
                    }
                }, col_ptr);
            }
            oss << "\n"; // 换行
        }

        return oss.str(); // 返回构建的字符串
    }
};


// 一些辅助读取Column信息的函数
// 获取页面的总行数（含NULL值）
inline uint16_t getRowCount(const Page* page) {
    return *reinterpret_cast<const uint16_t*>(page->data);
}

// 获取页面的非NULL值数量
inline uint16_t getNonNullCount(const Page* page) {
    return *reinterpret_cast<const uint16_t*>(page->data + 2);
}

// 一个页中数据区域起始指针
template<typename T>
inline const T* getPageData(const Page* page) {
    if constexpr (std::is_same_v<T, int32_t>) {
        return reinterpret_cast<const int32_t*>(page->data + 4);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return reinterpret_cast<const int64_t*>(page->data + 8);
    } else if constexpr (std::is_same_v<T, double>) {
        return reinterpret_cast<const double*>(page->data + 8);
    } else if constexpr (std::is_same_v<T, char>) {
        return reinterpret_cast<const char*>(page->data + 4 + getNonNullCount(page) * 2);
    } else {
        // 编译时静态断言：不支持的类型
        static_assert(std::is_same_v<T, void>, "Unsupported type: T must be int32_t or char");
        return nullptr;
    }
}

// 获取位图
inline const uint8_t* getBitmap(const Page* page){
    return reinterpret_cast<const uint8_t*>(page->data + PAGE_SIZE - (getRowCount(page) + 7) / 8);
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
    int count = 0;
    uint16_t full_bytes = n / 8;
    uint8_t remaining_bits = n % 8;

    // 统计完整字节中的 1 的位数
    for (uint16_t i = 0; i < full_bytes; ++i) {
        count += __builtin_popcount(bitmap[i]);
    }

    // 处理剩余位
    if (remaining_bits > 0) {
        uint8_t last_byte = bitmap[full_bytes];
        uint8_t mask = (1 << remaining_bits) - 1; // 保留低 remaining_bits 位
        count += __builtin_popcount(last_byte & mask);
    }

    return count;
}



class Operator {
public:
    Operator() = default;
    Operator(Operator&&) = default;
    Operator(const Operator&) = delete;
    virtual OperatorResultTable next() = 0;
    virtual size_t resultColumnNum() = 0;
    virtual ~Operator() = default;
};

// Scan算子
class Scan : public Operator {
public:
    struct Shared : public SharedState {
        std::atomic<size_t> pos_{0};    // 扫描到表的哪一个chunk
    };

private:
    Shared& shared_;         // 所有线程在该Scan节点上的共享状态

    size_t total_rows_;     // 表的总行数
    size_t last_row_{0};    // 上一次所返回批次的起始行号
    size_t vec_size_;       // 每个批次（vec）的行数

    size_t chunk_size_;      // 每个扫描块（chunk）包含多少个批次
    size_t vec_in_chunk_;    // 当前扫描块中已处理的批次数

    OperatorResultTable last_result_;       // 上一次调用next的返回结果。本次调用修改一些参数就可以返回。

public:
    // Shared& shared: 所有线程在该Scan节点上的共享状态
    // size_t total_rows: 总行数
    // size_t vec_size: 每批次的行数
    Scan(Shared& shared, size_t total_rows, size_t vec_size, std::vector<const Column*> &columns)
    : shared_(shared), total_rows_(total_rows), vec_size_(vec_size)
    {
        size_t scan_morsel_size = 1024 * 10;
        if (vec_size_ >= scan_morsel_size){  // 如果批次足够大，超过1024 * 10行，一个chunk仅一批次
            chunk_size_ = 1;
        } else {                             // 否则一个chunk大概要凑齐1024 * 10行
            chunk_size_ = scan_morsel_size / vec_size_ + 1;
        }
        vec_in_chunk_ = chunk_size_;

        // 构造last_result_
        for(auto column:columns){
            last_result_.columns_.emplace_back(std::make_tuple(column,0,0));
        }
    }

    size_t resultColumnNum() override{
        return last_result_.columns_.size();
    }

    OperatorResultTable next() override{
        size_t skip_rows;       // 本次调用，相比于上次跳过的行数

        if (vec_in_chunk_ == chunk_size_) {     // 如果本块已经处理完，取出新的一块，更新全局状态
            size_t current_chunk = shared_.pos_.fetch_add(1); // 取出shared_.pos_的旧值表示当前要处理的块，并加上1
            skip_rows = current_chunk * chunk_size_ * vec_size_ - last_row_;
            vec_in_chunk_ = 0;
        } else {    // 如果不取新块，那么前进一个批次的行
            skip_rows = vec_size_;
        }
        last_row_ += skip_rows;

        if(last_row_ >= total_rows_){
            last_result_.num_rows_ = 0;
            return last_result_;
        } else {
            last_result_.num_rows_ = std::min(total_rows_ - last_row_, vec_size_);
            // 设置last_result_中的页号行号。Scan给出的列都是ContinuousColumn
            for(OperatorResultTable::ColumnVariant& column : last_result_.columns_){
                // 对于每一列，计算current_row的页码
                auto& continuous_column = std::get<OperatorResultTable::ContinuousColumn>(column);
                const std::vector<Page *>& pages = std::get<0>(continuous_column)->pages;
                uint32_t start_page = std::get<1>(continuous_column);
                uint32_t offset_this_page = std::get<2>(continuous_column) + skip_rows;

                for(uint32_t i=start_page; i<pages.size(); i++){
                    uint16_t page_rows = getRowCount(pages[i]);
                    if (offset_this_page < page_rows) {
                        // 如果偏移在当前页范围内，更新页号和页内偏移
                        std::get<1>(continuous_column) = i;              // 更新为当前页码
                        std::get<2>(continuous_column) = offset_this_page; // 更新页内偏移
                        break;
                    }
                    offset_this_page -= page_rows; // 跳过当前页
                }
            }
            vec_in_chunk_ ++;
            return last_result_;
        }
    }
};


// Hashjoin算子
class Hashjoin : public Operator {
public:
    struct Shared : public SharedState {
        std::atomic<size_t> found_{0};             // 哈希表大小
        std::atomic<bool> size_is_set_{false};     // 哈希表大小是否已经设置
        Hashmap hashmap_;
    };

private:
    // 保存上次next的最后状态，用于下次next恢复任务继续进行
    struct IteratorContinuation
    {
        Hashmap::hash_t probe_hash_{0};        // 上次未探测完的哈希值
        Hashmap::EntryHeader* last_chain_{nullptr};  // 上次未完成的哈希链表

        // 额外的状态，指示循环队列的开始和结束位置。专门用于joinSelParallel
        uint32_t queue_begin_{0};
        uint32_t queue_end_{0};

        size_t num_probe_{0};              // right_result_的大小
        size_t next_probe_{0};             // right_result_中接下来要处理的行

    } cont_;

    Shared& shared_;

    uint32_t vec_size_;        // 批次大小

    std::unique_ptr<Operator> left_;     // 左侧算子（构建侧）
    size_t left_idx_;                    // 左侧键所在的列号
    std::unique_ptr<Operator> right_;    // 右侧算子（探测侧）
    size_t right_idx_;                   // 右侧键所在的列号

    size_t ht_entry_size_;  // 哈希表一行的大小。哈希表的一行由一个EntryHeader，加上其余要输出的构建侧列组成
    std::vector<size_t> build_value_offsets_;   // 构建侧每列相对于EntryHeader的偏移量

    bool is_build_{false};      // 指示哈希表是否已经构建完毕

    OperatorResultTable right_result_;      // 上次调用right_算子得到的结果
    Hashmap::hash_t* probe_hashes_;         // right_result_的键值列的哈希值
    Hashmap::EntryHeader** build_matches_;  // 哈希表中匹配的条目
    uint32_t * probe_matches_;              // 构建侧匹配的行的行号

    OperatorResultTable last_result_;       // 上一次调用next的返回结果。本次调用修改一些参数就可以返回。
    std::vector<size_t> output_attrs_;      // 需要输出哪些列

    std::vector<std::pair<uint8_t*, size_t>> allocations_;  // 存储多个哈希表条目数组<数组指针，数组大小>


    // 专门用于joinSelParallel的循环队列
    size_t circular_queue_size_{1025};      // 循环队列大小
    uint32_t *queue_probe_;                 // 探测侧的循环队列，存储行号
    Hashmap::EntryHeader** queue_build_;    // 构建侧的循环队列，指向Entry

#ifdef DEBUG_LOG
    size_t total_output{0};
    std::string table_str;
#endif

public:

    Hashjoin(Shared& shared, size_t vec_size, std::unique_ptr<Operator> left, size_t left_idx,
        std::unique_ptr<Operator> right, size_t right_idx,
        const std::vector<std::tuple<size_t, DataType>>& output_attrs, std::vector<std::tuple<size_t, DataType>> left_attrs)
    : shared_(shared), vec_size_(vec_size), left_(std::move(left)), left_idx_(left_idx),
    right_(std::move(right)), right_idx_(right_idx){
        // 计算ht_entry_size_和build_value_offsets_
        // 为了满足对齐的需求，并且让整个entry最小，需要对左表的列顺序重新排列，将int32_t放到前面，uint64_t放到后面
        build_value_offsets_.resize(left_attrs.size());
        ht_entry_size_ = (sizeof(Hashmap::EntryHeader) + 3) & ~3;   // 将EntryHeader大小补齐到4倍数
        for(uint32_t i=0; i<left_attrs.size(); i++){
            if(std::get<1>(left_attrs[i])==DataType::INT32){
                build_value_offsets_[i]=ht_entry_size_;
                ht_entry_size_ += 4;
            }
        }

        ht_entry_size_ = (ht_entry_size_ + 7) & ~7;   // 将ht_entry_size_补齐到8倍数
        for(uint32_t i=0; i<left_attrs.size(); i++){
            if(std::get<1>(left_attrs[i])!=DataType::INT32){
                build_value_offsets_[i]=ht_entry_size_;
                ht_entry_size_ += 8;
            }
        }

        // 分配缓冲区的内存
        probe_hashes_ = (Hashmap::hash_t*)malloc(vec_size_*sizeof(Hashmap::hash_t));
        build_matches_ = (Hashmap::EntryHeader**)malloc(vec_size_*sizeof(Hashmap::EntryHeader*));
        probe_matches_ = (uint32_t*)malloc(vec_size_*sizeof(uint32_t));

        // 分配结果表last_result_的内存，设置output_attrs_
        for(auto [col_idx, col_type]:output_attrs){
            output_attrs_.push_back(col_idx);
            if(col_type==DataType::INT32){
                void* col_buffer = malloc(vec_size*sizeof(int32_t));
                last_result_.columns_.emplace_back(std::make_pair(col_type, col_buffer));
            } else {
                void* col_buffer = malloc(vec_size*sizeof(uint64_t));
                last_result_.columns_.emplace_back(std::make_pair(col_type, col_buffer));
            }
        }

        // 分配循环队列的内存
        queue_probe_ = (uint32_t*)malloc(circular_queue_size_*sizeof(uint32_t));
        queue_build_ = (Hashmap::EntryHeader**)malloc(circular_queue_size_*sizeof(Hashmap::EntryHeader*));
    }

    ~Hashjoin() override{
        // 销毁缓冲区，以及last_result_
        free(probe_hashes_);
        free(build_matches_);
        free(probe_matches_);
        for(auto column_var:last_result_.columns_){
            auto column = std::get<OperatorResultTable::InstantiatedColumn>(column_var);
            free(column.second);
        }

        // 销毁哈希表
        for(auto [entrys,_]:allocations_){
            free(entrys);
        }

        // 销毁循环队列
        free(queue_probe_);
        free(queue_build_);
    }

    size_t resultColumnNum() override{
        return last_result_.columns_.size();
    }

    OperatorResultTable next() override{
        if (!is_build_) {     // 构建哈希表
            size_t found = 0;
            while (true){
                OperatorResultTable left_table = left_->next(); // 调用左算子
                size_t n = left_table.num_rows_;
                if(n==0) break;
                found += n;

                // 申请一块n*ht_entry_size_大小的内存，存放本批次的哈希表条目
                uint8_t *ht_entrys = (uint8_t*)malloc(n * ht_entry_size_);
                allocations_.emplace_back(ht_entrys, n);

                // 从数据源OperatorResultTable取出键值，计算键的哈希值并存储到EntryHeader当中。键本身也要存储到EntryHeader。
                OperatorResultTable::ColumnVariant  left_key = left_table.columns_[left_idx_];
                calculateColHash<true>(left_key, n, ht_entrys+offsetof(Hashmap::EntryHeader, hash), ht_entry_size_,
                    ht_entrys+build_value_offsets_[left_idx_],ht_entry_size_);

                // 将构建侧的其他列存储到EntryHeader后面。
                for(int col_idx=0; col_idx<left_table.columns_.size(); col_idx++){
                    if(col_idx==left_idx_) continue;
                    OperatorResultTable::ColumnVariant left_value = left_table.columns_[col_idx];
                    gatherCol<false>(left_value,n,ht_entrys+build_value_offsets_[col_idx],ht_entry_size_);
                }
            }


            shared_.found_.fetch_add(found);   // 加到所有线程的总found上
            current_barrier->wait([&]() {   // 等待所有线程完成计算，确定哈希表大小
                auto total_found = shared_.found_.load();
                if (total_found) shared_.hashmap_.setSize(total_found);
            });
            auto total_found = shared_.found_.load();
            if (total_found == 0) {
                is_build_ = true;
                last_result_.num_rows_ = 0;
                return last_result_;
            }

            // 将allocations中的所有哈希表条目插入哈希表
            for (auto [ht_entrys, n] : allocations_) {
                shared_.hashmap_.insertAll_tagged(
                    reinterpret_cast<Hashmap::EntryHeader*>(ht_entrys), n, ht_entry_size_);
            }
            is_build_ = true;
            current_barrier->wait(); // 等待所有线程插入哈希表
        }

        // 探测阶段
        while (true) {
            if (cont_.next_probe_ >= cont_.num_probe_) {
                right_result_ = right_->next();     // 调用右侧算子，结果保存到right_result_
                cont_.next_probe_ = 0;
                cont_.num_probe_ = right_result_.num_rows_;
                if (cont_.num_probe_ == 0) {
                    last_result_.num_rows_ = 0;
#ifdef DEBUG_LOG
                    // 打印该节点输出的总行数
                    printf("join output rows: %ld, details:\n",total_output);
                    std::cout<<table_str<<std::endl;
#endif
                    return last_result_;
                }
                // 计算right_result_中键的哈希值，存储到probe_hashes_数组中
                calculateColHash<false>(right_result_.columns_[right_idx_],right_result_.num_rows_,
                    (uint8_t *)probe_hashes_,sizeof(Hashmap::hash_t));
            }
            // 调用探测函数，检测哈希值相等的(Entry*, pos)对，分别存储在build_matches_和probe_matches_中
            uint32_t n = joinAll();
            // 检查(Entry*, pos)对的键值是否确实相等，移除不相等的对
            n = checkKeyEquality(n);
            if (n == 0) continue;
            // 物化最终结果。将匹配的(Entry*, pos)中，左侧的值收集起来，右侧对应的行的值也收集起来
            last_result_.num_rows_ = n;
            size_t curr_out_col = 0;
            for(size_t col_idx:output_attrs_){
                size_t left_column_num = left_->resultColumnNum();
                auto column = std::get<OperatorResultTable::InstantiatedColumn>(last_result_.columns_[curr_out_col]);
                curr_out_col++;
                if(col_idx < left_column_num){      //如果是构建表
                    gatherEntry(column,n,build_value_offsets_[col_idx]);
                } else {                            // 如果是探测表
                    gatherCol<true>(right_result_.columns_[col_idx-left_column_num],n,
                        (uint8_t*)column.second,column.first==DataType::INT32 ? 4:8,probe_matches_);
                }
            }
#ifdef DEBUG_LOG
            total_output += n;
            table_str.append(last_result_.toString(false));
#endif
            return last_result_;
        }
    }


    // 计算一列的哈希值，并存储到指定位置。该列必须为INT32，不含NULL
    template <bool RestoreColumn>   //可选：将列值本身也存储到指定位置
    void calculateColHash(OperatorResultTable::ColumnVariant input_column, size_t n, uint8_t* hash_target, size_t hast_step, uint8_t* col_target=nullptr, size_t col_step=0){
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                // 已实例化的列为 std::pair<DataType, void*>，是一块连续的数组
                assert(arg.first==DataType::INT32);
                const int32_t* base = (int32_t*)arg.second;
                for (size_t i = 0; i < n; ++i) {
                    *(Hashmap::hash_t *)hash_target=hash_32(base[i]);
                    hash_target += hast_step;
                    if constexpr (RestoreColumn){
                        *(int32_t*)(col_target) = base[i];
                        col_target += col_step;
                    }
                }
            } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn>) {
                // 连续未实例化的列为 std::tuple<Column*, uint32_t, uint32_t>，它存放在多个Page中。它作为键值，必须是非空的
                const Column* col = std::get<0>(arg);
                assert(col->type==DataType::INT32);
                Page *const * current_page = col->pages.data() + std::get<1>(arg);
                size_t row_num_this_page = std::min((size_t)FULL_INT32_PAGE - std::get<2>(arg), n);
                const int32_t* base = getPageData<int32_t>(*current_page) + std::get<2>(arg);
                do{
                    for (size_t i = 0; i < row_num_this_page; ++i) {
                        *(Hashmap::hash_t *)hash_target=hash_32(base[i]);
                        hash_target += hast_step;
                        if constexpr (RestoreColumn){
                            *(int32_t*)(col_target) = base[i];
                            col_target += col_step;
                        }
                    }
                    n -= row_num_this_page;
                    current_page++;
                    if(n) base = getPageData<int32_t>(*current_page);
                    row_num_this_page = std::min((size_t)FULL_INT32_PAGE, n);
                }while (n>0);
            }
        }, input_column);
    }


    // 收集一列的值，并存储到指定位置。
    template <bool SpecifiedIndex>   //可选：指定行的下标数组。下标数组必须递增
    void gatherCol(OperatorResultTable::ColumnVariant input_column, size_t n, uint8_t* col_target, size_t col_step, const uint32_t* idx=nullptr){
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                // 已实例化的列为 std::pair<DataType, void*>。
                if(arg.first==DataType::INT32){
                    const int32_t* base = (int32_t*)arg.second;
                    for (size_t i = 0; i < n; ++i) {
                        if constexpr (SpecifiedIndex){
                            *(int32_t*)(col_target) = base[idx[i]];
                        } else {
                            *(int32_t*)(col_target) = base[i];
                        }
                        col_target += col_step;
                    }
                } else if(arg.first==DataType::VARCHAR){
                    const uint64_t* base = (uint64_t*)arg.second;
                    for (size_t i = 0; i < n; ++i) {
                        if constexpr (SpecifiedIndex){
                            *(uint64_t*)(col_target) = base[idx[i]];
                        } else {
                            *(uint64_t*)(col_target) = base[i];
                        }
                        col_target += col_step;
                    }
                }
            } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn> & !SpecifiedIndex) {
                // 连续未实例化的列为 std::tuple<Column*, uint32_t, uint32_t>
                const Column* col = std::get<0>(arg);
                Page *const * current_page = col->pages.data() + std::get<1>(arg);
                size_t start_row = std::get<2>(arg);
                if(col->type==DataType::INT32) {
                    const uint8_t* bitmap = getBitmap(*current_page);
                    size_t num_rows = std::min((size_t)getRowCount(*current_page) - std::get<2>(arg), n);
                    const int32_t* base = getPageData<int32_t>(*current_page) + getNonNullCount(bitmap, start_row);
                    do{
                        for (size_t i=start_row; i < num_rows+start_row; i++) {
                            if (isNotNull(bitmap, i)) {
                                *(int32_t*)(col_target) = *(base++);
                            } else {
                                *(int32_t*)(col_target) = NULL_INT32;
                            }
                            col_target += col_step;
                        }
                        n -= num_rows;
                        current_page++;
                        base = getPageData<int32_t>(*current_page);
                        bitmap = getBitmap(*current_page);
                        num_rows = std::min((size_t)getRowCount(*current_page), n);
                    } while (n>0);
                } else if(col->type==DataType::VARCHAR){
                    const uint8_t* bitmap = getBitmap(*current_page);
                    size_t num_rows = std::min((size_t)getRowCount(*current_page) - std::get<2>(arg), n);
                    size_t non_null = getNonNullCount(bitmap,start_row);
                    const uint16_t* str_end = getVarcharOffset(*current_page) + non_null;
                    uint16_t str_begin = non_null==0 ? 0 : *(str_end-1);
                    do{
                        const char* base = getPageData<char>(*current_page);
                        for (size_t i=start_row; i < num_rows+start_row; i++) {
                            if (isNotNull(bitmap, i)) {
                                *(uint64_t *)(col_target) = ((uint64_t)(*str_end - str_begin)<<48) | (uint64_t)(base + str_begin);
                                str_begin = *str_end;
                                str_end ++;
                            } else {
                                *(uint64_t *)(col_target) = NULL_VARCHAR;
                            }
                            col_target += col_step;
                        }
                        n -= num_rows;
                        current_page++;
                        bitmap = getBitmap(*current_page);
                        num_rows = std::min((size_t)getRowCount(*current_page), n);
                        str_end = getVarcharOffset(*current_page);
                        str_begin = 0;
                    } while (n>0);
                }
            } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn> & SpecifiedIndex){
                // 假设下标数组idx是递增的
                const Column* col = std::get<0>(arg);
                Page *const * current_page = col->pages.data() + std::get<1>(arg);
                uint32_t offset = idx[0] + std::get<2>(arg);
                const uint8_t* bitmap = getBitmap(*current_page);
                if(col->type==DataType::INT32) {
                    for(uint32_t i=0; i<n; i++){
                        // 定位到probe_matches_[i]所在的Page，和页内偏移
                        while(offset >= getRowCount(*current_page)){
                            offset -= getRowCount(*current_page);
                            current_page++;
                            bitmap = getBitmap(*current_page);
                        }

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
                        // 定位到probe_matches_[i]所在的Page，和页内偏移
                        while(offset >= getRowCount(*current_page)){
                            offset -= getRowCount(*current_page);
                            current_page++;
                            bitmap = getBitmap(*current_page);
                        }

                        // 取出数据
                        if(isNotNull(bitmap, offset)){
                            // 获取varchar的起始和终止位置
                            size_t non_null = getNonNullCount(bitmap,offset);
                            const uint16_t* str_end = getVarcharOffset(*current_page) + non_null;
                            uint16_t str_begin = non_null==0 ? 0 : *(str_end-1);
                            *(uint64_t *)(col_target) = ((uint64_t)(*str_end - str_begin)<<48) | (uint64_t)(getPageData<char>(*current_page) + str_begin);
                        } else {
                            *(uint64_t*)(col_target) = NULL_VARCHAR;
                        }

                        col_target += col_step;
                        offset += idx[i+1] - idx[i]; // 假设下标数组idx是递增的
                    }
                }
            }
        }, input_column);
    }


    // 收集build_matches_中的值，并存储到指定的InstantiatedColumn中
    void gatherEntry(OperatorResultTable::InstantiatedColumn output_column, uint32_t n, size_t offset){
        if(output_column.first==DataType::INT32){
            int32_t* base = (int32_t*)output_column.second;
            for(uint32_t i=0; i<n; i++){
                base[i] = *(int32_t*)((uint8_t*)build_matches_[i] + offset);
            }
        } else if(output_column.first==DataType::VARCHAR){
            uint64_t * base = (uint64_t*)output_column.second;
            for(uint32_t i=0; i<n; i++){
                base[i] = *(uint64_t*)((uint8_t*)build_matches_[i] + offset);
            }
        }
    }


    // 检查(Entry*, pos)两侧键值是否相等，移除不相等的(Entry*, pos)，返回相等的对数
    uint32_t checkKeyEquality(uint32_t n){
        uint32_t found = 0;
        size_t key_offset = build_value_offsets_[left_idx_];
        OperatorResultTable::ColumnVariant right_col = right_result_.columns_[right_idx_];

        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                assert(arg.first==DataType::INT32);
                const int32_t* base = (int32_t*)arg.second;
                for (uint32_t i = 0; i < n; i++) {
                    int32_t left_key = *(int32_t*)((uint8_t*)build_matches_[i] + key_offset);
                    int32_t right_key = base[probe_matches_[i]];
                    if(left_key == right_key){
                        build_matches_[found] = build_matches_[i];
                        probe_matches_[found] = probe_matches_[i];
                        found++;
                    }
                }
            } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn>) {
                // 连续未实例化的列为 std::tuple<Column*, uint32_t, uint32_t>。
                const Column* col = std::get<0>(arg);
                uint32_t start_page = std::get<1>(arg);
                uint32_t start_row = std::get<2>(arg);
                assert(col->type==DataType::INT32);

                for (uint32_t i = 0; i < n; i++) {
                    int32_t left_key = *(int32_t*)((uint8_t*)build_matches_[i] + key_offset);
                    uint32_t current_page = (probe_matches_[i] + start_row) / FULL_INT32_PAGE + start_page;
                    uint32_t current_row = (probe_matches_[i] + start_row) % FULL_INT32_PAGE;
                    int32_t right_key = *(getPageData<int32_t>(col->pages[current_page])+current_row);
                    if(left_key == right_key){
                        build_matches_[found] = build_matches_[i];
                        probe_matches_[found] = probe_matches_[i];
                        found++;
                    }
                }
            }
        }, right_col);

        return found;
    }


    // 匹配哈希值相等的Entry和右侧行，存入buildMatches和probeMatches
    uint32_t joinAll(){
        size_t found = 0;
        // 处理上次未完成的哈希链表
        for (Hashmap::EntryHeader* entry = cont_.last_chain_; entry != nullptr; entry = entry->next) {
            if (entry->hash == cont_.probe_hash_) {
                build_matches_[found] = entry;              // 记录左表（哈希表）匹配的EntryHeader
                probe_matches_[found] = cont_.next_probe_;  // 记录右表匹配的行号
                found ++;
                if (found == vec_size_) {   // 本批次已满，保存状态交给下一轮处理
                    cont_.last_chain_ = entry->next;
                    return vec_size_;
                }
            }
        }
        if (cont_.last_chain_ != nullptr){
            cont_.next_probe_++;    // 此时上次的哈希链表已经处理完成了。next_probe_推进一行，除了cont_.last_chain_==null的情况（也就是第一次执行）
        }
        for (size_t i = cont_.next_probe_, end = cont_.num_probe_; i < end; i++) {
            Hashmap::hash_t hash = probe_hashes_[i];
            for (auto entry = shared_.hashmap_.find_chain_tagged(hash); entry != nullptr; entry = entry->next) {
                if (entry->hash == hash) {
                    build_matches_[found] = entry;              // 记录左表（哈希表）匹配的EntryHeader
                    probe_matches_[found] = i;  // 记录右表匹配的行号
                    found ++;
                    if (found == vec_size_ && (entry->next != nullptr || i + 1 < end)) {
                        // output buffers are full, save state for continuation
                        cont_.last_chain_ = entry->next;
                        cont_.probe_hash_ = hash;
                        cont_.next_probe_ = i;
                        return vec_size_;
                    }
                }
            }
        }
        cont_.last_chain_ = nullptr;
        cont_.next_probe_ = cont_.num_probe_;
        return found;   // 该探测批次处理完了，但是没凑够vec_size_个
    }

    /// computes join result into buildMatches and probeMatches
    /// Implementation: optimized for long CPU pipelines
    uint32_t joinAllParallel(){
        size_t found = 0;
        auto followup = cont_.queue_begin_;   // 队列读指针
        auto followupWrite = cont_.queue_end_;   // 队列写指针

        if (followup == followupWrite) {       // 当循环队列为空的时候
            for (size_t i = 0, end = cont_.num_probe_; i < end; ++i) {    // 遍历该批次所有探测元组
                auto hash = probe_hashes_[i];
                auto entry = shared_.hashmap_.find_chain_tagged(hash);
                if (entry != nullptr) {        // 匹配项添加到buildMatches和probeMatches
                    if (entry->hash == hash) {
                        build_matches_[found] = entry;
                        probe_matches_[found] = i;
                        found += 1;
                    }
                    if (entry->next != nullptr) {  // 添加id-entry对到循环队列
                        queue_probe_[followupWrite] = i;
                        queue_build_[followupWrite] = entry->next;
                        followupWrite += 1;
                    }
                }
            }
        }

        followupWrite %= circular_queue_size_;

        while (followup != followupWrite) {          // 消耗直至队列为空
            auto remainingSpace = vec_size_ - found;  // 该批次的剩余容量
            auto nrFollowups = followup <= followupWrite
                                    ? followupWrite - followup
                                    : circular_queue_size_ - (followup - followupWrite);  // 队列的大小
            // std::cout << "nrFollowups: " << nrFollowups << "\n";
            auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);   // fittingElements是所能处理的最大数目
            for (size_t j = 0; j < fittingElements; ++j) {
                size_t i = queue_probe_[followup];
                auto entry = queue_build_[followup];
                // followup = (followup + 1) % followupBufferSize;
                followup = (followup + 1);
                if (followup == circular_queue_size_) followup = 0;
                auto hash = probe_hashes_[i];      // 取出队列最前端
                if (entry->hash == hash) {
                    build_matches_[found] = entry;
                    probe_matches_[found++] = i;
                }
                if (entry->next != nullptr) {  // 向队列尾部添加元素
                    queue_probe_[followupWrite] = i;
                    queue_build_[followupWrite] = entry->next;
                    followupWrite = (followupWrite + 1) % circular_queue_size_;
                }
            }
            if (fittingElements < nrFollowups) {   // 当remainingSpace < nrFollowups时，会在此提前结束该批次。尽管批次可能未满。
                // continuation
                cont_.queue_end_ = followupWrite;
                cont_.queue_begin_ = followup;
                return found;
            }
        }
        cont_.next_probe_ = cont_.num_probe_;
        cont_.queue_end_ = 0;
        cont_.queue_begin_ = 0;
        return found;
    }

    // join implementation after Peter's suggestions
    uint32_t joinBoncz();
    /// computes join result into buildMatches and probeMatches
    /// Implementation: Using AVX 512 SIMD
    uint32_t joinAllSIMD();
    /// computes join result into buildMatches and probeMatches, respecting
    /// selection vector probeSel for probe side
    uint32_t joinSel();
    /// computes join result into buildMatches and probeMatches, respecting
    /// selection vector probeSel for probe side
    /// Implementation: optimized for long CPU pipelines
    uint32_t joinSelParallel();
    /// computes join result into buildMatches and probeMatches, respecting
    /// selection vector probeSel for probe side
    /// Implementation: For SkylakeX using AVX512
    uint32_t joinSelSIMD();
};


// 用于收集各线程的结果，合并形成页，是整个计划树的根节点
class ResultWriter {
public:
    struct Shared : public SharedState {
        ColumnarTable output_;       // 最终要输出的表
        std::mutex m_;               // 对该表的锁
        Shared(const std::vector<std::tuple<size_t, DataType>>& output_attrs){
            // 根据output_attrs构建output_
            for(auto [_, col_type]: output_attrs){
                output_.columns.emplace_back(col_type);
            }
        }
    };

    Shared& shared_;

    std::unique_ptr<Operator> child_;   // 子算子。一般来说是个JOIN

    std::vector<std::vector<Page*>> page_buffers_;   // 暂时存储的每一列的page

    ResultWriter(Shared& shared, std::unique_ptr<Operator> child)
    : shared_{shared}, page_buffers_(shared.output_.columns.size()), child_(std::move(child))
    { }

    void next(){
        size_t found = 0;
        while(true){
            OperatorResultTable child_result = child_->next();
            size_t row_num = child_result.num_rows_;

            if(row_num==0) break;

            // 将每一列写入page_buffers_
            for(size_t i=0; i<child_result.columns_.size(); i++){
                auto from_column = std::get<OperatorResultTable::InstantiatedColumn>(child_result.columns_[i]);
                auto rows = child_result.num_rows_; 
                std::vector<Page*>& pages = page_buffers_[i];

                if(from_column.first==DataType::INT32){
                    int32_t *fomr_data = (int32_t*)from_column.second;
                    uint16_t             to_num_rows = 0;
                    std::vector<int32_t> to_data;
                    std::vector<uint8_t> to_bitmap;
                    to_data.reserve(2048);
                    to_bitmap.reserve(256);
                    auto gen_page = [&pages, &to_num_rows, &to_data, &to_bitmap]() {
                        auto* page                             = new Page;
                        *reinterpret_cast<uint16_t*>(page->data)     = to_num_rows;
                        *reinterpret_cast<uint16_t*>(page->data + 2) = static_cast<uint16_t>(to_data.size());
                        memcpy(page->data + 4, to_data.data(), to_data.size() * 4);
                        memcpy(page->data + PAGE_SIZE - to_bitmap.size(), to_bitmap.data(), to_bitmap.size());
                        to_num_rows = 0;
                        to_data.clear();
                        to_bitmap.clear();
                        pages.push_back(page);
                    };
                    for (size_t j = 0; j < rows; ++j) {
                        int value = fomr_data[j];
                        if (value != NULL_INT32) {
                            if (4 + (to_data.size() + 1) * 4 + (to_num_rows / 8 + 1) > PAGE_SIZE) {
                                gen_page();
                            }
                            set_bitmap(to_bitmap, to_num_rows);
                            to_data.emplace_back(value);
                            ++to_num_rows;
                        } else {
                            if (4 + (to_data.size()) * 4 + (to_num_rows / 8 + 1) > PAGE_SIZE) {
                                gen_page();
                            }
                            unset_bitmap(to_bitmap, to_num_rows);
                            ++to_num_rows;
                        }
                    }
                    if (to_num_rows != 0) {
                        gen_page();
                    }
                    continue;
                } else if (from_column.first==DataType::VARCHAR){
                    varchar_ptr *fomr_data = (varchar_ptr *)from_column.second;
                    uint16_t              num_rows = 0;
                    std::vector<char>     data;
                    std::vector<uint16_t> offsets;
                    std::vector<uint8_t>  bitmap;
                    data.reserve(8192);
                    offsets.reserve(4096);
                    bitmap.reserve(512);
                    auto save_page = [&pages, &num_rows, &data, &offsets, &bitmap]() {
                        auto* page                             = new Page;
                        *reinterpret_cast<uint16_t*>(page->data)     = num_rows;
                        *reinterpret_cast<uint16_t*>(page->data + 2) = static_cast<uint16_t>(offsets.size());
                        memcpy(page->data + 4, offsets.data(), offsets.size() * 2);
                        memcpy(page->data + 4 + offsets.size() * 2, data.data(), data.size());
                        memcpy(page->data + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());
                        num_rows = 0;
                        data.clear();
                        offsets.clear();
                        bitmap.clear();
                        pages.push_back(page);
                    };
                    for (int j = 0; j < rows; ++j) {
                        auto value = fomr_data[j];
                        if (!value.is_null()) {
                            if (value.length() > PAGE_SIZE - 7) {
                                throw std::runtime_error("long string is not support");
                            } else {
                                if (4 + (offsets.size() + 1) * 2 + (data.size() + value.length())
                                        + (num_rows / 8 + 1)
                                    > PAGE_SIZE) {
                                    save_page();
                                }
                                set_bitmap(bitmap, num_rows);
                                data.insert(data.end(), value.string(), value.string() + value.length());
                                offsets.emplace_back(data.size());
                                ++num_rows;
                            }
                        } else {
                            if (4 + offsets.size() * 2 + data.size() + (num_rows / 8 + 1)
                                > PAGE_SIZE) {
                                save_page();
                            }
                            unset_bitmap(bitmap, num_rows);
                            ++num_rows;
                        }
                    }
                    if (num_rows != 0) {
                        save_page();
                    }
                    continue;
                } else {
                    throw std::runtime_error("Unsupported data type");
                }
            }
            found += row_num;
        }

        // 申请shared_.output_的锁...
        {
            std::lock_guard lock(shared_.m_);
            shared_.output_.num_rows += found;
            for(size_t i=0; i<shared_.output_.columns.size(); i++){
                auto& column = shared_.output_.columns[i];
                column.pages.insert(column.pages.end(),
                    page_buffers_[i].begin(), page_buffers_[i].end());
            }
        }

        for (auto& pages : page_buffers_) {
            pages.clear();
        }
    }

};


}