#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sys/types.h>
#include <utility>
#include <variant>
#include <vector>
#include <iostream>
#include <fstream>
#include "Profiler.hpp"
#include "SharedState.hpp"
#include "attribute.h"
#include "statement.h"
#include "plan.h"
#include "HashMap.hpp"
#include "Barrier.hpp"
#include "DataStructure.hpp"
#include "MemoryPool.hpp"

namespace Contest {
//#define DEBUG_LOG

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
        ProfileGuard profile_guard(global_profiler, "Scan_" + std::to_string(shared_.get_operator_id()));

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
                    if(page_rows==LONG_STRING_START){
                        // 如果是LONG STRING的起始页
                        page_rows=1;
                    } else if(page_rows==LOGNG_STRING_FOLLOW){
                        // 如果是LONG STRING的后续页
                        page_rows=0;
                    }
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
            profile_guard.add_input_row_count(last_result_.num_rows_);
            profile_guard.add_output_row_count(last_result_.num_rows_);

            return last_result_;
        }
    }
};


class Naivejoin: public Operator {
private:
    uint32_t vec_size_;        // 批次大小

    std::vector<const Column*> columns_;
    Operator *multi_;    // 右侧算子（探测侧）
    size_t multi_idx_;
    size_t one_idx_;

    size_t num_probe_ = 0;
    size_t next_probe_ = 0;

    uint8_t *ht_entrys_ = nullptr;
    uint32_t * probe_matches_;
    uint32_t * probe_keys_;

    OperatorResultTable multi_result_;
    OperatorResultTable last_result_;       // 上一次调用next的返回结果。本次调用修改一些参数就可以返回。
    LocalVector<size_t> output_attrs_;      // 需要输出哪些列

    size_t ht_entry_size_ = 0;  // 哈希表一行的大小。哈希表的一行由一个EntryHeader，加上其余要输出的构建侧列组成
    LocalVector<size_t> build_value_offsets_;

public:
    Naivejoin(size_t vec_size, Operator *multi, size_t multi_idx,
        std::vector<const Column*> columns, size_t one_idx,
        const std::vector<std::tuple<size_t, DataType>>& output_attrs)
    : vec_size_(vec_size), multi_(multi), multi_idx_(multi_idx), columns_(std::move(columns)), one_idx_(one_idx){
        // 计算ht_entry_size_和build_value_offsets_
        // 为了满足对齐的需求，并且让整个entry最小，需要对左表的列顺序重新排列，将int32_t放到前面，uint64_t放到后面
        build_value_offsets_.resize(columns_.size());
        for(uint32_t i=0; i<columns_.size(); i++){
            if(columns_[i]->type==DataType::INT32){ // 此处跳过构建侧键值。它存储在EntryHeader里面而不是后面
                build_value_offsets_[i]=ht_entry_size_;
                ht_entry_size_ += 4;
            }
        }

        ht_entry_size_ = (ht_entry_size_ + 7) & ~7;   // 将ht_entry_size_补齐到8倍数
        for(uint32_t i=0; i<columns_.size(); i++){
            if(columns_[i]->type!=DataType::INT32){
                build_value_offsets_[i]=ht_entry_size_;
                ht_entry_size_ += 8;
            }
        }

        probe_matches_ = (uint32_t*)local_allocator.allocate(vec_size_ * sizeof(uint32_t));
        probe_keys_ = (uint32_t*)local_allocator.allocate(vec_size_ * sizeof(uint32_t));

        // 分配结果表last_result_的内存，设置output_attrs_
        for(auto [col_idx, col_type]:output_attrs){
            // 如果构建侧的连接键需要输出，那么换成探测侧的连接键来输出。这样方便收集
            if(col_idx==one_idx_){
                col_idx = multi_idx_ + columns_.size();
            }
            output_attrs_.push_back(col_idx);
            if(col_type==DataType::INT32){
                void* col_buffer = local_allocator.allocate(vec_size*sizeof(int32_t));
                last_result_.columns_.emplace_back(std::make_pair(col_type, col_buffer));
            } else {
                void* col_buffer = local_allocator.allocate(vec_size*sizeof(uint64_t));
                last_result_.columns_.emplace_back(std::make_pair(col_type, col_buffer));
            }
        }
    }

    ~Naivejoin() override = default;

    size_t resultColumnNum() override{
        return last_result_.columns_.size();
    }

    void gatherEntry(OperatorResultTable::InstantiatedColumn output_column, uint32_t n, size_t offset){
        if(output_column.first==DataType::INT32){
            auto* base = (int32_t*)output_column.second;
            std::fill_n(base, n, *(int32_t*)(ht_entrys_ + offset));
        } else if(output_column.first==DataType::VARCHAR){
            auto * base = (uint64_t*)output_column.second;
            std::fill_n(base, n, *(uint64_t*)(ht_entrys_ + offset));
        }
    }

    uint32_t joinAllNaive() {
        size_t found = 0;
        size_t i = next_probe_;
        auto one_line_key = *reinterpret_cast<uint32_t*>(ht_entrys_ + build_value_offsets_[one_idx_]);
        for (;i < num_probe_ && found < vec_size_; i++) {
            uint32_t key = probe_keys_[i];
            if (one_line_key == key) {
                probe_matches_[found] = i;  // 记录右表匹配的行号
                found ++;
//                if (found == vec_size_) {
//                    if (i + 1 < end){    //本次哈希链已处理完，但是后面还有待probe的hash
//                        next_probe_ = i+1;
//                        return vec_size_;
//                    }
//                }
            }
        }
        next_probe_ = i;
        return found;   // 该探测批次处理完了，但是没凑够vec_size_个
    }

    OperatorResultTable next() override {
        if (ht_entrys_ == nullptr) {
            ht_entrys_ = (uint8_t*)local_allocator.allocate(1 * ht_entry_size_);
//            OperatorResultTable::ColumnVariant  left_key = columns_[one_idx_];
//            calculateColHash<false>(left_key, 1, ht_entrys+offsetof(Hashmap::EntryHeader, hash), ht_entrys+offsetof(Hashmap::EntryHeader, key), ht_entry_size_);
            for(int col_idx=0; col_idx< columns_.size(); col_idx++){
                gatherCol<false>(std::tuple<const Column*, uint32_t, uint32_t>{columns_[col_idx], 0, 0},
                    1,ht_entrys_ + build_value_offsets_[col_idx],ht_entry_size_);
            }
        }

        // 探测阶段
        while (true) {
            if (next_probe_ >= num_probe_) {
                multi_result_ = multi_->next();
                next_probe_ = 0;
                num_probe_ = multi_result_.num_rows_;
                if (num_probe_ == 0) {
                    last_result_.num_rows_ = 0;
                    return last_result_;
                }
                gatherCol<false>(multi_result_.columns_[multi_idx_], multi_result_.num_rows_,
                    (uint8_t*)probe_keys_, sizeof(uint32_t));
            }
            uint32_t n;
            n = joinAllNaive();
#ifdef SIMD_SIZE

//            if (typeid(*right_) == typeid(Scan))
//                n = joinAll();
//            else
//                n = joinAllSIMD();
#endif
            if (n == 0) continue;
            // 物化最终结果。将匹配的(Entry*, pos)中，左侧的值收集起来，右侧对应的行的值也收集起来
            last_result_.num_rows_ = n;
            size_t one_column_num = columns_.size();
            for(size_t out_idx=0;out_idx<output_attrs_.size();out_idx++) {
                auto column = std::get<OperatorResultTable::InstantiatedColumn>(last_result_.columns_[out_idx]);
                size_t in_idx = output_attrs_[out_idx];
                if (in_idx < one_column_num) { // 如果是构建表
                    gatherEntry(column, n, build_value_offsets_[in_idx]);
                } else if(in_idx - one_column_num != multi_idx_ ){  // 如果是探测表的非键值侧
                    gatherCol<true>(multi_result_.columns_[in_idx - one_column_num], n,
                        (uint8_t*)column.second, column.first == DataType::INT32 ? 4 : 8, probe_matches_);
                } else {    // 如果是探测表的键值侧
                    OperatorResultTable::ColumnVariant key_col = multi_result_.columns_[in_idx - one_column_num];
                    std::visit([&](auto&& key_col) {
                        using T = std::decay_t<decltype(key_col)>;
                        if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                            // 如果键值列已经实例化
                            gatherCol<true>(key_col, n,
                                (uint8_t*)column.second, column.first == DataType::INT32 ? 4 : 8, probe_matches_);
                        } else {
                            // 如果键值列未实例化，则从probe_hks_中提取
                            gatherCol<true>(std::make_pair(DataType::INT32, (void*)probe_keys_), n,
                                (uint8_t*)column.second, column.first == DataType::INT32 ? 4 : 8, probe_matches_);
                        }
                    }, key_col);
                }
            }

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
        uint32_t probe_key_{0};        // 上次未探测完的键值
        Hashmap::EntryHeader* last_chain_{nullptr};  // 上次未完成的哈希链表

        // 额外的状态，指示循环队列的开始和结束位置。专门用于joinSelParallel
        uint32_t queue_begin_{0};
        uint32_t queue_end_{0};

        size_t num_probe_{0};              // right_result_的大小
        size_t next_probe_{0};             // right_result_中接下来要处理的行

    } cont_;

    Shared& shared_;

    uint32_t vec_size_;        // 批次大小

    Operator *left_;     // 左侧算子（构建侧）
    size_t left_idx_;                    // 左侧键所在的列号
    Operator *right_;    // 右侧算子（探测侧）
    size_t right_idx_;                   // 右侧键所在的列号

    size_t ht_entry_size_;  // 哈希表一行的大小。哈希表的一行由一个EntryHeader，加上其余要输出的构建侧列组成
    LocalVector<size_t> build_value_offsets_;   // 构建侧每列相对于EntryHeader的偏移量

    bool is_build_{false};      // 指示哈希表是否已经构建完毕

    OperatorResultTable right_result_;      // 上次调用right_算子得到的结果
    uint32_t * probe_hashs_;                  // right_result_的键值列的哈希值与键值组合
    uint32_t * probe_keys_;                  // right_result_的键值列的哈希值与键值组合
    Hashmap::EntryHeader** build_matches_;  // 哈希表中匹配的条目
    uint32_t * probe_matches_;              // 构建侧匹配的行的行号

    OperatorResultTable last_result_;       // 上一次调用next的返回结果。本次调用修改一些参数就可以返回。
    LocalVector<size_t> output_attrs_;      // 需要输出哪些列

    LocalVector<std::pair<uint8_t*, size_t>> allocations_;  // 存储多个哈希表条目数组<数组指针，数组大小>


    // 专门用于joinSelParallel的循环队列
    size_t circular_queue_size_{1025};      // 循环队列大小
    uint32_t *queue_probe_;                 // 探测侧的循环队列，存储行号
    Hashmap::EntryHeader** queue_build_;    // 构建侧的循环队列，指向Entry

#ifdef DEBUG_LOG
    size_t      probe_rows_{0};
    size_t      output_rows_{0};
    std::string table_str;
#endif

public:

    Hashjoin(Shared& shared, size_t vec_size, Operator *left, size_t left_idx,
        Operator *right, size_t right_idx,
        const std::vector<std::tuple<size_t, DataType>>& output_attrs, std::vector<std::tuple<size_t, DataType>> left_attrs)
    : shared_(shared), vec_size_(vec_size), left_(left), left_idx_(left_idx),
    right_(right), right_idx_(right_idx){
        // 计算ht_entry_size_和build_value_offsets_
        // 为了满足对齐的需求，并且让整个entry最小，需要对左表的列顺序重新排列，将int32_t放到前面，uint64_t放到后面
        build_value_offsets_.resize(left_attrs.size());
        ht_entry_size_ = (sizeof(Hashmap::EntryHeader) + 3) & ~3;   // 将EntryHeader大小补齐到4倍数
        for(uint32_t i=0; i<left_attrs.size(); i++){
            if(std::get<1>(left_attrs[i])==DataType::INT32 && i!=left_idx){ // 此处跳过构建侧键值。它存储在EntryHeader里面而不是后面
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
        probe_hashs_   = (uint32_t *)local_allocator.allocate(vec_size_*sizeof(uint32_t));
        probe_keys_    = (uint32_t *)local_allocator.allocate(vec_size_*sizeof(uint32_t));
        build_matches_ = (Hashmap::EntryHeader**)local_allocator.allocate(vec_size_*sizeof(Hashmap::EntryHeader*));
        probe_matches_ = (uint32_t*)local_allocator.allocate((vec_size_+1)*sizeof(uint32_t));

        // 分配结果表last_result_的内存，设置output_attrs_
        for(auto [col_idx, col_type]:output_attrs){
            // 如果构建侧的连接键需要输出，那么换成探测侧的连接键来输出。这样方便收集
            if(col_idx==left_idx){
                col_idx = right_idx + left_attrs.size();
            }
            output_attrs_.push_back(col_idx);
            if(col_type==DataType::INT32){
                void* col_buffer = local_allocator.allocate(vec_size*sizeof(int32_t));
                last_result_.columns_.emplace_back(std::make_pair(col_type, col_buffer));
            } else {
                void* col_buffer = local_allocator.allocate(vec_size*sizeof(uint64_t));
                last_result_.columns_.emplace_back(std::make_pair(col_type, col_buffer));
            }
        }

        // 分配循环队列的内存
        queue_probe_ = (uint32_t*)local_allocator.allocate(circular_queue_size_*sizeof(uint32_t));
        queue_build_ = (Hashmap::EntryHeader**)local_allocator.allocate(circular_queue_size_*sizeof(Hashmap::EntryHeader*));
    }

    ~Hashjoin() override {}

    size_t resultColumnNum() override{
        return last_result_.columns_.size();
    }

    OperatorResultTable next() override{
        ProfileGuard profile_guard(global_profiler, "HashJoin_" + std::to_string(shared_.get_operator_id()));

        if (!is_build_) {     // 构建哈希表
            size_t found = 0;
            while (true){
                profile_guard.pause();
                OperatorResultTable left_table = left_->next(); // 调用左算子
                profile_guard.resume();
                profile_guard.add_input_row_count( left_table.num_rows_);
                profile_guard.add_output_row_count( left_table.num_rows_);
                size_t n = left_table.num_rows_;
                if(n==0) break;
                found += n;

                // 申请一块n*ht_entry_size_大小的内存，存放本批次的哈希表条目
                uint8_t *ht_entrys = (uint8_t*)local_allocator.allocate(n * ht_entry_size_);
                allocations_.emplace_back(ht_entrys, n);

                // 从数据源OperatorResultTable取出键值，计算键的哈希值并与键本身一起存储到EntryHeader当中。
                OperatorResultTable::ColumnVariant  left_key = left_table.columns_[left_idx_];
                calculateColHash<false>(left_key, n, ht_entrys+offsetof(Hashmap::EntryHeader, hash), ht_entrys+offsetof(Hashmap::EntryHeader, key), ht_entry_size_);

                // 将构建侧的其他列存储到EntryHeader后面。
                for(int col_idx=0; col_idx<left_table.columns_.size(); col_idx++){
                    if(col_idx==left_idx_) continue;
                    OperatorResultTable::ColumnVariant left_value = left_table.columns_[col_idx];
                    gatherCol<false>(left_value,n,ht_entrys+build_value_offsets_[col_idx],ht_entry_size_);
                }
            }


            shared_.found_.fetch_add(found);   // 加到所有线程的总found上
            profile_guard.pause();
            current_barrier->wait([&]() {   // 等待所有线程完成计算，确定哈希表大小
                auto total_found = shared_.found_.load();
#ifdef DEBUG_LOG
                printf("join %zu: build_rows=%lu\n",shared_.get_operator_id()-1,total_found);
#endif
                if (total_found) shared_.hashmap_.setSize(total_found);
            });
            profile_guard.resume();
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
            profile_guard.pause();
            current_barrier->wait(); // 等待所有线程插入哈希表
            profile_guard.resume();
        }

        // 探测阶段
        while (true) {
            if (cont_.next_probe_ >= cont_.num_probe_) {
                profile_guard.pause();
                right_result_ = right_->next();     // 调用右侧算子，结果保存到right_result_
                profile_guard.resume();
                profile_guard.add_input_row_count(right_result_.num_rows_);
                cont_.next_probe_ = 0;
                cont_.num_probe_ = right_result_.num_rows_;
#ifdef DEBUG_LOG
                probe_rows_ += cont_.num_probe_;
#endif
                if (cont_.num_probe_ == 0) {
                    last_result_.num_rows_ = 0;
#ifdef DEBUG_LOG
                    printf("join %zu: output_rows=%ld, probe_rows=%ld\n",shared_.get_operator_id()-1,output_rows_,probe_rows_);
//                    std::cout<<table_str<<std::endl;
//                    std::ofstream log("log_false.txt", std::ios::app);
//                    log << "join "<< shared_.get_operator_id()-1 <<" output rows: " << output_rows_ << ", details:\n";
//                    log << table_str <<std::endl;
//                    log.close();
#endif
                    return last_result_;
                }
                // 计算right_result_中键的哈希值，存储到probe_hks_数组中
                calculateColHash<true>(right_result_.columns_[right_idx_], right_result_.num_rows_,
                    (uint8_t *)probe_hashs_, (uint8_t*)probe_keys_, sizeof(uint32_t));
            }
            // 调用探测函数，检测哈希值和键值都相等的(Entry*, pos)对，分别存储在build_matches_和probe_matches_中
            uint32_t n;
            n = joinAll();
//#ifdef SIMD_SIZE
//            if (typeid(*right_) == typeid(Scan))
//                n = joinAll();
//            else
//                n = joinAllSIMD();
//#else
//            n = joinAll();
//#endif
            if (n == 0) continue;
            // 物化最终结果。将匹配的(Entry*, pos)中，左侧的值收集起来，右侧对应的行的值也收集起来
            last_result_.num_rows_ = n;
            size_t left_column_num = left_->resultColumnNum();
            for(size_t out_idx=0;out_idx<output_attrs_.size();out_idx++) {
                auto column = std::get<OperatorResultTable::InstantiatedColumn>(last_result_.columns_[out_idx]);
                size_t in_idx = output_attrs_[out_idx];
                if (in_idx < left_column_num) { // 如果是构建表
                    gatherEntry(column, n, build_value_offsets_[in_idx]);
                } else if(in_idx - left_column_num != right_idx_ ){  // 如果是探测表的非键值侧
                    gatherCol<true>(right_result_.columns_[in_idx - left_column_num], n,
                        (uint8_t*)column.second, column.first == DataType::INT32 ? 4 : 8, probe_matches_);
                } else {    // 如果是探测表的键值侧
                    OperatorResultTable::ColumnVariant key_col = right_result_.columns_[in_idx - left_column_num];
                    std::visit([&](auto&& key_col) {
                        using T = std::decay_t<decltype(key_col)>;
                        if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                            // 如果键值列已经实例化
                            gatherCol<true>(key_col, n,
                                (uint8_t*)column.second, column.first == DataType::INT32 ? 4 : 8, probe_matches_);
                        } else {
                            // 如果键值列未实例化，则从probe_hks_中提取
                            gatherCol<true>(std::make_pair(DataType::INT32, (void*)probe_keys_), n,
                                (uint8_t*)column.second, column.first == DataType::INT32 ? 4 : 8, probe_matches_);
                        }
                    }, key_col);
                }
            }

            profile_guard.add_output_row_count(n);
#ifdef DEBUG_LOG
            output_rows_ += n;
//            table_str.append(last_result_.toString(false));
#endif
            return last_result_;
        }
    }

#ifdef SIMD_SIZE
    // 计算一列的哈希值，与该列本身一起组成uint64，存储到指定位
    template <bool targetDense>
    void calculateColHash(OperatorResultTable::ColumnVariant input_column, size_t n, uint8_t* target, uint8_t* target_keys, size_t step){
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                // 已实例化的列为 std::pair<DataType, void*>，是一块连续的数组
                assert(arg.first == DataType::INT32);
                const int32_t* base = static_cast<int32_t*>(arg.second);

                size_t i = 0;
                while (reinterpret_cast<uintptr_t>(base + i) % 32 != 0){
                    *(uint32_t *)target = hash_32(base[i]);
                    *(uint32_t *)target_keys = base[i];
                    target += step;
                    target_keys += step;
                    i++;
                }

                if constexpr (targetDense){
                    for (; i + SIMD_SIZE-1 < n; i += SIMD_SIZE) {
                        compute_hashes(reinterpret_cast<const uint32_t*>(base + i), reinterpret_cast<uint32_t*>(target));
                        *reinterpret_cast<v8u32*>(target_keys) = *reinterpret_cast<const v8u32*>(base + i);
                        target += sizeof(v8u32);
                        target_keys += sizeof(v8u32);
                    }
                } else {
                    for (; i + SIMD_SIZE-1 < n; i += SIMD_SIZE) {
                        alignas(32) uint32_t hash_arr[SIMD_SIZE];
                        compute_hashes(reinterpret_cast<const uint32_t*>(base + i), hash_arr);
                        for (int j = 0; j < SIMD_SIZE; ++j) {
                            *(uint32_t*)target = hash_arr[j];
                            *(uint32_t*)(target_keys) = base[i + j];
                            target += step;
                            target_keys += step;
                        }
                    }
                }

                // 处理剩余元素（标量处理）
                for (; i < n; ++i) {
                    uint32_t hash = hash_32(base[i]);
                    *(uint32_t*)target = hash;
                    *(uint32_t*)(target_keys) = base[i];
                    target += step;
                    target_keys += step;
                }
            } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn>) {
                // 连续未实例化的列为 std::tuple<Column*, uint32_t, uint32_t>，它存放在多个Page中。
                const Column* col = std::get<0>(arg);
                assert(col->type==DataType::INT32);
                size_t processed = 0; 
                size_t remaining = n;  

                for (size_t i = std::get<1>(arg); i < col->pages.size(); i++) {
                    const Page* current_page = col->pages[i];
                    size_t start_row = (i == std::get<1>(arg)) ? std::get<2>(arg) : 0;
                    size_t end_row = std::min((size_t)getRowCount(current_page), start_row + remaining);
                    if (true || __glibc_unlikely(getNonNullCount(current_page) != getRowCount(current_page))){
//                        printf("unlikely! %d %d\n", getNonNullCount(current_page), getRowCount(current_page));
                        const uint8_t* bitmap = getBitmap(current_page);
                        const int32_t* base = getPageData<int32_t>(current_page) + getNonNullCount(bitmap, start_row);
                        size_t j = start_row;
                        while (j < end_row && processed < n) { // 这个8不是SIMD_SIZE，而是byte / bit
                            if ((j % 8 == 0) && (j + 8 <= end_row) && (processed + 8 <= n)) {
                                size_t byte_idx = j / 8;
                                uint8_t bitmap_byte = bitmap[byte_idx];
                                int number_of_one = __builtin_popcount(bitmap_byte);
                                v8u32   keys;

                                if (number_of_one == 8) {
                                    memcpy(&keys, base, sizeof(keys));
                                    base += 8;
                                } else {
                                    // 步骤1：将位图转换为有效位置掩码
                                    for (int k = 0; k < 8; k++) {
                                        if (isNotNull(bitmap, j + k)) {
                                            keys[k] = *base++;
                                        } else {
                                            keys[k] = static_cast<uint32_t>(NULL_INT32);
                                        }
                                    }
                                }

                                // 将向量存储到临时数组
                                if constexpr (targetDense){
                                    alignas(32) uint32_t hash_arr[SIMD_SIZE];
                                    compute_hashes(reinterpret_cast<const uint32_t*>(&keys), hash_arr);
                                    memcpy(target_keys, &keys, sizeof(keys));
                                    memcpy(target, hash_arr, sizeof(hash_arr));
                                    target += sizeof(v8u32);
                                    target_keys += sizeof(v8u32);
                                } else {
                                    alignas(32) uint32_t hash_arr[SIMD_SIZE];
                                    compute_hashes(reinterpret_cast<const uint32_t*>(&keys), hash_arr);
                                    for (int k = 0; k < SIMD_SIZE; ++k) {
                                        *(uint32_t*)target = hash_arr[k];
                                        *(uint32_t*)(target_keys) = keys[k];
                                        target += step;
                                        target_keys += step;
                                    }
                                }
                                processed += 8;
                                j += 8;
                            } else {
                                int32_t key = NULL_INT32;
                                if (isNotNull(bitmap, j)) {
                                    key = *base;
                                    base++;
                                }
                                uint32_t hash = hash_32(key);
                                *(uint32_t*)target = hash;
                                *(uint32_t*)(target_keys) = key;
                                target += step;
                                target_keys += step;
                                processed++;
                                j++;
                            }
                        }
                    } else {
                        const int32_t* base = getPageData<int32_t>(current_page) + start_row;
                        size_t j = start_row;
                        while (j < end_row && processed < n) {
                            if((j % 8 == 0) && (j + 8 <= end_row) && (processed + 8 <= n)){
                                v8u32 keys;
                                memcpy(&keys, base, sizeof(keys));
                                base += 8;

                                // 将向量存储到临时数组
                                if (targetDense){
                                    alignas(32) uint32_t hash_arr[SIMD_SIZE];
                                    compute_hashes(reinterpret_cast<const uint32_t*>(&keys), hash_arr);
                                    memcpy(target_keys, &keys, sizeof(keys));
                                    memcpy(target, hash_arr, sizeof(hash_arr));
                                    target += sizeof(v8u32);
                                    target_keys += sizeof(v8u32);
                                } else {
                                    alignas(32) uint32_t hash_arr[SIMD_SIZE];
                                    compute_hashes(reinterpret_cast<const uint32_t*>(&keys), hash_arr);
                                    for (int k = 0; k < SIMD_SIZE; ++k) {
                                        *(uint32_t*)target = hash_arr[k];
                                        *(uint32_t*)(target_keys) = keys[k];
                                        target += step;
                                        target_keys += step;
                                    }
                                }
                                processed += 8;
                                j += 8;
                            }else{
                                int32_t key = *base++;
                                uint32_t hash = hash_32(key);
                                *(uint32_t*)target = hash;
                                *(uint32_t*)(target_keys) = key;
                                target += step;
                                target_keys += step;
                                processed++;
                                j++;
                            }
                        }
                    }
                    remaining = n - processed;
                    if (remaining <= 0) break;
                }
            }
        }, input_column);
    }
#else
    // 计算一列的哈希值，与该列本身一起组成uint64，存储到指定位置
    template <bool targetDense>
    void calculateColHash(OperatorResultTable::ColumnVariant input_column, size_t n, uint8_t* target, uint8_t* target_keys, size_t step){
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                // 已实例化的列为 std::pair<DataType, void*>，是一块连续的数组
                assert(arg.first==DataType::INT32);
                const int32_t* base = (int32_t*)arg.second;
                for (size_t i = 0; i < n; ++i) {
                    *(uint32_t *)target= hash_32(base[i]);
                    *(uint32_t *)target_keys = base[i];
                    target += step;
                    target_keys += step;
                }
            } else if constexpr (std::is_same_v<T, OperatorResultTable::ContinuousColumn>) {
                // 连续未实例化的列为 std::tuple<Column*, uint32_t, uint32_t>，它存放在多个Page中。
                const Column* col = std::get<0>(arg);
                assert(col->type==DataType::INT32);

                for(size_t i=std::get<1>(arg); i<col->pages.size(); i++){
                    const Page* current_page = col->pages[i];                       // 要读取的页面
                    const uint8_t* bitmap = getBitmap(current_page);                // 要读取页面的位图
                    size_t start_row = i==std::get<1>(arg) ? std::get<2>(arg) : 0;  // 本页的起始行
                    size_t end_row = std::min((size_t)getRowCount(current_page), n + start_row);  // 本页的终止行
                    const int32_t* base = getPageData<int32_t>(current_page) + getNonNullCount(bitmap, start_row);

                    for (size_t j=start_row; j<end_row; j++) {
                        int32_t key=NULL_INT32;
                        if (isNotNull(bitmap, j)) {
                            key=*base;
                            base++;
                        }

                        uint32_t hash = hash_32(key);
                        *(uint32_t *)target=hash;
                        *(uint32_t *)target_keys=key;
                        target += step;
                        target_keys += step;
                    }
                    n -= end_row-start_row;
                    if(n<=0) break;
                }

            }
        }, input_column);
    }
#endif
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


    // 匹配哈希值相等的Entry和右侧行，存入buildMatches和probeMatches
    uint32_t joinAll(){
        size_t found = 0;
        // 处理上次未完成的哈希链表
        if(cont_.last_chain_!= nullptr){
            for (Hashmap::EntryHeader* entry = cont_.last_chain_; entry != nullptr; entry = entry->next) {
                if (entry->key == cont_.probe_key_) {
                    build_matches_[found] = entry;              // 记录左表（哈希表）匹配的EntryHeader
                    probe_matches_[found] = cont_.next_probe_;  // 记录右表匹配的行号
                    found ++;
                    if (__glibc_unlikely(found == vec_size_)) {   // 本批次已满，保存状态交给下一轮处理
                        cont_.last_chain_ = entry->next;
                        if(cont_.last_chain_==nullptr){
                            cont_.next_probe_++;    // 如果本次的哈希链其实已经处理完毕
                        }

                        return vec_size_;
                    }
                }
            }
            cont_.next_probe_++;
        }

        for (size_t i = cont_.next_probe_, end = cont_.num_probe_; i < end; i++) {
            uint32_t hash = probe_hashs_[i];
            uint32_t key = probe_keys_[i];
            auto tmp = 0;
            for (auto entry = shared_.hashmap_.find_chain_tagged(hash); entry != nullptr; entry = entry->next) {
                tmp++;
                if (entry->key == key) {
                    build_matches_[found] = entry;              // 记录左表（哈希表）匹配的EntryHeader
                    probe_matches_[found] = i;  // 记录右表匹配的行号
                    found ++;
                    if (found == vec_size_) {
                        // 缓冲已满，保存状态等待下次调用
                        if(entry->next != nullptr){ //如果本次的哈希链没有处理完毕
                            cont_.last_chain_ = entry->next;
                            cont_.probe_key_ = key;
                            cont_.next_probe_ = i;
//                            printf("%d\n", tmp);
                            return vec_size_;
                        } else if (i + 1 < end){    //本次哈希链已处理完，但是后面还有待probe的hash
                            cont_.last_chain_ = nullptr;
                            cont_.next_probe_ = i+1;
//                            printf("%d\n", tmp);
                            return vec_size_;
                        }
                    }
                }
            }
//            printf("%d\n", tmp);
        }
        cont_.last_chain_ = nullptr;
        cont_.next_probe_ = cont_.num_probe_;
        return found;   // 该探测批次处理完了，但是没凑够vec_size_个
    }

#ifdef SIMD_SIZE
    uint32_t joinAllSIMD() {
        size_t found = 0;
        auto followup = cont_.queue_begin_;
        auto followupWrite = cont_.queue_end_;

        // 只有当没有等待处理的followup时，执行SIMD join
        if (followup == followupWrite) {
            size_t i = 0;
            for (; i + 8 < cont_.num_probe_; i += 8) {
                // --- 1. 加载 probe_hashes_ ---
                // 使用向量扩展直接加载8个32位探测哈希值
                v8u32 hashDense = *reinterpret_cast<const v8u32*>(probe_hashs_ + i);

                // --- 2. 在哈希表中查找对应链 ---
                auto entries = shared_.hashmap_.find_chain_tagged(hashDense);
//                for (int k = 0 ; k < 8; ++k){
//                    if (entries[k] != reinterpret_cast<uint64_t>(shared_.hashmap_.find_chain_tagged(hashDense[k])))
//                        printf("ERROR!\n");
//                }

                // --- 3. 加载哈希表中对应条目的哈希值 ---
                // 每个条目的哈希存放在 EntryHeader 的某个偏移处
                v8u32 entry_keys;
                for (int k = 0; k < 8; ++k) {
                    // 如果该条目有效则加载其哈希，否则设为0（或其他不匹配的值）
                    if (entries[k]) {
                        auto entry = reinterpret_cast<const Hashmap::EntryHeader*>(entries[k]);
                        entry_keys[k] = entry->key;
                    } else {
                        entry_keys[k] = 0x80000000U; // 破罐子破摔
                    }
                }

                // --- 4. 比较探测哈希与条目哈希 ---
                // 利用向量扩展的逐元素比较
                auto cmp = (entry_keys == *reinterpret_cast<v8u32*>(probe_keys_ + i));

                // --- 5. 将匹配结果压缩存储到 build_matches_ 和 probe_matches_ ---
                // 这里采用循环，将满足条件的元素写入结果数组
                for (int k = 0; k < 8; ++k) {
                    if (entries[k] && cmp[k] == 0xffffffffU) {
                        build_matches_[found] = reinterpret_cast<Hashmap::EntryHeader*>(entries[k]);
                        probe_matches_[found] = i + k;
                        ++found;
                    }
                }

                // --- 6. 处理链上的后续匹配（冲突链） ---
                // 读取每个条目的 next 指针
                v8u64 nextPtrs;
                for (int k = 0; k < 8; ++k) {
                    if (entries[k]) {
                        auto entry = reinterpret_cast<const Hashmap::EntryHeader*>(entries[k]);
                        nextPtrs[k] = (uint64_t)entry->next;
                    } else {
                        nextPtrs[k] = (uint64_t)Contest::Hashmap::end(); // nullptr
                    }
                }

                // 将存在后续链的条目压缩存入 followupEntries 和 followupIds
                for (int k = 0; k < 8; ++k) {
                    if (nextPtrs[k] != (uint64_t)Contest::Hashmap::end()) {
                        queue_build_[followupWrite] = reinterpret_cast<Hashmap::EntryHeader*>(nextPtrs[k]);
                        queue_probe_[followupWrite] = i + k;
                        ++followupWrite;
                    }
                }
            } // for i
            for (; i < cont_.num_probe_; ++i) {
                auto hash = probe_hashs_[i];
                auto entry = shared_.hashmap_.find_chain_tagged(hash);
                if (entry != Contest::Hashmap::end()) {
                    if (entry->hash == hash) {
                       build_matches_[found] = entry;
                       probe_matches_[found] = i;
                       found += 1;
                    }
                    if (entry->next != Contest::Hashmap::end()) {
                       queue_probe_[followupWrite] = i;
                       queue_build_[followupWrite] = entry->next;
                       followupWrite = followupWrite == circular_queue_size_ - 1 ? 0 : followupWrite + 1;
                    }
                }
            }
        }
        
        while (followup != followupWrite) {
            auto remainingSpace = vec_size_ - found;
            auto nrFollowups = followup <= followupWrite
                                 ? followupWrite - followup
                                 : circular_queue_size_ - (followup - followupWrite);
            auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);
            for (size_t j = 0; j < fittingElements; ++j) {
                size_t i = queue_probe_[followup];
                auto entry = queue_build_[followup];
                followup = (followup + 1);
                if (followup == circular_queue_size_) followup = 0;
                auto hash = probe_hashs_[i];
                if (entry->hash == hash) {
                    build_matches_[found] = entry;
                    probe_matches_[found] = i;
                    found++;
                }
                if (entry->next != Contest::Hashmap::end()) {
                    queue_probe_[followupWrite] = i;
                    queue_build_[followupWrite] = entry->next;
                    followupWrite = (followupWrite + 1) % circular_queue_size_;
                }
            }
            if (fittingElements < nrFollowups) {
             // continuation
             cont_.queue_end_ = followupWrite;
             cont_.queue_begin_ = followup;
             return found;
            }
        }
        cont_.next_probe_ = cont_.num_probe_;
        cont_.queue_begin_ = 0;
        cont_.queue_end_ = 0;
        
        return found;
    }
#endif

    /// computes join result into build_matches_ and probe_matches_
    /// Implementation: optimized for long CPU pipelines
//    uint32_t joinAllParallel(){
//        size_t found = 0;
//        auto followup = cont_.queue_begin_;   // 队列读指针
//        auto followupWrite = cont_.queue_end_;   // 队列写指针
//
//        if (followup == followupWrite) { // 当循环队列为空的时候
//            for (size_t i = 0, end = cont_.num_probe_; i < end; ++i) {    // 遍历该批次所有探测元组
//                auto hk = probe_hks_[i];
//                auto entry = shared_.hashmap_.find_chain_tagged(hk>>32);
//                if (entry != nullptr) {        // 匹配项添加到buildMatches和probeMatches
//                    if (entry->hash_and_key == hk) {
//                        build_matches_[found] = entry;
//                        probe_matches_[found] = i;
//                        found++;
//                    }
//                    if (entry->next != nullptr) {  // 添加id-entry对到循环队列
//                        queue_probe_[followupWrite] = i;
//                        queue_build_[followupWrite] = entry->next;
//                        followupWrite = followupWrite == circular_queue_size_ - 1 ? 0 : followupWrite + 1;
//                        // 这里保证不会绕好几圈，以至覆盖了先前写入的值
//                    }
//                }
//            }
//        }
//
//        while (followup != followupWrite) {          // 消耗直至队列为空
//            auto remainingSpace = vec_size_ - found;  // 该批次的剩余容量
//            auto nrFollowups = followup <= followupWrite
//                                    ? followupWrite - followup
//                                    : circular_queue_size_ - (followup - followupWrite);  // 队列的大小
//            // std::cout << "nrFollowups: " << nrFollowups << "\n";
//            auto fittingElements = std::min((size_t)nrFollowups, remainingSpace);   // fittingElements是所能处理的最大数目
//            for (size_t j = 0; j < fittingElements; ++j) {
//                size_t i = queue_probe_[followup];
//                auto entry = queue_build_[followup];
//                followup = followup == circular_queue_size_ - 1 ? 0 : (followup + 1);
//                auto hk = probe_hks_[i];      // 取出队列最前端
//                if (entry->hash_and_key == hk) {
//                    build_matches_[found] = entry;
//                    probe_matches_[found] = i;
//                    found++;
//                }
//                if (entry->next != nullptr) {  // 向队列尾部添加元素
//                    queue_probe_[followupWrite] = i;
//                    queue_build_[followupWrite] = entry->next;
//                    followupWrite = followupWrite == circular_queue_size_ - 1 ? 0 : (followupWrite + 1);
//                }
//            }
//            if (fittingElements < nrFollowups) {   // 当remainingSpace < nrFollowups时，会在此提前结束该批次。尽管批次可能未满。
//                // continuation
//                cont_.queue_end_ = followupWrite;
//                cont_.queue_begin_ = followup;
//                return found;
//            }
//        }
//        cont_.next_probe_ = cont_.num_probe_;
//        cont_.queue_end_ = 0;
//        cont_.queue_begin_ = 0;
//        return found;
//    }

    // join implementation after Peter's suggestions
    uint32_t joinBoncz();
    /// computes join result into build_matches_ and probe_matches_
    /// Implementation: Using AVX 512 SIMD
//    uint32_t joinAllSIMD();
    /// computes join result into build_matches_ and probe_matches_, respecting
    /// selection vector probeSel for probe side
    uint32_t joinSel();
    /// computes join result into build_matches_ and probe_matches_, respecting
    /// selection vector probeSel for probe side
    /// Implementation: optimized for long CPU pipelines
    uint32_t joinSelParallel();
    /// computes join result into build_matches_ and probe_matches_, respecting
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
        explicit Shared(const std::vector<std::tuple<size_t, DataType>>& output_attrs){
            // 根据output_attrs构建output_
            for(auto [_, col_type]: output_attrs){
                output_.columns.emplace_back(col_type);
            }
        }
    };

    Shared& shared_;

    Operator *child_;   // 子算子。一般来说是个JOIN

    LocalVector<LocalVector<Page*>> page_buffers_;   // 暂时存储的每一列的page
    LocalVector<TempPage *> unfilled_page_;   // 未满的 Page

    ResultWriter(Shared &shared, Operator *child)
        : shared_{shared}, page_buffers_(shared.output_.columns.size()),
          child_(child),
          unfilled_page_() {
            for (auto & column : shared.output_.columns) {
                if (column.type == DataType::INT32) {
                    unfilled_page_.push_back(new (local_allocator.allocate(sizeof (TempIntPage))) TempIntPage);
                } else if (column.type == DataType::VARCHAR) {
                    unfilled_page_.push_back(new (local_allocator.allocate(sizeof (TempStringPage))) TempStringPage);
                } else {
                    throw std::runtime_error("Unsupported data type");
                }
            }
          }

    void next(){
        size_t found = 0;
        while(true){
            OperatorResultTable child_result = child_->next();
            size_t row_num = child_result.num_rows_;
            ProfileGuard profile(global_profiler, "resultwriter");
            global_profiler->add_input_row_count("resultwriter", row_num);
            global_profiler->add_output_row_count("resultwriter", row_num);
            if(row_num==0) break;

            // 将每一列写入page_buffers_
            for(size_t i=0; i<child_result.columns_.size(); i++){
                auto from_column = std::get<OperatorResultTable::InstantiatedColumn>(child_result.columns_[i]);
                auto rows = child_result.num_rows_;
                LocalVector<Page*>& pages = page_buffers_[i];

                if(from_column.first==DataType::INT32){
                    auto *temp = (TempIntPage *)unfilled_page_[i];
                    auto *form_data = (int32_t*)from_column.second;

                    for (size_t j = 0; j < rows; ++j) {
                        if (temp->is_full()) {
                            pages.emplace_back(temp->dump_page());
                        }
                        temp->add_row(form_data[j]);
                    }
                    continue;
                } else if (from_column.first==DataType::VARCHAR){
                    auto *temp = (TempStringPage *)unfilled_page_[i];
                    auto *from_data = (varchar_ptr *)from_column.second;

                    for (int j = 0; j < rows; ++j) {
                        auto value = from_data[j];
                        if(value.isLongString()){
                            // 先将temp page清空输出
                            if(!temp->is_empty()){
                                pages.emplace_back(temp->dump_page());
                            }
                            uint16_t page_num = value.longStringPageNum();  // 该Long String有多少页
                            Page* const* page_start = value.longStringPage();
                            for(size_t k=0; k<page_num; k++){
                                Page* copy_page = new Page;
                                memcpy(copy_page->data,page_start[k]->data,PAGE_SIZE);
                                pages.emplace_back(copy_page);
                            }
                        } else {
                            if (!temp->can_store_string(value)) {
                                pages.emplace_back(temp->dump_page());
                            }
                            temp->add_row(value);
                        }
                    }
                    continue;
                } else {
                    throw std::runtime_error("Unsupported data type");
                }
            }
            found += row_num;
        }

        for (int i = 0; i < shared_.output_.columns.size(); ++i) {
            auto temp = unfilled_page_[i];
            if (!temp->is_empty()) {
                page_buffers_[i].emplace_back(temp->dump_page());
            }
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