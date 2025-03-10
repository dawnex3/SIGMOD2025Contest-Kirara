#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sys/types.h>
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

namespace Contest {
// #define DEBUG_LOG

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
        probe_matches_ = (uint32_t*)malloc((vec_size_+1)*sizeof(uint32_t));

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
            profile_guard.pause();
            current_barrier->wait([&]() {   // 等待所有线程完成计算，确定哈希表大小
                auto total_found = shared_.found_.load();
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
                if (cont_.num_probe_ == 0) {
                    last_result_.num_rows_ = 0;
#ifdef DEBUG_LOG
                    // 打印该节点输出的总行数
//                    printf("join output rows: %ld, details:\n",total_output);
//                    std::cout<<table_str<<std::endl;
                    std::ofstream log("log_false.txt", std::ios::app);
                    log << "join "<< shared_.get_operator_id()-1 <<" output rows: " << total_output << ", details:\n";
                    log << table_str <<std::endl;
                    log.close();
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
            profile_guard.add_output_row_count(n);
#ifdef DEBUG_LOG
            total_output += n;
            table_str.append(last_result_.toString(false));
#endif
            return last_result_;
        }
    }


    // 计算一列的哈希值，并存储到指定位置。该列必须为INT32
    template <bool RestoreColumn>   //可选：将列值本身也存储到指定位置
    void calculateColHash(OperatorResultTable::ColumnVariant input_column, size_t n, uint8_t* hash_target, size_t hast_step, uint8_t* col_target=nullptr, size_t col_step=0){
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, OperatorResultTable::InstantiatedColumn>) {
                // 已实例化的列为 std::pair<DataType, void*>，是一块连续的数组
                assert(arg.first==DataType::INT32);
                const int32_t* base = (int32_t*)arg.second;
                for (size_t i = 0; i < n; ++i) {
                    if(base[i]!=NULL_INT32 && hash_32(base[i])==NULL_HASH){
                        std::cout<<base[i];
                        exit(-1);
                    }
                    assert(base[i]==NULL_INT32 || hash_32(base[i])!=NULL_HASH);
                    *(Hashmap::hash_t *)hash_target=hash_32(base[i]);
                    hash_target += hast_step;
                    if constexpr (RestoreColumn){
                        *(int32_t*)(col_target) = base[i];
                        col_target += col_step;
                    }
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

                        if(key!=NULL_INT32 && hash_32(key)==NULL_HASH){
                            std::cout<<key;
                            exit(-1);
                        }
                        assert(key==NULL_INT32 || hash_32(key)!=NULL_HASH);
                        *(Hashmap::hash_t *)hash_target=hash_32(key);
                        hash_target += hast_step;
                        if constexpr (RestoreColumn){
                            *(int32_t*)(col_target) = key;
                            col_target += col_step;
                        }
                    }

                    n -= end_row-start_row;
                    if(n<=0) break;
                }

            }
        }, input_column);
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
                Page *const * current_page = col->pages.data() + std::get<1>(arg);
                uint32_t offset = probe_matches_[0] + std::get<2>(arg);

                // 支持含null列
                assert(col->type==DataType::INT32);
                for(uint32_t i=0; i<n; i++){
                    // 定位到probe_matches_[i]所在的Page，和页内偏移
                    while(offset >= getRowCount(*current_page)){
                        offset -= getRowCount(*current_page);
                        current_page++;
                    }
                    const uint8_t* bitmap = getBitmap(*current_page);

                    // 取出数据，进行比较
                    if(!isNotNull(bitmap, offset)){
                        offset += probe_matches_[i+1] - probe_matches_[i];
                        continue;
                    }
                    const int32_t* data_ptr = getPageData<int32_t>(*current_page) + getNonNullCount(bitmap, offset);
                    int32_t right_key = *data_ptr;
                    int32_t left_key = *(int32_t*)((uint8_t*)build_matches_[i] + key_offset);
                    if(left_key == right_key){
                        build_matches_[found] = build_matches_[i];
                        probe_matches_[found] = probe_matches_[i];
                        found++;
                    }

                    offset += probe_matches_[i+1] - probe_matches_[i];
                }
            }
        }, right_col);

        return found;
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
            cont_.next_probe_++;    // 如果cont_.last_chain_==null，说明刚刚处理了上次未处理完的哈希链。此时上次的哈希链表已经处理完成了，next_probe_推进一行。
        }
        for (size_t i = cont_.next_probe_, end = cont_.num_probe_; i < end; i++) {
            Hashmap::hash_t hash = probe_hashes_[i];
            for (auto entry = shared_.hashmap_.find_chain_tagged(hash); entry != nullptr; entry = entry->next) {
                if (entry->hash == hash) {
                    build_matches_[found] = entry;              // 记录左表（哈希表）匹配的EntryHeader
                    probe_matches_[found] = i;  // 记录右表匹配的行号
                    found ++;
                    if (found == vec_size_) {
                        // 缓冲已满，保存状态等待下次调用
                        if(entry->next != nullptr){ //如果本次的哈希链没有处理完毕
                            cont_.last_chain_ = entry->next;
                            cont_.probe_hash_ = hash;
                            cont_.next_probe_ = i;
                            return vec_size_;
                        } else if (i + 1 < end){    //本次哈希链已处理完，但是后面还有待probe的hash
                            cont_.last_chain_ = nullptr;
                            cont_.next_probe_ = i+1;
                            return vec_size_;
                        }
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
    std::vector<std::unique_ptr<TempPage>> unfilled_page_;   // 未满的 Page

    ResultWriter(Shared &shared, std::unique_ptr<Operator> child)
        : shared_{shared}, page_buffers_(shared.output_.columns.size()),
          child_(std::move(child)),
          unfilled_page_() {
            for (size_t i = 0; i < shared.output_.columns.size(); ++i) {
                if (shared.output_.columns[i].type == DataType::INT32) {
                    unfilled_page_.emplace_back(std::make_unique<TempIntPage>());
                } else if (shared.output_.columns[i].type == DataType::VARCHAR) {
                    unfilled_page_.emplace_back(std::make_unique<TempStringPage>());
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
                std::vector<Page*>& pages = page_buffers_[i];

                if(from_column.first==DataType::INT32){
                    TempIntPage *temp = (TempIntPage *)unfilled_page_[i].get();
                    int32_t *form_data = (int32_t*)from_column.second;

                    for (size_t j = 0; j < rows; ++j) {
                        if (temp->is_full()) {
                            pages.emplace_back(temp->dump_page());
                        }
                        temp->add_row(form_data[j]);
                    }
                    continue;
                } else if (from_column.first==DataType::VARCHAR){
                    TempStringPage *temp = (TempStringPage *)unfilled_page_[i].get();
                    varchar_ptr *from_data = (varchar_ptr *)from_column.second;

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
            auto temp = unfilled_page_[i].get();
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