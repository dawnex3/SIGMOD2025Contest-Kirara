#include <algorithm>
#include <chrono>
#include <plan.h>
#include <table.h>
#include <thread>
#include "Profiler.hpp"
#include "SharedState.hpp"
#include "Operator.hpp"
#include "Barrier.hpp"

namespace Contest {


using ExecuteResult = std::vector<std::vector<Data>>;
using namespace std::chrono;

ExecuteResult execute_impl(const Plan& plan, size_t node_idx);

struct JoinAlgorithm {
    bool                                             build_left;
    ExecuteResult&                                   left;
    ExecuteResult&                                   right;
    ExecuteResult&                                   results;
    size_t                                           left_col, right_col;
    const std::vector<std::tuple<size_t, DataType>>& output_attrs;

    template <class T>
    auto run() {
        namespace views = ranges::views;
        std::unordered_map<T, std::vector<size_t>> hash_table;
        if (build_left) {
            for (auto&& [idx, record]: left | views::enumerate) {
                std::visit(
                    [&hash_table, idx = idx](const auto& key) {
                        using Tk = std::decay_t<decltype(key)>;
                        if constexpr (std::is_same_v<Tk, T>) {
                            if (auto itr = hash_table.find(key); itr == hash_table.end()) {
                                hash_table.emplace(key, std::vector<size_t>(1, idx));
                            } else {
                                itr->second.push_back(idx);
                            }
                        } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                            throw std::runtime_error("wrong type of field");
                        }
                    },
                    record[left_col]);
            }
            for (auto& right_record: right) {
                std::visit(
                    [&](const auto& key) {
                        using Tk = std::decay_t<decltype(key)>;
                        if constexpr (std::is_same_v<Tk, T>) {
                            if (auto itr = hash_table.find(key); itr != hash_table.end()) {
                                for (auto left_idx: itr->second) {
                                    auto&             left_record = left[left_idx];
                                    std::vector<Data> new_record;
                                    new_record.reserve(output_attrs.size());
                                    for (auto [col_idx, _]: output_attrs) {
                                        if (col_idx < left_record.size()) {
                                            new_record.emplace_back(left_record[col_idx]);
                                        } else {
                                            new_record.emplace_back(
                                                right_record[col_idx - left_record.size()]);
                                        }
                                    }
                                    results.emplace_back(std::move(new_record));
                                }
                            }
                        } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                            throw std::runtime_error("wrong type of field");
                        }
                    },
                    right_record[right_col]);
            }
        } else {
            for (auto&& [idx, record]: right | views::enumerate) {
                std::visit(
                    [&hash_table, idx = idx](const auto& key) {
                        using Tk = std::decay_t<decltype(key)>;
                        if constexpr (std::is_same_v<Tk, T>) {
                            if (auto itr = hash_table.find(key); itr == hash_table.end()) {
                                hash_table.emplace(key, std::vector<size_t>(1, idx));
                            } else {
                                itr->second.push_back(idx);
                            }
                        } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                            throw std::runtime_error("wrong type of field");
                        }
                    },
                    record[right_col]);
            }
            for (auto& left_record: left) {
                std::visit(
                    [&](const auto& key) {
                        using Tk = std::decay_t<decltype(key)>;
                        if constexpr (std::is_same_v<Tk, T>) {
                            if (auto itr = hash_table.find(key); itr != hash_table.end()) {
                                for (auto right_idx: itr->second) {
                                    auto&             right_record = right[right_idx];
                                    std::vector<Data> new_record;
                                    new_record.reserve(output_attrs.size());
                                    for (auto [col_idx, _]: output_attrs) {
                                        if (col_idx < left_record.size()) {
                                            new_record.emplace_back(left_record[col_idx]);
                                        } else {
                                            new_record.emplace_back(
                                                right_record[col_idx - left_record.size()]);
                                        }
                                    }
                                    results.emplace_back(std::move(new_record));
                                }
                            }
                        } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                            throw std::runtime_error("wrong type of field");
                        }
                    },
                    left_record[left_col]);
            }
        }
    }
};

ExecuteResult execute_hash_join(const Plan&          plan,
    const JoinNode&                                  join,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
    auto                           left_idx    = join.left;
    auto                           right_idx   = join.right;
    auto&                          left_node   = plan.nodes[left_idx];
    auto&                          right_node  = plan.nodes[right_idx];
    auto&                          left_types  = left_node.output_attrs;
    auto&                          right_types = right_node.output_attrs;
    auto                           left        = execute_impl(plan, left_idx);
    auto                           right       = execute_impl(plan, right_idx);
    std::vector<std::vector<Data>> results;

    JoinAlgorithm join_algorithm{.build_left = join.build_left,
        .left                                = left,
        .right                               = right,
        .results                             = results,
        .left_col                            = join.left_attr,
        .right_col                           = join.right_attr,
        .output_attrs                        = output_attrs};
    if (join.build_left) {
        switch (std::get<1>(left_types[join.left_attr])) {
        case DataType::INT32:   join_algorithm.run<int32_t>(); break;
        case DataType::INT64:   join_algorithm.run<int64_t>(); break;
        case DataType::FP64:    join_algorithm.run<double>(); break;
        case DataType::VARCHAR: join_algorithm.run<std::string>(); break;
        }
    } else {
        switch (std::get<1>(right_types[join.right_attr])) {
        case DataType::INT32:   join_algorithm.run<int32_t>(); break;
        case DataType::INT64:   join_algorithm.run<int64_t>(); break;
        case DataType::FP64:    join_algorithm.run<double>(); break;
        case DataType::VARCHAR: join_algorithm.run<std::string>(); break;
        }
    }

    return results;
}

ExecuteResult execute_scan(const Plan&               plan,
    const ScanNode&                                  scan,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
    auto                           table_id = scan.base_table_id;
    auto&                          input    = plan.inputs[table_id];
    auto                           table    = Table::from_columnar(input);
    std::vector<std::vector<Data>> results;
    for (auto [col_idx, _]: output_attrs) {
        printf("scan col %lu\n", col_idx);
    }
    for (auto& record: table.table()) {
        std::vector<Data> new_record;
        new_record.reserve(output_attrs.size());
        for (auto [col_idx, _]: output_attrs) {
            new_record.emplace_back(record[col_idx]);
        }
        results.emplace_back(std::move(new_record));
    }
    return results;
}

ExecuteResult execute_impl(const Plan& plan, size_t node_idx) {
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                return execute_hash_join(plan, value, node.output_attrs);
            } else {
                return execute_scan(plan, value, node.output_attrs);
            }
        },
        node.data);
}

std::unique_ptr<Operator> getOperator(const Plan& plan, size_t node_idx, SharedStateManager& shared_manager, size_t vector_size = 1024){
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value)-> std::unique_ptr<Operator> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                // 如果是join节点
                auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1); //共享状态id设为node_idx+1
                std::unique_ptr<Operator> left_op = getOperator(plan, value.left, shared_manager, vector_size);
                std::unique_ptr<Operator> right_op = getOperator(plan, value.right, shared_manager, vector_size);
                if(value.build_left){   // 左侧构建，正常情况
                    std::unique_ptr<Hashjoin> hash_join = std::make_unique<Hashjoin>(
                        shared, vector_size, std::move(left_op), value.left_attr,
                        std::move(right_op), value.right_attr, node.output_attrs, plan.nodes[value.left].output_attrs);
                    return std::move(hash_join);
                } else {    // 右侧构建，调换算子顺序，以及output_attrs的顺序
                    std::vector<std::tuple<size_t, DataType>> output_attrs;
                    size_t left_size = plan.nodes[value.left].output_attrs.size();
                    size_t right_size = plan.nodes[value.right].output_attrs.size();
                    for(auto [col_idx, col_type]: node.output_attrs){
                        output_attrs.emplace_back(col_idx>=left_size ? col_idx-left_size : col_idx+right_size, col_type);
                    }
                    std::unique_ptr<Hashjoin> hash_join = std::make_unique<Hashjoin>(
                        shared, vector_size, std::move(right_op), value.right_attr,
                        std::move(left_op), value.left_attr, output_attrs, plan.nodes[value.right].output_attrs);
                    return std::move(hash_join);
                }
            } else if constexpr (std::is_same_v<T, ScanNode>){
                // 如果是scan节点
                const ColumnarTable& table = plan.inputs[value.base_table_id];
                auto& shared = shared_manager.get<Scan::Shared>(node_idx + 1); //共享状态id设为node_idx+1
                size_t row_num = table.num_rows;
                // 填充数据源
                std::vector<const Column*> columns;
                for(auto [col_idx, _]: node.output_attrs){
                    columns.push_back(&table.columns[col_idx]);
                }
                std::unique_ptr<Scan> scan = std::make_unique<Scan>(shared,row_num,vector_size,columns);

                return std::move(scan);
            }
        },
        node.data);
}

// 将原来的计划，翻译为物理执行计划树，返回树的根节点
std::unique_ptr<ResultWriter> getPlan(const Plan& plan, SharedStateManager& shared_manager, size_t vector_size = 1024){
    // 从plan的根节点进入，递归创建Operator
    ProfileGuard profile_guard(global_profiler, "make plan");
    std::unique_ptr<Operator> op = getOperator(plan, plan.root, shared_manager, vector_size);
    // ResultWriter算子的共享状态id设为0，其余算子的共享状态id设为其在plan.nodes中的id+1
    auto& shared = shared_manager.get<ResultWriter::Shared>(0,plan.nodes[plan.root].output_attrs);
    std::unique_ptr<ResultWriter> result_writer = std::make_unique<ResultWriter>(shared,std::move(op));
    return std::move(result_writer);
}

ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {
    const int thread_num = 4;                           // 线程数（包括主线程）
    const int vector_size = 1024;                       // 向量化的批次大小
    std::vector<std::thread> threads;                   // 线程池
    std::vector<Barrier*> barriers = Barrier::create(thread_num);     // 屏障组
    SharedStateManager shared_manager;                  // 创建共享状态
    ColumnarTable result;                               // 执行结果
    global_profiler = new Profiler(thread_num);

    // 启动所有线程
    for (int i = 0; i < thread_num; ++i) {
        int barrier_group = i / Barrier::threads_per_barrier_;    // 每threads_per_barrier_个线程属于一个barrier_group

        if (i == thread_num - 1) {
            [&plan, &shared_manager, &result, &barriers, barrier_group, i]() {
                global_profiler->set_thread_id(i);
                ProfileGuard profile_guard(global_profiler, "execute");
                // 确定当前线程的Barrier
                current_barrier = barriers[barrier_group];
                // 每个线程生成各自的执行计划
                std::unique_ptr<ResultWriter> result_writer = getPlan(plan,shared_manager,vector_size);
                // 执行计划
                result_writer->next();
                // 等待所有线程完成
                bool is_last = current_barrier->wait();
                // 由最后一个线程转移结果
                if (is_last) result = std::move(result_writer->shared_.output_);
            }();
        } else {
            threads.emplace_back(        [&plan, &shared_manager, &result, &barriers, barrier_group, i]() {
                global_profiler->set_thread_id(i);
                ProfileGuard profile_guard(global_profiler, "execute");
                // 确定当前线程的Barrier
                current_barrier = barriers[barrier_group];
                // 每个线程生成各自的执行计划
                std::unique_ptr<ResultWriter> result_writer = getPlan(plan,shared_manager,vector_size);
                // 执行计划
                result_writer->next();
                // 等待所有线程完成
                bool is_last = current_barrier->wait();
                // 由最后一个线程转移结果
                if (is_last) result = std::move(result_writer->shared_.output_);
            });
        }   
    }

    // 等待所有线程结束
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // 销毁屏障
    Barrier::destroy(barriers);

    global_profiler->print_profiles();
    delete global_profiler;
    global_profiler = nullptr;

    return result;
}

void* build_context() {
    return nullptr;
}

void destroy_context([[maybe_unused]] void* context) {}

} // namespace Contest
