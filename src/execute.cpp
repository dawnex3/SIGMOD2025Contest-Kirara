#include <algorithm>
#include <chrono>
#include <hardware.h>
#include <plan.h>
#include <table.h>
#include <thread>
#include "MemoryPool.hpp"
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
    size_t                                           node_idx;

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
    const std::vector<std::tuple<size_t, DataType>>& output_attrs,
    size_t node_idx) {
    auto                           left_idx    = join.left;
    auto                           right_idx   = join.right;
    auto&                          left_node   = plan.nodes[left_idx];
    auto&                          right_node  = plan.nodes[right_idx];
    auto&                          left_types  = left_node.output_attrs;
    auto&                          right_types = right_node.output_attrs;
    auto                           right       = execute_impl(plan, right_idx);
    auto                           left        = execute_impl(plan, left_idx);
    std::vector<std::vector<Data>> results;

    JoinAlgorithm join_algorithm{.build_left = join.build_left,
        .left                                = left,
        .right                               = right,
        .results                             = results,
        .left_col                            = join.left_attr,
        .right_col                           = join.right_attr,
        .output_attrs                        = output_attrs,
        .node_idx                            = node_idx};
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

//    // 打印结果
//    std::ostringstream oss;
//    oss << "join "<< node_idx <<" output rows: " << results.size() << ", details:\n";
//    for (const auto &row : results) {
//        for (const auto &data : row) {
//            std::visit([&](auto&& d) {
//                using T = std::decay_t<decltype(d)>;
//                if constexpr (std::is_same_v<T, int32_t>) {
//                    oss << d << "\t\t"; // 输出 int32_t
//                } else if constexpr (std::is_same_v<T, std::basic_string<char>>) {
//                    oss << d << "\t\t"; // 输出字符串
//                } else if constexpr (std::is_same_v<T, std::monostate>) {
//                    oss << "NULL\t\t";
//                } else {
//                    throw std::runtime_error("Unsupported data type");
//                }
//            }, data);
//        }
//        oss << "\n";
//    }
//    oss << "\n";
//
//    std::ofstream log("log_true.txt", std::ios::app);
//    log << oss.str();
//    log.close();

    return results;
}

ExecuteResult execute_scan(const Plan&               plan,
    const ScanNode&                                  scan,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs,
    size_t node_idx) {
    auto                           table_id = scan.base_table_id;
    auto&                          input    = plan.inputs[table_id];
    auto                           table    = Table::from_columnar(input);
    std::vector<std::vector<Data>> results;
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
                return execute_hash_join(plan, value, node.output_attrs,node_idx);
            } else {
                return execute_scan(plan, value, node.output_attrs,node_idx);
            }
        },
        node.data);
}

Operator *getOperator(const Plan& plan, size_t node_idx, SharedStateManager& shared_manager, size_t vector_size = 1024){
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value)-> Operator *{
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                // 如果是join节点
                auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1); //共享状态id设为node_idx+1
                Operator *left_op = getOperator(plan, value.left, shared_manager, vector_size);
                Operator *right_op = getOperator(plan, value.right, shared_manager, vector_size);
                if(value.build_left){   // 左侧构建，正常情况
                    Operator *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                        shared, vector_size, left_op, value.left_attr,
                        right_op, value.right_attr, node.output_attrs, plan.nodes[value.left].output_attrs);
                    return hash_join;
                } else {    // 右侧构建，调换算子顺序，以及output_attrs的顺序
                    std::vector<std::tuple<size_t, DataType>> output_attrs;
                    size_t left_size = plan.nodes[value.left].output_attrs.size();
                    size_t right_size = plan.nodes[value.right].output_attrs.size();
                    for(auto [col_idx, col_type]: node.output_attrs){
                        output_attrs.emplace_back(col_idx>=left_size ? col_idx-left_size : col_idx+right_size, col_type);
                    }
                    Hashjoin *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                        shared, vector_size, right_op, value.right_attr,
                        left_op, value.left_attr, output_attrs, plan.nodes[value.right].output_attrs);
                    return hash_join;
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
                Scan *scan = new (local_allocator.allocate(sizeof(Scan))) Scan(shared,row_num,vector_size,columns);

                return scan;
            }
        },
        node.data);
}

// 将原来的计划，翻译为物理执行计划树，返回树的根节点
ResultWriter *getPlan(const Plan& plan, SharedStateManager& shared_manager, size_t vector_size = 1024){
    // 从plan的根节点进入，递归创建Operator
    ProfileGuard profile_guard(global_profiler, "make plan");
    Operator *op = getOperator(plan, plan.root, shared_manager, vector_size);
    // ResultWriter算子的共享状态id设为0，其余算子的共享状态id设为其在plan.nodes中的id+1
    auto& shared = shared_manager.get<ResultWriter::Shared>(0,plan.nodes[plan.root].output_attrs);
    ResultWriter *result_writer = new (local_allocator.allocate(sizeof(ResultWriter))) ResultWriter(shared, op);
    return result_writer;
}

void test_hash(){
    std::vector<uint32_t> hash_to_key((uint32_t)0xFFFFFFFF, 0);
    std::vector<bool> is_key_repeat((uint32_t)0xFFFFFFFF, false);

    uint32_t h0 = hash_32(0);

    uint32_t key = 0;
    do {
        uint32_t h = hash_32(key);
        if(h==h0){
            is_key_repeat[key]=true;
        } else if(hash_to_key[h]==0){
            hash_to_key[h]=key;
        } else {
            is_key_repeat[hash_to_key[h]]=true;
            is_key_repeat[key]=true;
        }
    } while (key++ != 0xFFFFFFFF);

    std::ofstream file("unrepeat_key.txt", std::ios::app);
    // 从int32的最小值开始，输出不碰撞的key
    int32_t test_num=10000;
    int32_t skey=NULL_INT32;
    for(int32_t i=0; i<test_num;skey++,i++){
        if(!is_key_repeat[(uint32_t)skey]){
            file<<skey<<std::endl;
        }
    }
    file.close();
    exit(1);
}

ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {
//    namespace views = ranges::views;
//    auto ret        = execute_impl(plan, plan.root);
//    auto ret_types  = plan.nodes[plan.root].output_attrs
//                   | views::transform([](const auto& v) { return std::get<1>(v); })
//                   | ranges::to<std::vector<DataType>>();
//    Table table{std::move(ret), std::move(ret_types)};
//    return table.to_columnar();

    size_t all_scan_size = 0;
    for (const auto& plan_node: plan.nodes){
        all_scan_size += std::visit([&plan](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, ScanNode>) {
                return plan.inputs[value.base_table_id].num_rows;
            } else {return static_cast<size_t>(0);}
        }, plan_node.data);
    }
//    printf("total = %ld \t", all_scan_size);
    int thread_num = all_scan_size >= 10000000 ? std::min(64, std::max((SPC__THREAD_COUNT / 4 - (SPC__THREAD_COUNT % 4 == 0)) * 4, 24))
                            : (all_scan_size >= 5000000 ? 24 : 16);
#ifdef SPC__PPC64LE
    if (thread_num == 16)
        thread_num = 12;
    else if (thread_num != 24)
        thread_num = 60;
#endif
//    const int thread_num = 1;
    const int vector_size = 1024;                       // 向量化的批次大小
    std::vector<std::thread> threads;                   // 线程池
    std::vector<Barrier*> barriers = Barrier::create(thread_num);     // 屏障组
    SharedStateManager shared_manager;                  // 创建共享状态
    ColumnarTable result;                               // 执行结果
    global_profiler = new Profiler(thread_num);
    global_mempool.reset();

    // 启动所有线程
    for (int i = 0; i < thread_num; ++i) {
        int barrier_group = i / Barrier::threads_per_barrier_;    // 每threads_per_barrier_个线程属于一个barrier_group

        if (i == thread_num - 1) {
            [&plan, &shared_manager, &result, &barriers, barrier_group, i]() {
                local_allocator.reuse();
                global_profiler->set_thread_id(i);
                ProfileGuard profile_guard(global_profiler, "execute");
                // 确定当前线程的Barrier
                current_barrier = barriers[barrier_group];
                // 每个线程生成各自的执行计划
                ResultWriter *result_writer = getPlan(plan,shared_manager,vector_size);
                // 执行计划
                result_writer->next();
                // 等待所有线程完成
                bool is_last = current_barrier->wait();
                // 由最后一个线程转移结果
                if (is_last) result = std::move(result_writer->shared_.output_);
            }();
        } else {
            threads.emplace_back(        [&plan, &shared_manager, &result, &barriers, barrier_group, i]() {
                local_allocator.reuse();
                global_profiler->set_thread_id(i);
                ProfileGuard profile_guard(global_profiler, "execute");
                // 确定当前线程的Barrier
                current_barrier = barriers[barrier_group];
                // 每个线程生成各自的执行计划
                ResultWriter *result_writer = getPlan(plan,shared_manager,vector_size);
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
    // delete global_mempool;
    // global_mempool = nullptr;

    std::this_thread::sleep_for(std::chrono::milliseconds (1200));   // 让cpu休息一下吧 :)
    // 2.27 1.71 xxx 3.58
    return result;
}


void* build_context() {
    return nullptr;
}

void destroy_context([[maybe_unused]] void* context) {}

} // namespace Contest