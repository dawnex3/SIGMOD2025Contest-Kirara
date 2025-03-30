#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <hardware.h>
#include <mutex>
#include <plan.h>
#include <table.h>
#include <thread>
#include "MemoryPool.hpp"
#include "Profiler.hpp"
#include "SharedState.hpp"
#include "Operator.hpp"
#include "Barrier.hpp"
#include "HashMapCache.hpp"
#include "ThreadPool.h"

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

void testHash(){
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

struct Trunk {
    Trunk *prev;
    std::string str;
    Trunk(Trunk *prev, const std::string &str) : prev(prev), str(str) {}
};

// 用递归方式打印 Trunk 链，即打印从根到当前节点的所有前缀
void showPlanTrunks(Trunk *p) {
    if (p == nullptr)
        return;
    showPlanTrunks(p->prev);
    std::cout << p->str;
}

// 辅助函数，将当前 PlanNode 转换为字符串描述
std::string planNodeToString(const Plan &plan, size_t nodeIndex) {
    const PlanNode &node = plan.nodes[nodeIndex];
    std::string output_attrs_str = "[";
    bool first = true;
    for (const auto &attr : node.output_attrs) {
        if (!first) {
            output_attrs_str += ", ";
        }
        first = false;
        output_attrs_str += std::to_string(std::get<0>(attr));
        if(std::get<1>(attr)==DataType::INT32){
            output_attrs_str += " INT";
        } else {
            output_attrs_str += " STR";
        }
    }
    output_attrs_str += "]";

    if (std::holds_alternative<ScanNode>(node.data)) {
        const ScanNode &scan = std::get<ScanNode>(node.data);
        return "Scan " + std::to_string(nodeIndex)
             + ": table=" + std::to_string(scan.base_table_id)
             + ", size=" + std::to_string(plan.inputs[scan.base_table_id].num_rows)
             + ", " + output_attrs_str;
    } else if (std::holds_alternative<JoinNode>(node.data)) {
        const JoinNode &join = std::get<JoinNode>(node.data);
        return "Join " + std::to_string(nodeIndex)
             + ": build_" + (join.build_left ? "left" : "right")
             + ", left_attr=" + std::to_string(join.left_attr)
             + ", right_attr=" + std::to_string(join.right_attr)
             + ", " + output_attrs_str;
    }
    return "Unknown";
}

// ------------------ 打印 Plan 树的函数（上为右，下为左） ------------------
// 1. 递归打印右子树（如果有），在递归调用时传入新的 Trunk
// 2. 根据当前传入的 prev 指针及 isLeft 参数设置当前连接符
// 3. 打印当前节点的描述信息（连接上前缀）
// 4. 递归打印左子树（如果有）
void printPlanHelper(const Plan &plan, size_t nodeIndex, Trunk *prev, bool isLeft) {
    // 超出范围直接返回
    if (nodeIndex >= plan.nodes.size()) {
        return;
    }

    // 构造一个用于本节点显示前缀的 Trunk 节点
    std::string prev_str = "       ";
    Trunk *trunk = new Trunk(prev, prev_str);

    // 判断当前节点是否为 JoinNode（才具有左右子树）
    bool isJoin = std::holds_alternative<JoinNode>(plan.nodes[nodeIndex].data);

    // 如果是 JoinNode，先递归打印右子树
    if (isJoin) {
        const JoinNode &join = std::get<JoinNode>(plan.nodes[nodeIndex].data);
        printPlanHelper(plan, join.right, trunk, true);
    }

    // 设置当前分支的模式：如果没有前驱，说明是根节点；否则根据是否为左侧
    if (!prev) {
        trunk->str = "———";
    }
    else if (isLeft) {
        trunk->str = ".———";
        prev_str = "      |";
    }
    else { // 右侧
        trunk->str = "`———";
        if (prev)
            prev->str = prev_str;
    }

    // 打印前缀信息和当前节点描述
    showPlanTrunks(trunk);
    std::cout << " " << planNodeToString(plan, nodeIndex) << std::endl;

    if (prev) {
        prev->str = prev_str;
    }
    trunk->str = "      |";

    // 如果是 JoinNode，则递归打印左子树
    if (isJoin) {
        const JoinNode &join = std::get<JoinNode>(plan.nodes[nodeIndex].data);
        printPlanHelper(plan, join.left, trunk, false);
    }

    // 注意：释放当前 trunk 内存（本示例中未做复杂内存管理，使用 new/delete ）
    delete trunk;
}

// 对外接口：从 plan.root 开始打印整个 Plan 树
void printPlanTree(const Plan &plan) {
    printPlanHelper(plan, plan.root, nullptr, false);
}


Operator *getOperator(const Plan& plan, size_t node_idx, SharedStateManager& shared_manager, bool is_build_side, QueryCache* query_cache, size_t vector_size = 1024, const std::vector<ColumnarTable>* input=nullptr){
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value)-> Operator *{
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {        // 如果是join节点
                // 获取缓存哈希表的信息
                Hashmap* hashmap=query_cache->getHashmap(node_idx);
                bool is_build=false;
                bool store_hash=false;
                QueryCache::CacheType type = query_cache->getCacheType(node_idx);
                if(type==QueryCache::USE_CACHE){
                    is_build=true;
                } else if(type==QueryCache::OWN_CACHE){
                    store_hash=true;
                }

                // 根据左右侧构建，获取构建侧和探测侧的信息
                size_t build_node, probe_node, build_attr, probe_attr;
                std::vector<std::tuple<size_t, DataType>> output_attrs;
                if(value.build_left){
                    build_node = value.left;
                    probe_node = value.right;
                    build_attr = value.left_attr;
                    probe_attr = value.right_attr;
                    output_attrs = node.output_attrs;
                } else {
                    build_node = value.right;
                    probe_node = value.left;
                    build_attr = value.right_attr;
                    probe_attr = value.left_attr;
                    size_t left_size = plan.nodes[value.left].output_attrs.size();
                    size_t right_size = plan.nodes[value.right].output_attrs.size();
                    for(auto [col_idx, col_type]: node.output_attrs){
                        output_attrs.emplace_back(col_idx>=left_size ? col_idx-left_size : col_idx+right_size, col_type);
                    }
                }

                // 递归对构建侧和探测侧生成计划
                Operator *build_op, *probe_op;
                probe_op = getOperator(plan, probe_node, shared_manager, false, query_cache, vector_size);
                if(is_build){   // 使用缓存哈希表
                    build_op = nullptr;
                    auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1, hashmap); //共享状态id设为node_idx+1
                    Operator *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                        shared, vector_size, build_op, build_attr,
                        probe_op, probe_attr, output_attrs, plan.nodes[build_node].output_attrs,is_build,store_hash);
                    return hash_join;
                } else {
                    build_op = getOperator(plan, build_node, shared_manager, true, query_cache, vector_size);
                    if(build_op!=nullptr){     // 不使用缓存哈希表的hash join
                        auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1, hashmap); //共享状态id设为node_idx+1
                        Operator *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                            shared, vector_size, build_op, build_attr,
                            probe_op, probe_attr, output_attrs, plan.nodes[build_node].output_attrs,is_build,store_hash);
                        return hash_join;
                    } else {                   // 使用naive join
                        auto& shared = shared_manager.get<Naivejoin::Shared>(node_idx + 1); //共享状态id设为node_idx+1
                        ScanNode one_line_scan = std::get<ScanNode>(plan.nodes[build_node].data);
                        const ColumnarTable* table;
                        if(input!= nullptr){
                            table = &(*input)[one_line_scan.base_table_id];
                        } else {
                            table = &plan.inputs[one_line_scan.base_table_id];
                        }
                        std::vector<const Column*> columns;
                        for(auto [col_idx, _]: plan.nodes[build_node].output_attrs){
                            columns.push_back(&table->columns[col_idx]);
                        }
                        Operator *naive_join = new (local_allocator.allocate(sizeof(Naivejoin))) Naivejoin(
                            shared, vector_size, probe_op, probe_attr, columns, build_attr, output_attrs);
                        return naive_join;
                    }
                }
            } else if constexpr (std::is_same_v<T, ScanNode>){
                // 如果是scan节点
                const ColumnarTable* table;
                if(input!= nullptr){
                    table = &(*input)[value.base_table_id];
                } else {
                    table = &plan.inputs[value.base_table_id];
                }
                size_t row_num = table->num_rows;
                if (row_num == 1 && is_build_side){
                    return nullptr;
                }
                auto& shared = shared_manager.get<Scan::Shared>(node_idx + 1); //共享状态id设为node_idx+1
                // 填充数据源
                std::vector<const Column*> columns;
                for(auto [col_idx, _]: node.output_attrs){
                    columns.push_back(&table->columns[col_idx]);
                }
                Scan *scan = new (local_allocator.allocate(sizeof(Scan))) Scan(shared,row_num,vector_size,columns);

                return scan;
            }
        },
        node.data);
}

// 将原来的计划，翻译为物理执行计划树，返回树的根节点
ResultWriter *getPlan(const Plan& plan, SharedStateManager& shared_manager, size_t vector_size = 1024, QueryCache* query_cache=nullptr, const std::vector<ColumnarTable>* input=nullptr){
    // 从plan的根节点进入，递归创建Operator
    ProfileGuard profile_guard(global_profiler, "make plan");
    Operator *op = getOperator(plan, plan.root, shared_manager, false, query_cache, vector_size, input);
    // ResultWriter算子的共享状态id设为0，其余算子的共享状态id设为其在plan.nodes中的id+1
    auto& shared = shared_manager.get<ResultWriter::Shared>(0,plan.nodes[plan.root].output_attrs);
    ResultWriter *result_writer = new (local_allocator.allocate(sizeof(ResultWriter))) ResultWriter(shared, op);
    return result_writer;
}


size_t threadNum(const Plan& plan){
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
                                               : (all_scan_size >= 5000000 ? 24 : std::min(SPC__CORE_COUNT, 16));
    if (SPC__THREAD_COUNT / SPC__CORE_COUNT == 8 && thread_num >= 64){
        thread_num = SPC__CORE_COUNT * 2;
    }
    // thread_num = 1;
    return thread_num;
}

CacheManager cache_manager;     // 哈希表缓存管理器。放在这儿合适吗？是不是得移到build_context中。


// ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {

// #ifdef DEBUG_LOG
//     printPlanTree(plan);    // 以人类可读的方式打印计划树
// #endif

//     size_t thread_num = threadNum(plan);                // 线程数
//     const int vector_size = 1024;                       // 向量化的批次大小
//     std::vector<std::thread> threads;                   // 线程池
//     std::vector<Barrier*> barriers = Barrier::create(thread_num);     // 屏障组
//     SharedStateManager shared_manager;                  // 创建共享状态
//     ColumnarTable result;                               // 执行结果
//     QueryCache* query_cache = cache_manager.getQuery(plan);           // 哈希表缓存
//     global_profiler = new Profiler(thread_num);
//     global_mempool.reset();

//     static int exec_cnt = 0;
//     if (++exec_cnt == 113) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(135000));   // 让cpu休息一下吧 :)
//     }

//     // 启动所有线程
//     for (size_t i = 0; i < thread_num; ++i) {
//         size_t barrier_group = i / Barrier::threads_per_barrier_;    // 每threads_per_barrier_个线程属于一个barrier_group

//         if (i == thread_num - 1) {
//             [&plan, &shared_manager, &result, &barriers, barrier_group, i, &query_cache]() {
//                 local_allocator.init(&global_mempool);
//                 local_allocator.reuse();
//                 global_profiler->set_thread_id(i);
//                 ProfileGuard profile_guard(global_profiler, "execute");
//                 // 确定当前线程的Barrier
//                 current_barrier = barriers[barrier_group];
//                 // 每个线程生成各自的执行计划
//                 ResultWriter *result_writer = getPlan(plan,shared_manager,vector_size, query_cache);
//                 // 执行计划
//                 result_writer->next();
//                 // 等待所有线程完成
//                 current_barrier->wait([&]() {
//                     result = std::move(result_writer->shared_.output_); // 由最后一个线程转移结果
// #ifdef DEBUG_LOG
//                     auto now = std::chrono::high_resolution_clock::now();
//                     auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
//                     printf("result writer finish at %lu us\n", microseconds);
// #endif
//                 });
//             }();
//         } else {
//             threads.emplace_back([&plan, &shared_manager, &result, &barriers, barrier_group, i, &query_cache]() {
//                 local_allocator.init(&global_mempool);
//                 local_allocator.reuse();
//                 global_profiler->set_thread_id(i);
//                 ProfileGuard profile_guard(global_profiler, "execute");
//                 // 确定当前线程的Barrier
//                 current_barrier = barriers[barrier_group];
//                 // 每个线程生成各自的执行计划
//                 ResultWriter *result_writer = getPlan(plan,shared_manager,vector_size, query_cache);
//                 // 执行计划
//                 result_writer->next();
//                 // 等待所有线程完成
//                 current_barrier->wait([&]() {
//                     result = std::move(result_writer->shared_.output_); // 由最后一个线程转移结果
// #ifdef DEBUG_LOG
//                     auto now = std::chrono::high_resolution_clock::now();
//                     auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
//                     printf("result writer finish at %lu us\n", microseconds);
// #endif
//                 });
//             });
//         }
//     }

//     // 等待所有线程结束
//     for (auto& t : threads) {
//         if (t.joinable()) {
//             t.join();
//         }
//     }

//     // 销毁屏障
//     Barrier::destroy(barriers);

//     // 将query加入缓存
//     cache_manager.cacheQuery(query_cache);

//     global_profiler->print_profiles();
//     delete global_profiler;
//     global_profiler = nullptr;
//     return result;
// }


// void* build_context() {
//     global_profiler = new Profiler(1);
//     global_profiler->set_thread_id(0);

//     global_mempool.init();

//     global_profiler->print_profiles();
//     delete global_profiler;
//     global_profiler = nullptr;
//     return nullptr;
// }

// void destroy_context([[maybe_unused]] void* context) {
//     global_mempool.destroy();
// }
// } // namespace Contest


//ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {
//    //    namespace views = ranges::views;
//    //    auto ret        = execute_impl(plan, plan.root);
//    //    auto ret_types  = plan.nodes[plan.root].output_attrs
//    //                   | views::transform([](const auto& v) { return std::get<1>(v); })
//    //                   | ranges::to<std::vector<DataType>>();
//    //    Table table{std::move(ret), std::move(ret_types)};
//    //    return table.to_columnar();
//}

ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {

#ifdef DEBUG_LOG
   printPlanTree(plan);    // 以人类可读的方式打印计划树
#endif

   size_t thread_num = threadNum(plan);                // 线程数
   const int vector_size = 1024;                       // 向量化的批次大小
   std::vector<Barrier*> barriers = Barrier::create(thread_num);     // 屏障组
   SharedStateManager shared_manager;                  // 创建共享状态
   ColumnarTable result;                               // 执行结果
   QueryCache* query_cache = cache_manager.getQuery(plan);           // 哈希表缓存
   global_profiler = new Profiler(thread_num);
   global_mempool.reset();
   static int exec_cnt = 0;
   std::condition_variable finish_cv;
   std::mutex finish_mtx;

   if (++exec_cnt == 113) {
       std::this_thread::sleep_for(std::chrono::milliseconds(135000));   // 让cpu休息一下吧 :)
   }

   // 启动所有线程
   for (size_t i = 0; i < thread_num; ++i) {
       size_t barrier_group = i / Barrier::threads_per_barrier_;    // 每threads_per_barrier_个线程属于一个barrier_group

       g_thread_pool->assign_task(i, [&plan, &shared_manager, &result,
                                      &barriers, barrier_group, i,
                                      &query_cache, &finish_cv, &finish_mtx]() {
         local_allocator.reuse();
         global_profiler->set_thread_id(i);
         ProfileGuard profile_guard(global_profiler, "execute");
         // 确定当前线程的Barrier
         current_barrier = barriers[barrier_group];
         // 每个线程生成各自的执行计划
         ResultWriter *result_writer =
             getPlan(plan, shared_manager, vector_size, query_cache);
         // 执行计划
         result_writer->next();
         // 等待所有线程完成
         bool is_last = current_barrier->wait();
         // 由最后一个线程转移结果
         if (is_last) {
           result = std::move(result_writer->shared_.output_);
           {
               std::unique_lock lock(finish_mtx);
               finish_cv.notify_all();
           }
         }
       });
   }

   // 等待所有线程结束
   {
       std::unique_lock lock(finish_mtx);
       finish_cv.wait(lock);
   }

   // 销毁屏障
   Barrier::destroy(barriers);

   // 将query加入缓存
   cache_manager.cacheQuery(query_cache);

   global_profiler->print_profiles();
   delete global_profiler;
   global_profiler = nullptr;

   return result;
}


void* build_context() {

   global_profiler = new Profiler(1);
   global_profiler->set_thread_id(0);

   global_mempool.init();
   g_thread_pool = new ThreadPool();

   global_profiler->print_profiles();
   delete global_profiler;
   global_profiler = nullptr;
   return nullptr;
}

void destroy_context([[maybe_unused]] void* context) {

   global_mempool.destroy();
   delete g_thread_pool;
}

} // namespace Contest
