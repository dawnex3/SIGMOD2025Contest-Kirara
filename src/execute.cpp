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

Operator *getOperator(const Plan& plan, size_t node_idx, SharedStateManager& shared_manager, size_t vector_size = 1024, bool accept_null=true){
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value)-> Operator *{
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                // 如果是join节点
                auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1); //共享状态id设为node_idx+1
                Operator *right_op = getOperator(plan, value.right, shared_manager, vector_size);
                Operator *left_op = getOperator(plan, value.left, shared_manager, vector_size, right_op != nullptr);
                if (__glibc_unlikely(left_op == nullptr)){
//                    printf("BUILD LEFT\n");
                    auto one_line = std::get<ScanNode>(plan.nodes[value.left].data);
                    const ColumnarTable& table = plan.inputs[one_line.base_table_id];
                    std::vector<const Column*> columns;
                    for(auto [col_idx, _]: plan.nodes[value.left].output_attrs){
                        columns.push_back(&table.columns[col_idx]);
                    }
                    Operator *naive_join = new (local_allocator.allocate(sizeof(Naivejoin))) Naivejoin(
                        vector_size, right_op, value.right_attr, columns, value.left_attr, node.output_attrs);
                    return naive_join;
                } else if (right_op == nullptr){
//                    printf("BUILD RIGHT\n");
                    std::vector<std::tuple<size_t, DataType>> output_attrs;
                    size_t left_size = plan.nodes[value.left].output_attrs.size();
                    size_t right_size = plan.nodes[value.right].output_attrs.size();
                    for(auto [col_idx, col_type]: node.output_attrs){
                        output_attrs.emplace_back(col_idx>=left_size ? col_idx-left_size : col_idx+right_size, col_type);
                    }

                    auto one_line = std::get<ScanNode>(plan.nodes[value.right].data);
                    const ColumnarTable& table = plan.inputs[one_line.base_table_id];
                    std::vector<const Column*> columns;
                    for(auto [col_idx, _]: plan.nodes[value.right].output_attrs){
                        columns.push_back(&table.columns[col_idx]);
                    }

                    Operator *naive_join = new (local_allocator.allocate(sizeof(Naivejoin))) Naivejoin(
                        vector_size, left_op, value.left_attr, columns, value.right_attr, output_attrs);
                    return naive_join;
                } else if(__glibc_unlikely(value.build_left)){   // 左侧构建，不正常情况
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
                    auto *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                        shared, vector_size, right_op, value.right_attr,
                        left_op, value.left_attr, output_attrs, plan.nodes[value.right].output_attrs);
                    return hash_join;
                }
            } else if constexpr (std::is_same_v<T, ScanNode>){
                // 如果是scan节点
                const ColumnarTable& table = plan.inputs[value.base_table_id];
                if (table.num_rows == 1 && accept_null){
                    return nullptr;
                }
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
    }
    output_attrs_str += "]";

    if (std::holds_alternative<ScanNode>(node.data)) {
        const ScanNode &scan = std::get<ScanNode>(node.data);
        return "Scan " + std::to_string(nodeIndex)
             + ": table=" + std::to_string(scan.base_table_id)
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


ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {
//    namespace views = ranges::views;
//    auto ret        = execute_impl(plan, plan.root);
//    auto ret_types  = plan.nodes[plan.root].output_attrs
//                   | views::transform([](const auto& v) { return std::get<1>(v); })
//                   | ranges::to<std::vector<DataType>>();
//    Table table{std::move(ret), std::move(ret_types)};
//    return table.to_columnar();

#ifdef DEBUG_LOG
    printPlanTree(plan);    // 以人类可读的方式打印计划树
#endif

    size_t thread_num = threadNum(plan);                // 线程数
    const int vector_size = 1024;                       // 向量化的批次大小
    std::vector<std::thread> threads;                   // 线程池
    std::vector<Barrier*> barriers = Barrier::create(thread_num);     // 屏障组
    SharedStateManager shared_manager;                  // 创建共享状态
    ColumnarTable result;                               // 执行结果
    global_profiler = new Profiler(thread_num);
    global_mempool.reset();

    // 启动所有线程
    for (size_t i = 0; i < thread_num; ++i) {
        size_t barrier_group = i / Barrier::threads_per_barrier_;    // 每threads_per_barrier_个线程属于一个barrier_group

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
    // 1.85 1.48 ??? 2.71
    return result;
}


void* build_context() {
    return nullptr;
}

void destroy_context([[maybe_unused]] void* context) {}

} // namespace Contest