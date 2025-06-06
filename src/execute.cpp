#include <algorithm>
#include <chrono>
#include <condition_variable>
#include "hardware.h"
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

using namespace std::chrono;

struct Trunk {
    Trunk *prev;
    std::string str;
    Trunk(Trunk *prev, const std::string &str) : prev(prev), str(str) {}
};

// Recursively print the Trunk chain, i.e., print all prefixes from the root to the current node.
void showPlanTrunks(Trunk *p) {
    if (p == nullptr)
        return;
    showPlanTrunks(p->prev);
    std::cout << p->str;
}

// Helper function to convert the current PlanNode to a string description.
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

// ------------------ Function to print the Plan tree (top is right, bottom is left) ------------------
// 1. Recursively print the right subtree (if any), passing a new Trunk in the recursive call.
// 2. Set the current connector based on the passed prev pointer and the isLeft parameter.
// 3. Print the description of the current node (prefixed with connectors).
// 4. Recursively print the left subtree (if any).
void printPlanHelper(const Plan &plan, size_t nodeIndex, Trunk *prev, bool isLeft) {
// Return directly if out of bounds.
    if (nodeIndex >= plan.nodes.size()) {
        return;
    }

// Construct a Trunk node for the prefix display of this node.
    std::string prev_str = "       ";
    Trunk *trunk = new Trunk(prev, prev_str);

// Check if the current node is a JoinNode (which has left and right subtrees).
    bool isJoin = std::holds_alternative<JoinNode>(plan.nodes[nodeIndex].data);

// If it is a JoinNode, first recursively print the right subtree.
    if (isJoin) {
        const JoinNode &join = std::get<JoinNode>(plan.nodes[nodeIndex].data);
        printPlanHelper(plan, join.right, trunk, true);
    }

// Set the pattern for the current branch: if there's no predecessor, it's the root node; otherwise, it depends on whether it's the left side.
    if (!prev) {
        trunk->str = "———";
    }
    else if (isLeft) {
        trunk->str = ".———";
        prev_str = "      |";
    }
    else { // Right side.
        trunk->str = "`———";
        if (prev)
            prev->str = prev_str;
    }

// Print the prefix information and the current node description.
    showPlanTrunks(trunk);
    std::cout << " " << planNodeToString(plan, nodeIndex) << std::endl;

    if (prev) {
        prev->str = prev_str;
    }
    trunk->str = "      |";

// If it is a JoinNode, then recursively print the left subtree.
    if (isJoin) {
        const JoinNode &join = std::get<JoinNode>(plan.nodes[nodeIndex].data);
        printPlanHelper(plan, join.left, trunk, false);
    }

// Note: Free the memory of the current trunk (this example uses new/delete without complex memory management).
    delete trunk;
}

// Public interface: Print the entire Plan tree starting from plan.root.
void printPlanTree(const Plan &plan) {
    printPlanHelper(plan, plan.root, nullptr, false);
}


Operator *getOperator(const Plan& plan, size_t node_idx, SharedStateManager& shared_manager, bool is_build_side, QueryCache* query_cache, size_t vector_size = 1024, const std::vector<ColumnarTable>* input=nullptr){
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value)-> Operator *{
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, JoinNode>) {        // If it is a join node.
// Get information about the cached hash table.
              Hashmap* hashmap=query_cache->getHashmap(node_idx);
              bool is_build=false;
              bool store_hash=false;
              QueryCache::CacheType type = query_cache->getCacheType(node_idx);
              if(type==QueryCache::USE_CACHE){
                  is_build=true;
              } else if(type==QueryCache::OWN_CACHE){
                  store_hash=true;
              }

// Get build side and probe side information based on which side is the build side.
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

// Recursively generate the plan for the build and probe sides.
              Operator *build_op, *probe_op;
              probe_op = getOperator(plan, probe_node, shared_manager, false, query_cache, vector_size);
              if(is_build){   // Use the cached hash table.
                  build_op = nullptr;
                  auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1, hashmap); // Set the shared state ID to node_idx+1.
                  Operator *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                      shared, vector_size, build_op, build_attr,
                      probe_op, probe_attr, output_attrs, plan.nodes[build_node].output_attrs,is_build,store_hash);
                  return hash_join;
              } else {
                  build_op = getOperator(plan, build_node, shared_manager, true, query_cache, vector_size);
                  if(build_op!=nullptr){     // Hash join without using a cached hash table.
                      auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1, hashmap); // Set the shared state ID to node_idx+1.
                      Operator *hash_join = new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                          shared, vector_size, build_op, build_attr,
                          probe_op, probe_attr, output_attrs, plan.nodes[build_node].output_attrs,is_build,store_hash);
                      return hash_join;
                  } else {                   // Use naive join.
                      auto& shared = shared_manager.get<Naivejoin::Shared>(node_idx + 1); // Set the shared state ID to node_idx+1.
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
// If it is a scan node.
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
              auto& shared = shared_manager.get<Scan::Shared>(node_idx + 1); // Set the shared state ID to node_idx+1.
// Populate the data source.
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

// Translate the original plan into a physical execution plan tree and return the root of the tree.
ResultWriter *getPlan(const Plan& plan, SharedStateManager& shared_manager, size_t vector_size = 1024, QueryCache* query_cache=nullptr, const std::vector<ColumnarTable>* input=nullptr){
// Start from the root of the plan and recursively create Operators.
    ProfileGuard profile_guard(global_profiler, "make plan");
    Operator *op = getOperator(plan, plan.root, shared_manager, false, query_cache, vector_size, input);
// The shared state ID for the ResultWriter operator is set to 0, while other operators' shared state IDs are set to their ID in plan.nodes + 1.
    auto& shared = shared_manager.get<ResultWriter::Shared>(0,plan.nodes[plan.root].output_attrs);
    ResultWriter *result_writer = new (local_allocator.allocate(sizeof(ResultWriter))) ResultWriter(shared, op);
    return result_writer;
}

// simple function to determine `threadNum`.
// we sum up the number of all rows in all ScanNodes to estimate the size of query
// for small query, we use small `threadNum`, and vice versa.
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
    int thread_num = all_scan_size >= 10000000 ? std::min(64, std::max((SPC__THREAD_COUNT / 4 - (SPC__THREAD_COUNT % 4 == 0)) * 4, 24))
                                               : (all_scan_size >= 5000000 ? 24 : std::min(SPC__CORE_COUNT, 16));
    if (SPC__THREAD_COUNT / SPC__CORE_COUNT >= 4 && thread_num >= 64){
        thread_num = SPC__CORE_COUNT * 2;
    }
    return std::min(thread_num, SPC__THREAD_COUNT);
}

CacheManager cache_manager;     // Hash table cache manager.

ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {

#ifdef DEBUG_LOG
    printPlanTree(plan);    // Print the plan tree in a human-readable format.
#endif

    size_t thread_num = threadNum(plan);                // Number of threads.
    const int vector_size = 1024;                       // Vectorized batch size.
    std::vector<Barrier*> barriers = Barrier::create(thread_num);     // Barrier group.
    SharedStateManager shared_manager;                  // Create shared states.
    ColumnarTable result;                               // Execution result.
    QueryCache* query_cache = cache_manager.getQuery(plan);           // Hash table cache.
    global_profiler = new Profiler(thread_num);
    global_mempool.reset();
    std::condition_variable finish_cv;
    std::mutex finish_mtx;
    bool finished = false;

// Start all threads.
    for (size_t i = 0; i < thread_num; ++i) {
        size_t barrier_group = i / Barrier::threads_per_barrier_;    // Each group of threads_per_barrier_ threads belongs to one barrier_group.

        g_thread_pool->assign_task(i, [&plan, &shared_manager, &result,
            &barriers, barrier_group, i,
            &query_cache, &finish_cv, &finish_mtx, &finished]() {
          local_allocator.reuse();
          global_profiler->set_thread_id(i);
          ProfileGuard profile_guard(global_profiler, "execute");
// Determine the Barrier for the current thread.
          current_barrier = barriers[barrier_group];
// Each thread generates its own execution plan.
          ResultWriter *result_writer =
              getPlan(plan, shared_manager, vector_size, query_cache);
// Execute the plan.
          result_writer->next();
// Wait for all threads to finish.
          bool is_last = current_barrier->wait();
// The last thread transfers the result.
          if (is_last) {
              result = std::move(result_writer->shared_.output_);
              {
                  std::unique_lock lock(finish_mtx);
                  finished = true;
                  finish_cv.notify_all();
              }
          }
        });
    }

// Wait for all threads to terminate.
    {
        std::unique_lock lock(finish_mtx);
        finish_cv.wait(lock, [&]() { return finished; });
    }

// Destroy barriers.
    Barrier::destroy(barriers);

// Add the query to the cache.
    cache_manager.cacheQuery(query_cache);

    global_profiler->print_profiles();
    delete global_profiler;
    global_profiler = nullptr;

    return result;
}

void *build_context() {
    global_profiler = new Profiler(1);
    global_profiler->set_thread_id(0);

    global_mempool.init();
    if (g_thread_pool == nullptr) {
        g_thread_pool = new ThreadPool();
    }

    global_profiler->print_profiles();
    delete global_profiler;
    global_profiler = nullptr;
    return nullptr;
}

void destroy_context([[maybe_unused]] void *context) {

    global_mempool.destroy();
}

} // namespace Contest