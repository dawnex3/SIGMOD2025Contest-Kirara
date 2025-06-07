/**
* @file execute.cpp
* @brief Implements the main execution logic for the parallel query engine.
*
* This file contains the core components that translate a logical query plan into
* a physical, executable operator tree and run it across multiple threads. It
* orchestrates the entire query lifecycle from plan instantiation to result collection.
*
* The key functionalities provided by this file include:
*
* - **Plan-to-Operator Translation (`getOperator`, `getPlan`):**
*   These functions recursively traverse the logical `Plan` and construct a
*   corresponding physical `Operator` tree. They incorporate optimizations such as
*   hash table caching (via `QueryCache`) and adaptive join strategy selection
*   (e.g., choosing `NaiveJoin` for single-row build sides).
*
* - **Parallel Execution (`execute`):**
*   This is the main entry point for query execution. It orchestrates the entire
*   process: determining the optimal number of threads (`threadNum`), setting up
*   synchronization primitives (`Barrier`), managing shared state (`SharedStateManager`),
*   and dispatching tasks to a global thread pool. Each worker thread generates its
*   own instance of the physical plan and executes it independently.
*
* - **Plan Visualization (`PlanPrinter`):**
*   A utility class is provided to visualize the logical query plan as a
*   formatted tree, which is essential for debugging.
*
* - **Global Context Management (`build_context`, `destroy_context`):**
*   Lifecycle functions for initializing and tearing down global resources like
*   the thread pool and memory manager.
*/

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include "hardware.h"
#include "plan.h"
#include "MemoryPool.hpp"
#include "Profiler.hpp"
#include "SharedState.hpp"
#include "Operator.hpp"
#include "Barrier.hpp"
#include "HashMapCache.hpp"
#include "ThreadPool.h"

namespace Contest {

using namespace std::chrono;

/**
 * @class PlanPrinter
 * @brief A utility class for visualizing a query `Plan` as a formatted tree.
 *
 * This class encapsulates all logic required to print a query plan in a human-readable,
 * hierarchical format. It is designed to be a stateless utility.
 */
class PlanPrinter {
private:
    /**
     * @struct Trunk
     * @brief A helper struct to build the visual connectors (e.g., "|  ", "`--") for the tree.
     *
     * Each Trunk node represents one level of the tree and stores the string prefix
     * needed to connect it visually to its parent.
     */
    struct Trunk {
        Trunk* prev;
        std::string str;

        Trunk(Trunk* prev, const std::string& str) : prev(prev), str(str) {}
    };

    /**
     * @brief Recursively prints the visual connectors from the root to the current node.
     * @param p The current Trunk node in the chain.
     */
    static void showTrunks(Trunk* p) {
        if (p == nullptr) {
            return;
        }
        showTrunks(p->prev);
        std::cout << p->str;
    }

    /**
     * @brief Converts a PlanNode into a descriptive string.
     * @param plan The query plan containing the node.
     * @param nodeIndex The index of the node to describe.
     * @return A string summarizing the operator (e.g., "Scan", "Join") and its key properties.
     */
    static std::string planNodeToString(const Plan& plan, size_t nodeIndex) {
        const PlanNode& node = plan.nodes[nodeIndex];
        std::string output_attrs_str = "[";
        bool first = true;
        for (const auto& attr : node.output_attrs) {
            if (!first) {
                output_attrs_str += ", ";
            }
            first = false;
            output_attrs_str += std::to_string(std::get<0>(attr));
            output_attrs_str += (std::get<1>(attr) == DataType::INT32) ? " INT" : " STR";
        }
        output_attrs_str += "]";

        if (std::holds_alternative<ScanNode>(node.data)) {
            const ScanNode& scan = std::get<ScanNode>(node.data);
            return "Scan " + std::to_string(nodeIndex)
                 + ": table=" + std::to_string(scan.base_table_id)
                 + ", size=" + std::to_string(plan.inputs[scan.base_table_id].num_rows)
                 + ", " + output_attrs_str;
        } else if (std::holds_alternative<JoinNode>(node.data)) {
            const JoinNode& join = std::get<JoinNode>(node.data);
            return "Join " + std::to_string(nodeIndex)
                 + ": build_" + (join.build_left ? "left" : "right")
                 + ", left_attr=" + std::to_string(join.left_attr)
                 + ", right_attr=" + std::to_string(join.right_attr)
                 + ", " + output_attrs_str;
        }
        return "Unknown";
    }

    /**
     * @brief The recursive core of the printing logic.
     *
     * It performs a reverse in-order traversal (right, root, left) to print the tree
     * in a standard top-to-bottom layout where the right child is on top.
     * @param plan The query plan.
     * @param nodeIndex The index of the current node to print.
     * @param prev The Trunk of the parent node.
     * @param isLeft True if the current node is a left child, false otherwise.
     */
    static void printHelper(const Plan& plan, size_t nodeIndex, Trunk* prev, bool isLeft) {
        if (nodeIndex >= plan.nodes.size()) {
            return;
        }

        std::string prev_str = "    ";
        Trunk* trunk = new Trunk(prev, prev_str);

        // Recursively print the right subtree first (which will appear on top).
        if (std::holds_alternative<JoinNode>(plan.nodes[nodeIndex].data)) {
            const JoinNode& join = std::get<JoinNode>(plan.nodes[nodeIndex].data);
            printHelper(plan, join.right, trunk, true);
        }

        if (!prev) {
            trunk->str = "---";
        } else if (isLeft) {
            trunk->str = ".--";
            prev_str = "   |";
        } else {
            trunk->str = "`--";
            prev->str = prev_str;
        }

        showTrunks(trunk);
        std::cout << " " << planNodeToString(plan, nodeIndex) << std::endl;

        if (prev) {
            prev->str = prev_str;
        }
        trunk->str = "   |";

        // Recursively print the left subtree (which will appear on the bottom).
        if (std::holds_alternative<JoinNode>(plan.nodes[nodeIndex].data)) {
            const JoinNode& join = std::get<JoinNode>(plan.nodes[nodeIndex].data);
            printHelper(plan, join.left, trunk, false);
        }

        delete trunk;
    }

public:
    /**
     * @brief Public entry point to print an entire query plan tree.
     * @param plan The query plan to print.
     */
    static void printTree(const Plan& plan) {
        printHelper(plan, plan.root, nullptr, false);
    }
};


/**
 * @brief Recursively builds a physical operator tree from a logical query plan.
 *
 * This function is the factory for the physical query plan. It traverses the logical
 * `Plan` tree, and for each `PlanNode`, it constructs the corresponding physical
 * `Operator` (e.g., `Scan`, `Hashjoin`). It handles optimizations such as hash table
 * caching and switching to a `NaiveJoin` for single-row build sides.
 *
 * @param plan The logical query plan.
 * @param node_idx The index of the current `PlanNode` to process.
 * @param shared_manager Manages shared state between parallel operator instances, crucial for coordination.
 * @param is_build_side A flag indicating if the current recursive descent is for the build side of a hash join.
 * @param query_cache Manages caching and reuse of hash tables across different queries.
 * @param vector_size The number of rows to process in each batch (vector).
 * @param input An optional pointer to an alternative set of input tables, typically used for intermediate results.
 * @return A pointer to the root operator of the generated physical plan subtree. The memory is allocated
 *         from the thread-local `local_allocator`.
 */
Operator* getOperator(const Plan& plan, size_t node_idx, SharedStateManager& shared_manager, bool is_build_side, QueryCache* query_cache, size_t vector_size = 1024, const std::vector<ColumnarTable>* input = nullptr) {
    auto& node = plan.nodes[node_idx];
    // Use std::visit to handle the different operator types stored in the PlanNode's variant.
    return std::visit(
        [&](const auto& value) -> Operator* {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                // --- Construct a Join Operator ---

                // Determine the hash table caching strategy for this join.
                Hashmap* hashmap = query_cache->getHashmap(node_idx);
                QueryCache::CacheType type = query_cache->getCacheType(node_idx);
                // `use_cached_build` is true if we can reuse a hash table built by a previous query.
                bool use_cached_build = (type == QueryCache::USE_CACHE);
                // `store_hash_for_cache` is true if this query is the first to build the table,
                // indicating we should store hash values alongside the data for future reuse.
                bool store_hash_for_cache = (type == QueryCache::OWN_CACHE);

                // Normalize the build and probe side information. This simplifies the logic below.
                size_t build_node, probe_node, build_attr, probe_attr;
                std::vector<std::tuple<size_t, DataType>> output_attrs;
                if (value.build_left) {
                    build_node = value.left;
                    probe_node = value.right;
                    build_attr = value.left_attr;
                    probe_attr = value.right_attr;
                    output_attrs = node.output_attrs;
                } else {
                    // The right child is the build side.
                    build_node = value.right;
                    probe_node = value.left;
                    build_attr = value.right_attr;
                    probe_attr = value.left_attr;
                    // If the build/probe sides are swapped, the output column indices must be remapped
                    // to match the final output schema.
                    size_t left_size = plan.nodes[value.left].output_attrs.size();
                    size_t right_size = plan.nodes[value.right].output_attrs.size();
                    for (auto [col_idx, col_type] : node.output_attrs) {
                        output_attrs.emplace_back(col_idx >= left_size ? col_idx - left_size : col_idx + right_size, col_type);
                    }
                }

                // Recursively build the operator pipelines for the children.
                // The probe side is always built.
                Operator* probe_op = getOperator(plan, probe_node, shared_manager, false, query_cache, vector_size, input);

                if (use_cached_build) {
                    // Path 1: A pre-built hash table exists in the cache.
                    // We don't need a build-side pipeline; set the build operator to nullptr.
                    auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1, hashmap);
                    return new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                        shared, vector_size, nullptr, build_attr,
                        probe_op, probe_attr, output_attrs, plan.nodes[build_node].output_attrs, use_cached_build, store_hash_for_cache);
                } else {
                    // Path 2: No cached hash table available; we must build it.
                    // Recursively build the operator pipeline for the build side.
                    Operator* build_op = getOperator(plan, build_node, shared_manager, true, query_cache, vector_size, input);
                    if (build_op != nullptr) {
                        // Sub-path 2a: Standard Hash Join.
                        // The build side is a normal pipeline, so create a full Hashjoin operator.
                        auto& shared = shared_manager.get<Hashjoin::Shared>(node_idx + 1, hashmap);
                        return new (local_allocator.allocate(sizeof(Hashjoin))) Hashjoin(
                            shared, vector_size, build_op, build_attr,
                            probe_op, probe_attr, output_attrs, plan.nodes[build_node].output_attrs, use_cached_build, store_hash_for_cache);
                    } else {
                        // Sub-path 2b: Optimization for a single-row build side.
                        // The recursive call for the build side returned `nullptr`, signaling this optimization.
                        // Use a more lightweight `NaiveJoin` (a nested loop join), which is efficient here.
                        auto& shared = shared_manager.get<Naivejoin::Shared>(node_idx + 1);
                        ScanNode one_line_scan = std::get<ScanNode>(plan.nodes[build_node].data);
                        const ColumnarTable* table = (input) ? &(*input)[one_line_scan.base_table_id] : &plan.inputs[one_line_scan.base_table_id];

                        std::vector<const Column*> columns;
                        for (auto [col_idx, _] : plan.nodes[build_node].output_attrs) {
                            columns.push_back(&table->columns[col_idx]);
                        }
                        return new (local_allocator.allocate(sizeof(Naivejoin))) Naivejoin(
                            shared, vector_size, probe_op, probe_attr, columns, build_attr, output_attrs);
                    }
                }
            } else if constexpr (std::is_same_v<T, ScanNode>) {
                // --- Construct a Scan Operator ---
                // Determine the correct data source (either the plan's base tables or an intermediate result).
                const ColumnarTable* table = (input) ? &(*input)[value.base_table_id] : &plan.inputs[value.base_table_id];
                size_t row_num = table->num_rows;

                // Optimization: If this scan is on the build side of a join and has only one row,
                // return `nullptr`. This is the signal that tells the parent `JoinNode` to use
                // the more efficient `NaiveJoin` operator instead of a full `HashJoin`.
                if (row_num == 1 && is_build_side) {
                    return nullptr;
                }

                auto& shared = shared_manager.get<Scan::Shared>(node_idx + 1);
                // Populate the list of columns this scan operator needs to read.
                std::vector<const Column*> columns;
                for (auto [col_idx, _] : node.output_attrs) {
                    columns.push_back(&table->columns[col_idx]);
                }
                return new (local_allocator.allocate(sizeof(Scan))) Scan(shared, row_num, vector_size, columns);
            }
        },
        node.data);
}

/**
 * @brief Translates a logical plan into a full physical execution plan tree.
 * @return A pointer to the sink operator (`ResultWriter`) which is the root of the executable plan.
 */
ResultWriter* getPlan(const Plan& plan, SharedStateManager& shared_manager, size_t vector_size = 1024, QueryCache* query_cache = nullptr, const std::vector<ColumnarTable>* input = nullptr) {
    ProfileGuard profile_guard(global_profiler, "make plan");
    // Start recursion from the root of the logical plan.
    Operator* op = getOperator(plan, plan.root, shared_manager, false, query_cache, vector_size, input);
    // The sink operator (`ResultWriter`) always has a shared state ID of 0.
    auto& shared = shared_manager.get<ResultWriter::Shared>(0, plan.nodes[plan.root].output_attrs);
    return new (local_allocator.allocate(sizeof(ResultWriter))) ResultWriter(shared, op);
}

/**
 * @brief Heuristically determines the optimal number of threads for a query.
 *
 * It estimates the query size by summing the rows of all base table scans.
 * Larger queries are assigned more threads, up to a hardware-specific maximum.
 * @param plan The query plan.
 * @return The recommended number of threads to use.
 */
size_t threadNum(const Plan& plan) {
    size_t all_scan_size = 0;
    // Heuristic logic based on total scan size and hardware properties
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

CacheManager cache_manager; // Global cache manager for hash tables.

/**
 * @brief The main query execution function.
 *
 * This function orchestrates the entire query execution process:
 * 1. Determines the number of threads.
 * 2. Initializes barriers, shared states, and caches.
 * 3. Spawns worker threads. Each thread builds its own physical plan instance.
 * 4. All threads execute their plans in parallel.
 * 5. The last thread to finish consolidates and returns the result.
 * 6. Cleans up resources.
 *
 * @param plan The logical query plan to execute.
 * @param context A generic context pointer (unused in this implementation).
 * @return A `ColumnarTable` containing the query result.
 */
ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {
#ifdef DEBUG_LOG
    PlanPrinter::printTree(plan);
#endif

    size_t thread_num = threadNum(plan);
    const int vector_size = 1024;
    std::vector<Barrier*> barriers = Barrier::create(thread_num);
    SharedStateManager shared_manager;
    ColumnarTable result;
    QueryCache* query_cache = cache_manager.getQuery(plan);
    global_profiler = new Profiler(thread_num);
    global_mempool.reset();
    std::condition_variable finish_cv;
    std::mutex finish_mtx;
    bool finished = false;

    // --- Launch worker threads ---
    for (size_t i = 0; i < thread_num; ++i) {
        size_t barrier_group = i / Barrier::threads_per_barrier_;

        g_thread_pool->assign_task(i, [&, i, barrier_group]() { // Capture i and barrier_group
            local_allocator.reuse();
            global_profiler->set_thread_id(i);
            ProfileGuard profile_guard(global_profiler, "execute");

            // Each thread gets its assigned leaf barrier.
            current_barrier = barriers[barrier_group];
            // Each thread materializes its own instance of the physical plan.
            ResultWriter* result_writer = getPlan(plan, shared_manager, vector_size, query_cache);
            // Execute the plan by pulling data from the sink operator.
            result_writer->next();

            // Use the barrier to synchronize completion.
            bool is_last = current_barrier->wait();
            // The designated last thread is responsible for finalizing the result.
            if (is_last) {
                result = std::move(result_writer->shared_.output_);
                {
                    std::unique_lock lock(finish_mtx);
                    finished = true;
                    finish_cv.notify_all(); // Notify the main thread.
                }
            }
        });
    }

    // --- Wait for execution to complete ---
    {
        std::unique_lock lock(finish_mtx);
        finish_cv.wait(lock, [&]() { return finished; });
    }

    // --- Cleanup ---
    Barrier::destroy(barriers);
    cache_manager.cacheQuery(query_cache);
    global_profiler->print_profiles();
    delete global_profiler;
    global_profiler = nullptr;

    return result;
}

/**
 * @brief Initializes global resources needed for query execution.
 * @return A null context pointer.
 */
void* build_context() {
    global_mempool.init();
    if (g_thread_pool == nullptr) {
        g_thread_pool = new ThreadPool();
    }
    return nullptr;
}

/**
 * @brief Destroys global resources.
 * @param context A generic context pointer (unused).
 */
void destroy_context([[maybe_unused]] void* context) {
    delete g_thread_pool;
    g_thread_pool = nullptr;
    global_mempool.destroy();
}

} // namespace Contest