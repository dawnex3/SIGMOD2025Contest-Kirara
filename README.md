# SIGMOD Contest 2025 Runner-up Solution

This project is **TeamKirara's** submission for the **SIGMOD Contest 2025**, achieving **2nd place** in the final evaluation. For detailed information on the contest problem, environment, and execution instructions, please refer to the official [SIGMOD Contest 2025 website](https://sigmod-contest-2025.github.io/).

## Team Members

Our team is composed of passionate students dedicated to high-performance database systems.

| Name         | University                      | Contact                    |
|--------------|---------------------------------|----------------------------|
| Xiang Liming | Beijing Institute of Technology | `dawnex@163.com`           |
| Feng Jing    | Beijing Institute of Technology | `1330827323@qq.com`        |
| Shao Yibo    | Beijing Institute of Technology | `1626295293@qq.com`        |
| Yu Yongze    | Xidian University               | `uuz163@gmail.com`         |
| Hou Jiaxi    | Beijing Institute of Technology | `jiaxihou0122@outlook.com` |

We are always open to collaboration and feedback!

## Key Features

*   **Vectorized Volcano-Style Execution Engine**: Implements a modern, pull-based query processing model. The engine is composed of physical operators that each process data in batches. The query plan is executed by pulling results from the root, creating a highly efficient, demand-driven data flow.

*   **Optimized Columnar Abstraction with Delayed Materialization**:
    *   **Unified Column Interface**: A carefully designed column abstraction allows operators to process data transparently, regardless of its physical storage.
    *   **Delayed Materialization**: By default, operators pass lightweight views which simply reference data locations within the original pages. Data is only materialized into a dense, contiguous buffer when an operation explicitly requires it. This strategy of **delaying materialization** significantly reduces data copying overhead and memory footprint throughout the query pipeline.

*   **Advanced Parallel Execution & Synchronization**:
    *   **Parallel-Aware Operators**: All operators are designed for multi-thread execution, using atomic operations for task distribution and resumable stages to fit the volcano model.
    *   **Adaptive Join Strategy**: The engine automatically selects the optimal join algorithm, such as using an optimized Nested-Loop Join for single-row build sides to avoid hash table overhead.
    *   **Hierarchical Barrier**: A custom tree-structured barrier is used for efficient, low-overhead synchronization of a large number of worker threads.
    *   **Task-Based Thread Pool**: A static thread pool manages a fixed set of worker threads, allowing for fine-grained task assignment and execution on multi-core systems.

*   **High-Performance Memory Management**:
    *   **Two-Level Memory Pool**: A high-performance memory pool that pre-allocates a large memory block (optionally with huge pages). Each worker thread then carves out its own private sub-pool, enabling ultra-fast, lock-free allocations via simple pointer arithmetic.
    *   **Custom STL Allocators**: Standard containers like std::vector can be easily configured to use this custom memory system, bypassing the overhead of conventional malloc/free calls and benefiting directly from the pool's performance.

*   **Highly Parallelized with SIMD**:
    *   Extensive use of SIMD (Single Instruction, Multiple Data) intrinsics to accelerate core computational primitives and data processing operations.

*   **Developer-Friendly Tooling**:
    *   **Comprehensive Code-Level Documentation**: The codebase is thoroughly documented with standardized, Doxygen-style comments to facilitate understanding and maintenance.
    *   **Built-in, Multi-Threaded Profiler**: A thread-safe profiler allows for detailed performance analysis of events and operator statistics (e.g., input/output row counts) on a per-thread basis, crucial for identifying bottlenecks.

## Implementation Details

### Operator Construction

* There are four types of operators: `Scan`, `HashJoin`, `NaiveJoin` and `ResultWriter`.

  * `Scan` is responsible for reading data.
  * `HashJoin` performs join operations using a hash table.
  * `NaiveJoin` is a simple nested-loop join for when one side has just one row.
  * `ResultWriter` is responsible for pulling all results from the pipeline and assembling the final output table.

* The `HashJoin` operator selects the smaller input as the Build Side and the larger one as the Probe Side. If the Build Side contains only one row, the system falls back to using the simpler `NaiveJoin`.

* Both `HashJoin` and `NaiveJoin` pre-allocate buffers to store their join results efficiently.

* Every thread maintains its own copy of the same operator tree. These operator trees are coordinated using a Shared State Manager to ensure consistent execution across threads.

### Operator Execution

* Operators follow a pull-based execution model: each operator actively pulls data from its upstream operator until there’s nothing left to process.

* The `Scan` operator atomically claim large data 'chunks' and then process them locally into smaller, lock-free batches, which minimizes synchronization overhead.

* The system uses late materialization: instead of copying data, `Scan` passes lightweight references upward in the operator tree. This data remains as references until an operator like `HashJoin` or `NaiveJoin` must materialize it into a temporary buffer to produce join results.

* Finally, the `ResultWriter` pulls these materialized result batches, and writes the result into its own private, paged-format buffers. Once a thread's pipeline is exhausted, it briefly locks a shared result structure to append its locally generated pages.

### Hash Table Construction and Probing

* The hash table is built from the smaller input side of the `HashJoin`. During probing, vectorized instructions (via Clang extensions) are used to compare keys in batches, improving performance.

* Multiple threads build the hash table in parallel, with each thread responsible for part of the data.

* The hash table is made up of multiple hash chains. Each chain holds key-value pairs with the same hash. To quickly check whether a chain might contain a given key, the first 16 bits of the chain's head pointer are repurposed as a mini Bloom filter.

* To prevent concurrency issues, memory for key-value pairs is allocated outside the hash table. Once the construction is done, all entries are inserted into the table using lock-free CAS (Compare-And-Swap) operations.

### Caching Execution Results of Repeated Subtrees

* Many query plans contain identical subtrees. By caching the execution results of these repeated subtrees, the system can reduce redundant computation and improve performance.

* The cache works like a hash table:

  * The key is a hash value derived from the structure of the subtree and samples of its input data.
  * The value is the result produced by executing the subtree.

* An LRU (Least Recently Used) policy is used to evict outdated entries when the cache is full.

### Memory Allocation and Pooling

* A global memory pool is used to handle SQL memory needs. Before execution begins, the pool pre-allocates a large memory block for the query.

* Each thread has its own allocator that requests memory from the global pool. Memory is allocated using a lock-free pointer increment method.

* When query execution finishes, all memory is released in one go. This design avoids frequent allocation/deallocation and helps reduce fragmentation.

### Thread Management

* A static thread pool is used to manage all threads. The number of threads is fixed, which avoids the overhead of creating and destroying threads repeatedly.

* Tasks can be assigned to specific threads using thread IDs, helping to minimize contention and potential conflicts between threads.

## Acknowledgment

Our approach draws inspiration from and builds upon the concepts presented in several influential researches. We would like to express our sincere gratitude to the authors of the following key papers for their invaluable insights:

* "Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask" by Timo Kersten, Viktor Leis, Alfons Kemper, Thomas Neumann, Andrew Pavlo, and Peter Boncz.

* "To Partition, or Not to Partition, That is the Join Question in a Real System" by Maximilian Bandle, Jana Giceva, and Thomas Neumann.

Additionally, our implementation references the open-source code from the repository https://github.com/TimoKersten/db-engine-paradigms/, and we thank its contributors for making their work available.

# SIGMOD Contest 2025

## Task

Given the joining pipeline and the pre-filtered input data, your task is to implement an efficient joining algorithm to accelerate the execution time of the joining pipeline. Specifically, you need to implement the following function in `src/execute.cpp`:

```C++
ColumnarTable execute(const Plan& plan, void* context);
```

Optionally, you can implement these two functions as well to prepare any global context (e.g., thread pool) to accelerate the execution.

```C++
void* build_context();
void destroy_context(void*);
```

### Input format

The input plan in the above function is defined as the following struct.

```C++
struct ScanNode {
    size_t base_table_id;
};

struct JoinNode {
    bool   build_left;
    size_t left;
    size_t right;
    size_t left_attr;
    size_t right_attr;
};

struct PlanNode {
    std::variant<ScanNode, JoinNode>          data;
    std::vector<std::tuple<size_t, DataType>> output_attrs;
};

struct Plan {
    std::vector<PlanNode>      nodes;
    std::vector<ColumnarTable> inputs;
    size_t root;
}
```

**Scan**:
- The `base_table_id` member refers to which input table in the `inputs` member of a plan is used by the Scan node.
- Each item in the `output_attrs` indicates which column in the base table should be output and what type it is.

**Join**:
- The `build_left` member refers to which side the hash table should be built on, where `true` indicates building the hash table on the left child, and `false` indicates the opposite.
- The `left` and `right` members are the indexes of the left and right child of the Join node in the `nodes` member of a plan, respectively.
- The `left_attr` and `right_attr` members are the join condition of Join node. Supposing that there are two records, `left_record` and `right_record`, from the intermediate results of the left and right child, respectively. The members indicate that the two records should be joined when `left_record[left_attr] == right_record[right_attr]`.
- Each item in the `output_attrs` indicates which column in the result of children should be output and what type it is. Supposing that the left child has $n_l$ columns and the right child has $n_r$ columns, the value of the index $i \in \{0, \dots, n_l + n_r - 1\}$, where the ranges $\{0, \dots, n_l - 1\}$ and $\{n_l, \dots, n_l + n_r - 1\}$ indicate the output column is from left and right child respectively.

**Root**: The `root` member of a plan indicates which node is the root node of the execution plan tree.

### Data format

The input and output data both follow a simple columnar data format.

```C++
enum class DataType {
    INT32,       // 4-byte integer
    INT64,       // 8-byte integer
    FP64,        // 8-byte floating point
    VARCHAR,     // string of arbitary length
};

constexpr size_t PAGE_SIZE = 8192;

struct alignas(8) Page {
    std::byte data[PAGE_SIZE];
};

struct Column {
    DataType           type;
    std::vector<Page*> pages;
};

struct ColumnarTable {
    size_t              num_rows;
    std::vector<Column> columns;
};
```

A `ColumnarTable` first stores how many rows the table has in the `num_rows` member, then stores each column seperately as a `Column`. Each `Column` has a type and stores the items of the column into several pages. Each page is of 8192 bytes. In each page:

- The first 2 bytes are a `uint16_t` which is the number of rows $n_r$ in the page.
- The following 2 bytes are a `uint16_t` which is the number of non-`NULL` values $n_v$ in the page.
- The first $n_r$ bits in the last $\left\lfloor\frac{(n_r + 7)}{8}\right\rfloor$ bytes is a bitmap indicating whether the corresponding row has value or is `NULL`.

**Fixed-length attribute**: There are $n_v$ contiguous values begins at the first aligned position. For example, in a `Page` of `INT32`, the first value is at `data + 4`. While in a `Page` of `INT64` and `FP64`, the first value is at `data + 8`.

**Variable-length attribute**: There are $n_v$ contigous offsets (`uint16_t`) begins at `data + 4` in a `Page`, followed by the content of the varchars which begins at `char_begin = data + 4 + n_r * 2`. Each offset indicates the ending offset of the corresponding `VARCHAR` with respect to the `char_begin`.

**Long string**: When the length of a string is longer than `PAGE_SIZE - 7`, it can not fit in a normal page. Special pages will be used to store such string. If $n_r$ `== 0xffff` or $n_r$ `== 0xfffe`, the `Page` is a special page for long string. `0xffff` means the page is the first page of a long string and `0xfffe` means the page is the following page of a long string. The following 2 bytes is a `uint16_t` indicating the number of chars in the page, beginning at `data + 4`.

## Requirement

- You can only modify the file `src/execute.cpp` in the project.
- You must not use any third-party libraries. If you are using libraries for development (e.g., for logging), ensure to remove them before the final submission.
- The joining pipeline (including order and build side) is optimized by PostgreSQL for `Hash Join` only. However, in the `execute` function, you are free to use other algorithms and change the pipeline, as long as the result is equivalent.
- For any struct listed above, all of there members are public. You can manipulate them in free functions as desired as long as the original files are not changed and the manipulated objects can be destructed properly.
- Your program will be evaluated on an unpublished benchmark sampled from the original JOB benchmark. You will not be able to access the test benchmark.

## Quick start

> [!TIP]
> Run all the following commands in the root directory of this project.

First, download the imdb dataset.

```bash
./download_imdb.sh
```

Second, build the project.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -- -j $(nproc)
```

Third, prepare the DuckDB database for correctness checking.

```bash
./build/build_database imdb.db
```

Now, you can run the tests:
```bash
./build/run plans.json
```
> [!TIP]
> If you want to use `Ninja Multi-Config` as the generator. The commands will look like:
> 
>```bash
> cmake -S . -B build -Wno-dev -G "Ninja Multi-Config"
> cmake --build build --config Release -- -j $(nproc)
> ./build/Release/build_database imdb.db
> ./build/Release/run plans.json
> ```

# Hardware

The evaluation is automatically executed on four different servers. On multi-socket machines, the benchmarks are bound to a single socket (using `numactl -m 0 -N 0`).

 * **Intel #1**
    * CPU: 4x Intel Xeon E7-4880 v2 (SMT 2, 15 cores, 30 threads)
    * Main memory: 512 GB
 * **AMD #1**
    * CPU: 2x AMD EPYC 7F72 (SMT 2, 24 cores, 48 threads)
    * Main memory: 256 GB
 * **IBM #1**
    * CPU: 8x IBM Power8 (SMT 8, 12 cores, 96 threads)
    * Main memory: 1024 GB
 * **ARM #1**
    * CPU: 1x Ampere Altra Max (SMT 1, 128 cores, 128 threads)
    * Main memory: 512 GB


For the final evaluation after the submission deadline, four additional servers will be included. These additional servers cover the same platforms but might differ in the supported feature set as they can be significantly older or newer than the initial servers. All servers run Ubuntu Linux with versions ranging from 20.04 to 24.04. Code is compiled with Clang 18.
