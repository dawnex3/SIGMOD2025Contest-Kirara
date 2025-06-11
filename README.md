# SIGMOD Contest 2025 Runner-up Solution

> [!IMPORTANT]
> 
> This branch is a prototype implementation of our team's push-based execution engine. It is not the final submitted implementation and it's not fully tested.

We tried a push-based execution engine during the contest and implemented a simple prototype, here are some notes from the implementation.


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

## Notes
*   **Morsel-Driven:** Queries are processed on small, independent chunks of data (morsels or partitions), enabling fine-grained parallelism and better cache utilization.
*   **Parallel Push-Based Execution:** A scheduler decomposes query plans into pipelines (in this contest, there two kinds of pipelines: `ScanProbeTabularPipeline` and `ScanProbeBuildPipeline`) and dispatches tasks (morsels) to a pool of worker threads for concurrent execution. Operators in pipelines push data to downstream operators as soon as it's processed, minimizing control overhead and promoting data flow.
* **Column Vectors & Page Abstraction:** Designed to work with data stored in a paged format, with specialized readers for fixed-width types and different string encodings (compact and long strings).
*   **Hash-Based Joins:** HashJoin is implemented using `ChainedHashMap` with segment vectors for concurrent hash table construction.

## Acknowledgments

We would like to express our sincere gratitude to our mentor, **[Professor Hongchao Qin](https://qinhc.github.io/)** of the Beijing Institute of Technology. While this project was developed entirely by the student team, his mentorship and support were invaluable.

Our approach draws inspiration from and builds upon the concepts presented in several influential researches. We would like to express our sincere gratitude to the authors of the following key papers for their invaluable insights:

* "Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask" by Timo Kersten, Viktor Leis, Alfons Kemper, Thomas Neumann, Andrew Pavlo, and Peter Boncz.

* "To Partition, or Not to Partition, That is the Join Question in a Real System" by Maximilian Bandle, Jana Giceva, and Thomas Neumann.

* "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age" by Viktor Leis, Peter Boncz, Alfons Kemper, and Thomas Neumann,

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
