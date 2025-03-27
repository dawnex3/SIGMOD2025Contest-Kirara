# Announcements

### 2025-03-27
  - Please read the following notes on `build_context()`, third-party libraries, and the final evaluation carefully
  - **`build_context()`**:
    - The `build_context()` function cannot access any data, queries, or plans in any form.
    - Before the execution of `build_context()`, no computation by the participants' code is allowed, which means:
      - Global variables with constructors are not allowed.
      - Initializing global variables with functions is not allowed.
      - Any code that runs before `main()` is not allowed.
    - The `build_context()` function can create background tasks.
  - **Third-party libraries**:
    - It is not allowed to use third-party libraries already present in `CMakeLists.txt` as they are only for providing the general setup.
    - Learning from third-party libraries and writing related functions from scratch is allowed, but copying and modifying third-party libraries is not allowed.
  - **Final evaluation**:
    - As stated on the contest website, we will use a larger set of queries with different predicates for the final evaluation and increase the number of servers. Please do not make any assumptions about the execution and verification of queries as this might be different from the current evaluation.

### 2025-03-24
  - We are currently experiencing problems with the IBM machine. We are working on restoring the machine.

### 2025-03-18
  - With this pull request, we will evaluate all queries in the benchmark.
  - Several people asked what to expect in the final evaluation:
    - There will be no joins on any non-integer-column (similar to the original JoinOrder Benchmark)
    - There will be no "advanced tests" that cover edge cases which you need to pass
      - However, expect to handle any string that is part of the IMDB-JOB dataset
    - Benchmarks on the new evaluation servers are also limited to a single NUMA node
    - The data set will be the same as used right now without any further modifications
    - **Hardware headers**:
      - We create the `hardware.h` header in the Github workflow. For that reason there is no single `hardware.h` file as we have multiple (hidden) servers.
      - For local development, we recommend to take a suitable header file (e.g., `hardware__sidon.h` for Intel machines) and use that one as "your local" `hardware.h`. Please note that this file will be overwritten in the benchmark runs.
    - **Data Access and Global Context**:
      - You are not allowed to access the original data set or the queries in any way. Only data that is passed to you in `execute()`. A typical example for global context usage is creating a thread pool or scheduler.
      - The global context can further be used to store information that is gathered during query execution.

### 2025-03-13
  - We provide a new header, `hardware.h`. This header contains basic hardware information which enables optimizing for a server's cache sizes or vectorization capabilities.
    - If you miss any information or find issues with the headers, please do not hesitate to contact us.
  - As one of the goals of this contest is to write efficient code for multiple platforms (some of those are kept secret until the final evaluation), we encourage you to read about vector extensions (e.g., Clang's "Vectors and Extended Vectors").
  - We are considering changing the benchmark to include all queries of the standard JOB benchmark. We will reset the leaderboard in this case. We will let you know upfront when this change is about to land.
  - **Third-party libraries:**
    - We want to re-iterate our last notes from 2025-03-04: third-party libraries are **not allowed in your final submission**.
  - **Evaluation workload:**
    -  While there will be a larger variety of queries in the final evaluation workload, we will not add any "surprises". For example, as in the original JoinOrder Benchmark, there will be no joins on string columns.

### 2025-03-04
  - With today's changes to the main repository you forked from, we improved the performance of the evaluation phase
  - **Important notes:**
    - **Deadline change:** The deadline for the final submission has been extended to March 31
    - **Own source files**: The CMake file (which cannot be modified by participants) now includes all *.cpp fiels in the `src` directory. This way, you can add your own source files and better structure your code.
    - **Third-party library:** We found that some teams use third-party libraries, e.g., for  logging. Please note that third-party libraries are not allowed in the contest. You are free to use them during development, but you need to remove them prior to the final submission. Otherwise, your submission is disqualified.

### 2025-02-27
  - The recently pushed GitHub workflow will automatically compile, test, and benchmark your solution on all four systems
  - Check your repository's pull requests
  - The results are currently shown at https://sigmod-contest-25.hpi-sci.de/ and will soon be published on the official contest website
