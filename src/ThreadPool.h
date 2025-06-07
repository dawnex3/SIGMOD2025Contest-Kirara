/**
* @file ThreadPool.hpp
 * @brief A static thread pool implementation with a fixed number of threads.
 * 
 * This class manages a pool of threads, each with its own task slot, mutex, and condition variable.
 * Tasks can be assigned to specific threads, and each thread waits for its assigned task to execute.
 * A global variable `g_thread_pool` is initialized when build_context() is called, and is provided 
 * for use in the application.
 * 
 * @tparam ThreadCount The number of threads in the pool.
 * 
 * Usage:
 * - Assign tasks to specific threads using assign_task().
 * - The destructor ensures all threads are joined and resources are cleaned up.
 * 
 * Thread safety:
 * - Each thread/task slot is protected by its own mutex and condition variable.
 * 
 * Example:
 * @code
 * g_thread_pool.assign_task(0, []{ task code });
 * @endcode
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include "hardware.h"
#include "MemoryPool.hpp"

/**
 * @class StaticThreadPool
 * @brief A thread pool with a fixed number of threads and per-thread task assignment.
 *
 * This implementation creates a fixed number of worker threads upon construction.
 * Unlike a typical thread pool with a shared work queue, this design assigns a
 * dedicated task slot to each thread. A task must be explicitly assigned to a
 * specific worker thread by its index. This model is useful when there is a
 * known, fixed set of long-running, parallel tasks.
 *
 * @tparam ThreadCount The number of threads in the pool, determined at compile time.
 */
template <size_t ThreadCount>
class StaticThreadPool {
public:
    /**
     * @brief Constructs the thread pool and starts all worker threads.
     *
     * Each worker thread is immediately started and enters a loop where it waits
     * on its personal condition variable for a task to be assigned or for a
     * shutdown signal from the destructor. Each thread also initializes its
     * own thread-local memory allocator.
     */
    StaticThreadPool() : stop_flag(false) {
        for (size_t i = 0; i < ThreadCount; ++i) {
            workers[i] = std::thread([this, i] {
                // Each thread gets its own instance of the memory allocator.
                local_allocator.init(&global_mempool);
                std::unique_lock<std::mutex> lock(mutexes[i]);

                // Main worker loop.
                while (!stop_flag) {
                    // Wait until either the pool is stopping or a task is assigned to this specific thread.
                    conditions[i].wait(lock, [this, i] { return stop_flag || tasks[i]; });

                    // Check for spurious wakeups or shutdown condition.
                    if (stop_flag && !tasks[i]) {
                        break;
                    }

                    // Execute the assigned task.
                    tasks[i]();
                    // Clear the task slot to indicate completion.
                    tasks[i] = nullptr;
                }
            });
        }
    }

    /**
     * @brief Destroys the thread pool, ensuring all worker threads are properly joined.
     *
     * This signals all threads to stop, wakes them up from their waiting state,
     * and waits for each thread to finish its execution.
     */
    ~StaticThreadPool() {
        // Signal all threads to stop.
        stop_flag = true;
        // Wake up all threads that might be waiting on their condition variables.
        for (auto& condition : conditions) {
            condition.notify_one(); // notify_one is sufficient since only one thread waits on each
        }
        // Wait for each worker thread to complete.
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    /**
     * @brief Assigns a task to a specific worker thread.
     *
     * This method does not use a shared queue. It places the task directly into
     * the designated thread's slot and notifies that single thread to wake up
     * and execute it.
     *
     * @tparam Func A callable type (e.g., lambda, function pointer).
     * @param thread_index The index of the worker thread to assign the task to.
     * @param func The task to be executed.
     * @throws std::runtime_error if a task is already assigned to the target thread.
     */
    template <typename Func>
    void assign_task(size_t thread_index, Func&& func) {
        // If the index is out of bounds, execute the function immediately in the calling thread.
        if (thread_index >= ThreadCount) {
            func();
        } else {
            {
                // Lock the specific mutex for the target thread's task slot.
                std::unique_lock<std::mutex> lock(mutexes[thread_index]);
                // This pool model does not queue tasks; a thread must be idle to receive a new one.
                if (tasks[thread_index] != nullptr) {
                    printf("Thread %zu already has a task assigned\n", thread_index);
                    throw std::runtime_error("Thread already has a task assigned");
                }
                // Assign the task and wake up the corresponding worker thread.
                tasks[thread_index] = std::forward<Func>(func);
                conditions[thread_index].notify_one();
            }
        }
    }

private:
    /// @brief The array of worker thread objects.
    std::array<std::thread, ThreadCount> workers;
    /// @brief An array of task slots, one for each worker thread.
    std::array<std::function<void()>, ThreadCount> tasks{};
    /// @brief An array of mutexes, each protecting the corresponding task slot in `tasks`.
    std::array<std::mutex, ThreadCount> mutexes;
    /// @brief An array of condition variables, each used to wake up a specific worker thread.
    std::array<std::condition_variable, ThreadCount> conditions;
    /// @brief An atomic flag used to signal all worker threads to terminate.
    std::atomic<bool> stop_flag;
};

/**
 * @brief A convenience type alias for the `StaticThreadPool` with a concrete size.
 *
 * The number of threads is determined by the hardware constant `SPC__THREAD_COUNT`,
 * capped at a maximum of 64 to prevent over-subscription.
 */
using ThreadPool = StaticThreadPool<std::min(SPC__THREAD_COUNT, 64)>;

/**
 * @brief A global pointer to the singleton instance of the thread pool.
 *
 * It is initialized in `build_context()` and destroyed in `destroy_context()`.
 */
ThreadPool* g_thread_pool = nullptr;

