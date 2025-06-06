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

template <size_t ThreadCount>
class StaticThreadPool {
public:
    StaticThreadPool() : stop_flag(false) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ThreadCount; ++i) {
            workers[i] = std::thread([this, i] {
                local_allocator.init(&global_mempool);
                std::unique_lock<std::mutex> lock(mutexes[i]);
                
                while (!stop_flag) {
                        conditions[i].wait(lock, [this, i] { return stop_flag || tasks[i]; });

                        if (stop_flag && !tasks[i]) {
                            break;
                        }

                        tasks[i]();
                        tasks[i] = nullptr;
                }
            });
        }
    }

    ~StaticThreadPool() {
        stop_flag = true;
        for (auto& condition : conditions) {
            condition.notify_all();
        }
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template <typename Func>
    void assign_task(size_t thread_index, Func&& func) {
        if (thread_index >= ThreadCount) {
            func();
        } else {
            {
                std::unique_lock<std::mutex> lock(mutexes[thread_index]);
                if (tasks[thread_index] != nullptr) {
                    printf("Thread %zu already has a task assigned\n", thread_index);
                    throw std::runtime_error("Thread already has a task assigned");
                } 
                tasks[thread_index] = std::forward<Func>(func);
                conditions[thread_index].notify_all();
            }
        }
    }

private:
    std::array<std::thread, ThreadCount> workers;
    std::array<std::function<void()>, ThreadCount> tasks{};
    std::array<std::mutex, ThreadCount> mutexes;
    std::array<std::condition_variable, ThreadCount> conditions;
    std::atomic<bool> stop_flag;
};

using ThreadPool = StaticThreadPool<std::min(SPC__THREAD_COUNT, 64)>;

ThreadPool *g_thread_pool = nullptr;

