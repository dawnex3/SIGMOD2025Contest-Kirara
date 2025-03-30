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
#include <hardware.h>
#include "MemoryPool.hpp"

template <size_t ThreadCount>
class StaticThreadPool {
public:
    StaticThreadPool() : stop_flag(false) {
        printf("Creating threadPool with %zu threads\n", ThreadCount);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < ThreadCount; ++i) {
            workers[i] = std::thread([this, i] {
                printf("Thread %zu started\n", i);
                local_allocator.init(&global_mempool);
                while (!stop_flag) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutexes[i]);
                        conditions[i].wait(lock, [this, i] { return stop_flag || tasks[i]; });

                        if (stop_flag && !tasks[i]) {
                            break;
                        }

                        task = std::move(tasks[i]);
                        tasks[i] = nullptr;
                    }

                    printf("got a task in thread %zu\n", i);
                    task();
                }
                printf("Thread %zu stop\n", i);
            });
        }
    }

    ~StaticThreadPool() {
        printf("Destroying threadPool\n");
        stop_flag = true;
        for (auto& condition : conditions) {
            condition.notify_all();
        }
        printf("Waiting for threads to finish\n");
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        printf("ThreadPool destroyed successfully\n");
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
            }
            conditions[thread_index].notify_all();
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

