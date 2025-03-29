#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <hardware.h>
#include "MemoryPool.hpp"

class SimpleThreadPool {
public:
    // 构造函数：初始化线程池并启动指定数量的线程
    explicit SimpleThreadPool(size_t threadCount) : stop(false) {
        for (size_t i = 0; i < threadCount; ++i) {
            workers.emplace_back([this]() {
              local_allocator.init(&global_mempool);
                while (true) {
                  std::function<void()> task;

                  {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    condition.wait(lock,
                                   [this]() { return stop || !tasks.empty(); });

                    if (stop && tasks.empty())
                      return;

                    task = std::move(tasks.front());
                    tasks.pop();
                  }

                  task();
                }
            });
        }
    }

    // 向线程池添加任务
    template <typename F>
    void
    enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (stop)
                throw std::runtime_error("Enqueue on stopped ThreadPool");

            tasks.emplace([f = std::forward<F>(f)]() { f(); });
        }

        condition.notify_one();
    }

    // 析构函数：停止所有线程并等待它们完成
    ~SimpleThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        condition.notify_all();

        for (std::thread& worker : workers) worker.join();
    }

private:
    std::vector<std::thread> workers;         // 工作线程
    std::queue<std::function<void()>> tasks;  // 任务队列
    std::mutex queueMutex;                    // 任务队列互斥锁
    std::condition_variable condition;        // 条件变量
    std::atomic<bool> stop;                   // 停止标志
};


template <size_t ThreadCount>
class StaticThreadPool {
public:
    StaticThreadPool() : stop_flag(false) {
        for (size_t i = 0; i < ThreadCount; ++i) {
            workers[i] = std::thread([this, i] {
              local_allocator.init(&global_mempool);
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutexes[i]);
                        conditions[i].wait(lock, [this, i] { return stop_flag || tasks[i]; });

                        if (stop_flag && !tasks[i]) {
                            return;
                        }

                        task = std::move(tasks[i]);
                        tasks[i] = nullptr;
                    }

                    task();
                }
            });
            pthread_t thread = workers[i].native_handle();
            int ret = pthread_setname_np(pthread_self(),  "fast-worker");
            if (ret != 0) {
                throw std::runtime_error("failed to set thread name");
            }
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
                tasks[thread_index] = std::forward<Func>(func);
            }
            conditions[thread_index].notify_one();
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

thread_local ThreadPool *g_thread_pool = nullptr;

