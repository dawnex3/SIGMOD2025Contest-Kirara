#pragma once

#include <atomic>
#include <vector>
#include "Profiler.hpp"

namespace Contest {

#ifndef CACHELINE_SIZE
    #define CACHELINE_SIZE 64
#endif

// 层级屏障
class Barrier {
public:
    Barrier* parent_;    // 父级屏障
    static constexpr size_t threads_per_barrier_ = 8;     // 单个屏障节点内的最大线程数
private:
    const size_t thread_count_;   // 本屏障节点内的实际线程数
    alignas(CACHELINE_SIZE) std::atomic<size_t> cntr_;   // 等待计数（对齐以减少伪共享）
    alignas(CACHELINE_SIZE) std::atomic<uint8_t> round_; // 轮次（对齐以减少伪共享）

public:
    explicit Barrier(std::size_t thread_count, Barrier* p = nullptr)
    : parent_(p), thread_count_(thread_count), cntr_(thread_count), round_(0)
    {}

    // 等待函数，最后一个线程执行 finalizer
    template <typename F> bool wait(F finalizer) {
        auto my_round = round_.load(std::memory_order_acquire);        // 记录当前轮次
        auto prev    = cntr_.fetch_sub(1, std::memory_order_acq_rel); // 计数值减一
        if (prev == 1) {
            // 当前线程为最后一个到达的线程，重置计数器
            cntr_.store(thread_count_, std::memory_order_relaxed);
            // 层次性调用：如果有父屏障，则等待父屏障，否则直接调用 finalizer
            if(parent_){
                bool ret = parent_->wait(finalizer);
                // 通知本Barrier中所有等待的线程，新轮次开始
                round_.fetch_add(1, std::memory_order_release);
                return ret;
            } else {
                finalizer();
                round_.fetch_add(1, std::memory_order_release);
                return true;
            }
        } else {
            // 非最后一个到达的线程等待新轮次
            global_profiler->event_begin("barrier wait");
            while (round_.load(std::memory_order_acquire) == my_round) {
                // 使用 PAUSE 指令降低功耗，避免忙等待过于激烈
                asm("pause");
                asm("pause");
                asm("pause");
            }
            global_profiler->event_end("barrier wait");
            return false;
        }
    }

    // 无参数的 wait 函数：默认返回 true
    inline bool wait() {
        return wait([]() { return true; });
    }

    // 递归创建含有nrThreads个线程的层次屏障，返回层次屏障最低层的所有屏障形成的列表（屏障树的叶子节点）
    static std::vector<Barrier*> create(size_t thread_num) {
        std::vector<Barrier*> result;
        // 如果nrThreads <= threadsPerBarrier，创建单个屏障即可
        if (thread_num <= threads_per_barrier_) {
            result.push_back(new Barrier(thread_num, nullptr));
            return result;
        }
        // 如果nrThreads > threadsPerBarrier，则本层需要创建totalBarriers个屏障。每个屏障不超过nrThreads个线程
        size_t full_barrier_num = thread_num / threads_per_barrier_;
        size_t threads_in_rest = thread_num % threads_per_barrier_;
        size_t total_barriers = full_barrier_num + (threads_in_rest > 0 ? 1 : 0);
        // 先创建上层屏障
        std::vector<Barrier *> parent_barriers = create(total_barriers);
        // 再创建当前层屏障
        for (size_t i = 0; i < full_barrier_num; ++i) {
            result.push_back(new Barrier(threads_per_barrier_, parent_barriers[i / threads_per_barrier_]));
        }
        if (threads_in_rest > 0)
            result.push_back(new Barrier(threads_in_rest, parent_barriers.back()));
        return result;
    }

    // 销毁一个层次屏障。输入是该屏障树的叶子节点
    static void destroy(std::vector<Barrier*>& barriers) {
        // 收集父级屏障（注意同一个 parent 可能被多个子节点引用）
        std::vector<Barrier*> parents;
        for (size_t i = 0; i < barriers.size(); i += threads_per_barrier_) {
            parents.push_back(barriers[i]->parent_);
        }
        // 删除叶子级屏障
        for (auto b : barriers) {
            delete b;
        }
        // 销毁父级屏障
        if (parents.size() > 1){
            destroy(parents);   // 父屏障上面还有层级
        } else {
            delete parents.back();  // 父屏障已经是根节点
        }
    }
};

thread_local Barrier* current_barrier;  // 线程局部存储的全局变量，表示每个线程的Barrier

}

