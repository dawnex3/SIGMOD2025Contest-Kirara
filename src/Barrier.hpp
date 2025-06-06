
/**
 * @file Barrier.hpp
 * @brief Defines a hierarchical barrier synchronization primitive for multi-threaded environments.
 *
 * This file provides the implementation of a hierarchical barrier, which allows efficient synchronization
 * of a large number of threads by organizing them in a tree structure. Each barrier node can synchronize
 * up to `threads_per_barrier_` threads, and nodes can be composed hierarchically to support more threads.
 *
 * Usage:
 * - Thread-local variable `current_barrier` is provided for convenience to store each thread's assigned barrier.
 * - Use `current_barrier->wait()` to block the thread until all threads have reached this point.
 */
 
#pragma once

#include <atomic>
#include <vector>
#include "Profiler.hpp"

namespace Contest {

#ifndef CACHELINE_SIZE
    #define CACHELINE_SIZE 64
#endif

// Hierarchical barrier
class Barrier {
public:
    Barrier* parent_;                                       // Parent barrier
    static constexpr size_t threads_per_barrier_ = 8;       // Maximum number of threads per barrier node
private:
    const size_t thread_count_;                             // Actual number of threads in this barrier node
    alignas(CACHELINE_SIZE) std::atomic<size_t> cntr_;      // Wait counter (aligned to reduce false sharing)
    alignas(CACHELINE_SIZE) std::atomic<uint8_t> round_;    // Round (aligned to reduce false sharing)

public:
    explicit Barrier(std::size_t thread_count, Barrier* p = nullptr)
    : parent_(p), thread_count_(thread_count), cntr_(thread_count), round_(0)
    {}

    // Wait function, the last thread executes the finalizer
    template <typename F> bool wait(F finalizer) {
        auto my_round = round_.load(std::memory_order_acquire);        // Record current round
        auto prev    = cntr_.fetch_sub(1, std::memory_order_acq_rel); // Decrement counter
        if (prev == 1) {
            // This thread is the last to arrive, reset the counter
            cntr_.store(thread_count_, std::memory_order_relaxed);
            // Hierarchical call: if there is a parent barrier, wait on the parent barrier, otherwise call finalizer directly
            if(parent_){
                bool ret = parent_->wait(finalizer);
                // Notify all waiting threads in this Barrier that a new round has started
                round_.fetch_add(1, std::memory_order_release);
                return ret;
            } else {
                finalizer();
                round_.fetch_add(1, std::memory_order_release);
                return true;
            }
        } else {
            // Non-last threads wait for the new round
            global_profiler->event_begin("barrier wait");
            while (round_.load(std::memory_order_acquire) == my_round) {
                // Use yield instruction to reduce power consumption and avoid aggressive busy-waiting
                std::this_thread::yield();
            }
            global_profiler->event_end("barrier wait");
            return false;
        }
    }

    // Wait function without arguments: returns true by default
    inline bool wait() {
        return wait([]() { return true; });
    }

    // Recursively create a hierarchical barrier for nrThreads threads, return a list of all leaf barriers (leaf nodes of the barrier tree)
    static std::vector<Barrier*> create(size_t thread_num) {
        std::vector<Barrier*> result;
        // If nrThreads <= threadsPerBarrier, create a single barrier
        if (thread_num <= threads_per_barrier_) {
            result.push_back(new Barrier(thread_num, nullptr));
            return result;
        }
        // If nrThreads > threadsPerBarrier, create totalBarriers barriers at this level. Each barrier has at most nrThreads threads
        size_t full_barrier_num = thread_num / threads_per_barrier_;
        size_t threads_in_rest = thread_num % threads_per_barrier_;
        size_t total_barriers = full_barrier_num + (threads_in_rest > 0 ? 1 : 0);
        // First create parent barriers
        std::vector<Barrier *> parent_barriers = create(total_barriers);
        // Then create current level barriers
        for (size_t i = 0; i < full_barrier_num; ++i) {
            result.push_back(new Barrier(threads_per_barrier_, parent_barriers[i / threads_per_barrier_]));
        }
        if (threads_in_rest > 0)
            result.push_back(new Barrier(threads_in_rest, parent_barriers.back()));
        return result;
    }

    // Destroy a hierarchical barrier. Input is the leaf nodes of the barrier tree
    static void destroy(std::vector<Barrier*>& barriers) {
        // Collect parent barriers (note that the same parent may be referenced by multiple child nodes)
        std::vector<Barrier*> parents;
        for (size_t i = 0; i < barriers.size(); i += threads_per_barrier_) {
            parents.push_back(barriers[i]->parent_);
        }
        // Delete leaf barriers
        for (auto b : barriers) {
            delete b;
        }
        // Destroy parent barriers
        if (parents.size() > 1){
            destroy(parents);   // There are more levels above the parent barrier
        } else {
            delete parents.back();  // Parent barrier is already the root node
        }
    }
};

thread_local Barrier* current_barrier;  // Thread-local global variable, represents the Barrier for each thread

}

