/**
* @file Barrier.hpp
* @brief Defines a hierarchical barrier synchronization primitive for multi-threaded environments.
*
* This file provides the implementation of a hierarchical barrier, which allows efficient synchronization
* of a large number of threads by organizing them in a tree structure. Each barrier node can synchronize
* up to `threads_per_barrier_` threads, and nodes can be composed hierarchically to support more threads.
*
* Usage:
* - A tree of barriers is created using `Barrier::create(num_threads)`.
* - Each thread is assigned a unique leaf barrier from the returned vector.
* - This pointer should be stored in the thread-local variable `current_barrier`.
* - Use `current_barrier->wait()` to block the thread until all threads in the system have reached this point.
*/

#pragma once

#include <atomic>
#include <vector>
#include <thread> // Required for std::this_thread
#include "Profiler.hpp"

namespace Contest {

#ifndef CACHELINE_SIZE
   // Define a default cache line size to prevent false sharing.
   // False sharing occurs when unrelated atomic variables are located on the same cache line,
   // causing unnecessary cache invalidations and performance degradation.
   #define CACHELINE_SIZE 64
#endif

/**
* @class Barrier
* @brief A single node in a hierarchical barrier tree.
*/
class Barrier {
public:
   /// @brief Pointer to the parent barrier in the hierarchy. Null for the root node.
   Barrier* parent_;
   /// @brief The branching factor of the barrier tree; max number of threads per node.
   static constexpr size_t threads_per_barrier_ = 8;

private:
   /// @brief The actual number of threads this specific barrier node synchronizes.
   const size_t thread_count_;

   /// @brief Atomic counter for arriving threads. Aligned to a cache line to prevent false sharing with `round_`.
   alignas(CACHELINE_SIZE) std::atomic<size_t> cntr_;

   /// @brief Signals the current synchronization round. Waiting threads poll this variable.
   /// Aligned to a separate cache line to prevent false sharing with `cntr_`.
   alignas(CACHELINE_SIZE) std::atomic<uint8_t> round_;

public:
   /**
    * @brief Constructs a barrier node.
    * @param thread_count The number of threads this node is responsible for.
    * @param p A pointer to the parent barrier node, or `nullptr` if this is a root node.
    */
   explicit Barrier(std::size_t thread_count, Barrier* p = nullptr)
   : parent_(p), thread_count_(thread_count), cntr_(thread_count), round_(0)
   {}

   /**
    * @brief Blocks the calling thread until all threads in the barrier have arrived.
    *
    * The last thread to arrive at the root of the barrier tree will execute the `finalizer`.
    *
    * @tparam F The type of the finalizer function (a callable object).
    * @param finalizer The function to be executed by the last arriving thread at the root.
    * @return `true` if this thread was the one to execute the finalizer, `false` otherwise.
    */
   template <typename F> bool wait(F finalizer) {
       // Capture the current round number. An acquire load ensures we see the updated
       // round value written by the last thread from the *previous* barrier wait.
       auto my_round = round_.load(std::memory_order_acquire);

       // Decrement the counter. An acquire-release operation ensures that this
       // atomic operation synchronizes with other threads decrementing the counter.
       if (cntr_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
           // This is the last thread to arrive at this barrier node.

           // Propagate the wait to the parent barrier if it exists.
           bool is_root_finalizer = true;
           if (parent_) {
               is_root_finalizer = parent_->wait(finalizer);
           } else {
               // This is the root barrier, so execute the finalizer.
               finalizer();
           }

           // Reset the counter for the next round. Relaxed memory order is sufficient
           // because the subsequent release on `round_` will synchronize memory.
           cntr_.store(thread_count_, std::memory_order_relaxed);

           // Signal waiting threads to proceed by incrementing the round.
           // A release operation ensures that all previous writes (including the finalizer's effects)
           // are visible to the waiting threads that will now perform an acquire load on `round_`.
           round_.fetch_add(1, std::memory_order_release);
           return is_root_finalizer;
       } else {
           // This is not the last thread, so we must wait.
           global_profiler->event_begin("barrier wait");
           // Wait until the last thread signals the start of the next round.
           // The acquire load synchronizes with the release `fetch_add` from the last thread,
           // ensuring we see all its prior writes (like the finalizer's effects).
           while (round_.load(std::memory_order_acquire) == my_round) {
               // Hint to the scheduler that we can yield the CPU to another thread
               // to avoid aggressive, power-hungry busy-waiting.
               std::this_thread::yield();
           }
           global_profiler->event_end("barrier wait");
           return false;
       }
   }

   /**
    * @brief A convenience overload for `wait` that uses a no-op finalizer.
    * @return `true` if this thread was the last to arrive at the root, `false` otherwise.
    */
   inline bool wait() {
       return wait([]() {});
   }

   /**
    * @brief Factory function to recursively build a tree of barriers.
    * @param thread_num The total number of threads to synchronize.
    * @return A vector of pointers to the leaf `Barrier` nodes. Each thread should be
    *         assigned one of these leaf barriers. The caller is responsible for deallocation
    *         using `Barrier::destroy`.
    */
   static std::vector<Barrier*> create(size_t thread_num) {
       std::vector<Barrier*> leaves;
       // Base case: If the number of threads is small enough, create a single root barrier.
       if (thread_num <= threads_per_barrier_) {
           leaves.push_back(new Barrier(thread_num, nullptr));
           return leaves;
       }

       // Recursive step:
       // 1. Calculate how many barrier nodes are needed at the level above this one.
       size_t full_barrier_num = thread_num / threads_per_barrier_;
       size_t threads_in_rest = thread_num % threads_per_barrier_;
       size_t parent_count = full_barrier_num + (threads_in_rest > 0 ? 1 : 0);

       // 2. Recursively create the parent level.
       std::vector<Barrier*> parents = create(parent_count);

       // 3. Create the leaf nodes for the current level and link them to their parents.
       for (size_t i = 0; i < full_barrier_num; ++i) {
           // Assign each group of `threads_per_barrier_` children to one parent.
           leaves.push_back(new Barrier(threads_per_barrier_, parents[i / threads_per_barrier_]));
       }
       if (threads_in_rest > 0) {
           leaves.push_back(new Barrier(threads_in_rest, parents.back()));
       }
       return leaves;
   }

   /**
    * @brief Recursively deallocates all barriers in the tree.
    * @param barriers A vector of leaf barrier nodes, as returned by `create`.
    */
   static void destroy(std::vector<Barrier*>& barriers) {
       if (barriers.empty()) {
           return;
       }

       // Collect unique parent pointers from the current level of barriers.
       std::vector<Barrier*> parents;
       if (barriers[0]->parent_) { // Check if there is a parent level to destroy.
           for (size_t i = 0; i < barriers.size(); i += threads_per_barrier_) {
               parents.push_back(barriers[i]->parent_);
           }
       }

       // Delete the barriers at the current level.
       for (auto b : barriers) {
           delete b;
       }
       barriers.clear();

       // Recursively destroy the parent level.
       if (!parents.empty()) {
           destroy(parents);
       }
   }
};

/**
* @brief Thread-local pointer to the specific barrier node assigned to the current thread.
*
* This allows threads to call `current_barrier->wait()` without needing to pass the
* barrier pointer around.
*/
thread_local Barrier* current_barrier;

} // namespace Contest