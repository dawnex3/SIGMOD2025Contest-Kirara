/**
* @file SharedState.hpp
* @brief Defines a thread-safe management system for operator-specific shared states.
*
* In a parallel query engine, multiple threads often execute the same logical operator
* (e.g., multiple threads performing a single table scan). These threads need a way to
* coordinate and share information, such as the current scan position or a shared hash table.
*
* This file provides a robust mechanism for this:
*
* - **`SharedState`:** An abstract base class. Each physical operator (like `Scan` or
*   `Hashjoin`) defines its own derived struct (e.g., `Scan::Shared`) to hold the
*   data that all its worker threads need to share.
*
* - **`SharedStateManager`:** A thread-safe factory and registry for all `SharedState`
*   objects within a single query execution. It uses a "get-or-create" pattern: the
*   first thread to request the state for a particular operator causes it to be created,
*   and all subsequent threads receive a reference to that same, single instance.
*   This prevents race conditions and ensures all threads work on the same shared data.
*   Type safety is enforced at runtime via `dynamic_cast`.
*/
#pragma once

#include "mutex"
#include "unordered_map"
#include "atomic"
#include "memory"
#include "stdexcept"
#include <cstddef>

namespace Contest {

/**
* @class SharedState
* @brief An abstract base class for state that is shared across multiple threads
*        executing the same logical operator.
*/
class SharedState {
private:
   friend class SharedStateManager;
   /// @brief The unique ID of the operator this state belongs to. Set by the SharedStateManager.
   size_t operator_id_{};

public:
   virtual ~SharedState() = default;

   /// @brief Gets the unique ID of the operator associated with this state.
   [[nodiscard]] inline size_t get_operator_id() const { return operator_id_; }
};

/**
* @class SharedStateManager
* @brief A thread-safe factory and registry for all `SharedState` objects in a query plan.
*
* This manager ensures that for any given operator ID, exactly one `SharedState`
* object is created and shared among all worker threads.
*/
class SharedStateManager {
private:
   /// @brief A mutex to protect concurrent access to the state map.
   std::mutex m_;
   /// @brief The central map storing all shared states, keyed by operator ID.
   /// `std::unique_ptr` ensures that states are automatically deallocated when the manager is destroyed.
   std::unordered_map<size_t, std::unique_ptr<SharedState>> state_map_;

public:
   /**
    * @brief Gets or creates a shared state object for a given operator ID.
    *
    * This method is thread-safe. If the state for the given ID does not exist,
    * it will be created using the provided constructor arguments. If it already
    * exists, the existing instance is returned and the arguments are ignored.
    *
    * @tparam T The concrete `SharedState` derived type to retrieve or create (e.g., `Scan::Shared`).
    * @tparam Args The types of the arguments to forward to `T`'s constructor if creation is needed.
    * @param i The unique ID of the operator.
    * @param args The constructor arguments for `T`.
    * @return A reference to the requested shared state object.
    * @throws std::runtime_error if a state with the given ID exists but has a different type than requested.
    */
   template <typename T, typename... Args>
   T& get(size_t i, Args&&... args) {
       std::lock_guard<std::mutex> lock(m_);
       auto it = state_map_.find(i);

       // If the state for this ID doesn't exist, create it.
       if (it == state_map_.end()) {
           // Use perfect forwarding to pass constructor arguments.
           it = state_map_.emplace(i, std::make_unique<T>(std::forward<Args>(args)...)).first;
           // The manager uses its friend access to set the private operator ID.
           it->second->operator_id_ = i;
       }

       // Perform a runtime type check to ensure the caller is asking for the correct state type.
       T* res = dynamic_cast<T*>(it->second.get());
       if (!res) {
           // This is a critical error, indicating a logic bug in the plan construction.
           throw std::runtime_error("Failed to retrieve shared state. Wrong type found for operator ID " + std::to_string(i));
       }
       return *res;
   }
};
} // namespace Contest