#pragma once

#include "mutex"
#include "unordered_map"
#include "atomic"
#include "memory"
#include "stdexcept"
#include <cstddef>

namespace Contest {

// Base class for the shared state of an operator.
class SharedState
{
private:
    friend class SharedStateManager;
    size_t operator_id_{};
public:
    inline virtual ~SharedState() = default;
    [[nodiscard]] inline size_t get_operator_id() const { return operator_id_; }
};

// Manages all shared states in the execution plan. Each thread uses it to get the SharedState for the current operator.
class SharedStateManager {
    std::mutex m_;
    std::unordered_map<size_t, std::unique_ptr<SharedState>> state_map_;
public:
    // Gets the SharedState object with ID i. If it does not exist, it is created using the provided constructor arguments.
    // Note: If the object already exists, the extra arguments passed will be ignored.
    template <typename T, typename... Args>
    T& get(size_t i, Args&&... args) {
        std::lock_guard<std::mutex> lock(m_);
        auto it = state_map_.find(i);
        if (it == state_map_.end()) {
            // Use variadic template argument forwarding to construct an instance of type T.
            it = state_map_.emplace(i, std::make_unique<T>(std::forward<Args>(args)...)).first;
            it->second->operator_id_ = i;
        }
        T* res = dynamic_cast<T*>(it->second.get());
        if (!res) {
            throw std::runtime_error("Failed to retrieve shared state. Wrong type found.");
        }
        return *res;
    }

};
}