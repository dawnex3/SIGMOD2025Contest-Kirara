#pragma once

#include "mutex"
#include "unordered_map"
#include "atomic"
#include "memory"
#include "stdexcept"
#include <cstddef>

namespace Contest {

// 操作符的共享状态的父类
class SharedState
{
private:
    friend class SharedStateManager;
    size_t operator_id_;
public:
    inline virtual ~SharedState(){};
    inline size_t get_operator_id() const { return operator_id_; }
};

// 管理执行计划中的所有共享状态。每个线程用它来获取当前操作符的SharedState
class SharedStateManager {
    std::mutex m_;
    std::unordered_map<size_t, std::unique_ptr<SharedState>> state_map_;
public:
    // 获取编号为i的SharedState对象，如果不存在则使用传入的构造参数创建它。
    // 注意：如果对象已存在，则传入的额外参数会被忽略
    template <typename T, typename... Args>
    T& get(size_t i, Args&&... args) {
        std::lock_guard<std::mutex> lock(m_);
        auto it = state_map_.find(i);
        if (it == state_map_.end()) {
            // 利用可变参数转发来构造T类型的实例
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