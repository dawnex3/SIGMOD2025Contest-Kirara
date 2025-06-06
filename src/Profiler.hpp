/**
 * @file Profiler.hpp
 * @brief Provides classes for profiling events and operator statistics in a
 * multi-threaded environment.
 *
 * Usage:
 * - Instantiate a `ProfileGuard` when entering a function or an event to
 *   automatically start timing the event.
 * - Use `ProfileGuard.add_input_row_count()` and `ProfileGuard.add_output_row_count()`
 *   to track operator statistics.
 *
 * Macros:
 * - All profiling methods are enabled only if `PROFILER` macro is defined.
 *
 * Note:
 * - The Profiler significantly impacts performance. So it should only be
 *   used in debug builds or when profiling is needed.
 *
 * Thread Safety:
 * - The `Profiler` class is designed to be thread-safe.
 *
 * Code Example:
 * @code
 * function my_function() {
 *     ProfileGuard guard(global_profiler, "my_function");
 *     guard.event_begin("my_function");
 *     // ... function logic ...
 *     guard.event_end("my_function");
 * }
 * @endcode
 * 
 * Output Example:
 * --------------------------------
 * | thid | event1 | event2 | ... |
 * | 0    | 100ms  | 200ms  | ... |
 * | 1    | 150ms  | 250ms  | ... |
 * | ...  | ...    | ...    | ... |
 * --------------------------------
 */
 
#pragma once

#include "fmt/base.h"
#include "fmt/format.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


class Profiler {
  enum class EventType {
    OPERATOR,
    EVENT,
    INVALID,
  };

  struct Event {
    EventType type = EventType::INVALID;
    const char *name = nullptr;
    int thread_id = INT32_MIN;
    int64_t begin_time = INT64_MIN;    // us
    int64_t total_time = INT64_MIN;    // us
    int64_t begin_cputime = INT64_MIN; // us
    int64_t total_cputime = INT64_MIN; // us
    int64_t invoke_time = 0;
    int input_row_count = 0;
    int output_row_count = 0;
  };

public:
  Profiler(int thread_num) {
    thread_num_ = thread_num;
    events_.resize(thread_num);
    for (auto &ev : events_) {
      ev.reserve(128);
      event_names_.reserve(128);
    }
  }

  void set_thread_id(int thread_id) {
    if (thread_id >= thread_num_) {
      throw std::runtime_error("thread_id is incorrect");
    }
    this_thread_id = thread_id;
  }

  void event_begin(const std::string &event_name) {
#ifdef PROFILER
    int eveid = 0;
    std::shared_lock lock(rw_mtx_);
    if ((eveid = find_event_index(event_name)) == INT32_MIN) {
      lock.unlock();
      eveid = append_new_event(event_name, EventType::EVENT);
      lock.lock();
    }
    events_[get_current_id()][eveid].begin_time = get_current_time();
    events_[get_current_id()][eveid].begin_cputime = get_current_cpu_time();
#endif
  }

  void event_end(const std::string &event_name) {
#ifdef PROFILER
    int eveid = 0;
    std::shared_lock lock(rw_mtx_);
    if ((eveid = find_event_index(event_name)) == INT32_MIN) {
      throw std::runtime_error("event not esxists");
    }

    Event &current = events_[get_current_id()][eveid];
    if (current.begin_time == INT64_MIN || current.begin_cputime == INT64_MIN) {
      throw std::runtime_error("event not start");
    }
    if (current.type != EventType::EVENT) {
      throw std::runtime_error("event type is incorrect");
    }
    if (current.total_time == INT64_MIN) {
      current.total_time = 0;
    }
    if (current.total_cputime == INT64_MIN) {
      current.total_cputime = 0;
    }
    current.total_time += get_current_time() - current.begin_time;
    current.total_cputime += get_current_cpu_time() - current.begin_cputime;
    current.begin_time = INT64_MIN;
    current.begin_cputime = INT64_MIN;
#endif
  }

  void event_pause(const std::string &event_name) { event_end(event_name); }

  void event_resume(const std::string &event_name) { event_begin(event_name); }

  void add_input_row_count(const std::string &operator_name, int num) {
#ifdef PROFILER
    int eveid = 0;
    std::shared_lock lock(rw_mtx_);
    std::string op_name = operator_name + "(in/out)";
    if ((eveid = find_event_index(op_name)) == INT32_MIN) {
      lock.unlock();
      eveid = append_new_event(op_name, EventType::OPERATOR);
      lock.lock();
    }

    Event &current = events_[get_current_id()][eveid];
    if (current.type != EventType::OPERATOR) {
      throw std::runtime_error("event type is incorrect");
    }
    events_[get_current_id()][eveid].input_row_count += num;
#endif
  }

  void add_output_row_count(const std::string &operator_name, int num) {
#ifdef PROFILER
    int eveid = 0;
    std::shared_lock lock(rw_mtx_);
    std::string op_name = operator_name + "(in/out)";
    if ((eveid = find_event_index(op_name)) == INT32_MIN) {
      lock.unlock();
      eveid = append_new_event(op_name, EventType::OPERATOR);
      lock.lock();
    }

    Event &current = events_[get_current_id()][eveid];
    if (current.type != EventType::OPERATOR) {
      throw std::runtime_error("event type is incorrect");
    }
    events_[get_current_id()][eveid].output_row_count += num;
#endif
  }

  void print_profiles() {
#ifdef PROFILER
    std::unique_lock lock(rw_mtx_);
    fmt::print("| thid |");
    std::vector<int> algin;
    std::vector<std::pair<const char *, int>> sorted_event;

    for (int i = 0; i < event_names_.size(); i++) {
      sorted_event.push_back({event_names_[i].c_str(), i});
    }

    std::sort(sorted_event.begin(), sorted_event.end(),
              [](const std::pair<const char *, int> &a,
                 const std::pair<const char *, int> &b) {
                return strcmp(a.first, b.first) < 0;
              });

    for (int i = 0; i < sorted_event.size(); i++) {
      algin.push_back(std::max(strlen(sorted_event[i].first), (size_t)6));
      fmt::print(" {:<{}} |", sorted_event[i].first, algin[i]);
    }
    fmt::println("");
    for (int i = 0; i < thread_num_; i++) {
      fmt::print("| {:4} |", i);
      for (int j = 0; j < sorted_event.size(); j++) {
        Event &event = events_[i][sorted_event[j].second];
        if (event.type == EventType::EVENT) {
          fmt::print(" {:<{}} |", even_time_to_string(event), algin[j]);
        } else if (event.type == EventType::OPERATOR) {
          fmt::print(" {:<{}} |",
                     std::to_string(event.input_row_count) + "/" +
                         std::to_string(event.output_row_count),
                     algin[j]);
        }
      }
      fmt::println("");
    }
#endif
  }

private:
  std::string even_time_to_string(const Event &event) {
    if (event.total_time == INT64_MIN || event.total_cputime == INT64_MIN) {
      return "N/A";
    }
    int64_t time_use = event.total_time;
    int64_t time_use_cpu = event.total_cputime;
    if (time_use < 10 * 1000) {
      return fmt::format("{}us", time_use);
    } else if (time_use < 10 * 1000 * 1000) {
      return fmt::format("{}ms", time_use / 1000);
    } else {
      return fmt::format("{}s", time_use / 1000 / 1000);
    }
  }

  int find_event_index(const std::string &event_name) {
    auto it = event_indices_.find(event_name);
    if (it != event_indices_.end()) {
      return it->second;
    }
    return INT32_MIN;
  }

  int append_new_event(const std::string &event_name, EventType type) {
    std::lock_guard lock(rw_mtx_);
    int eventid = 0;
    if ((eventid = find_event_index(event_name)) != INT32_MIN) {
      return eventid;
    }

    event_names_.push_back(event_name);
    event_indices_.insert({event_name, event_indices_.size()});
    for (int i = 0; i < events_.size(); i++) {
      auto &events = events_[i];
      events.emplace_back();
      Event &current = events.back();
      current.name = event_names_.back().c_str();
      current.thread_id = i;
      current.type = type;
    }

    return event_indices_.size() - 1;
  }

  int64_t get_current_cpu_time() {
    return std::clock();
  }

  int64_t get_current_time() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
  }

  int get_current_id() {
    if (this_thread_id == INT32_MIN) {
      throw std::runtime_error("current thread does not set an id");
    } else {
      return this_thread_id;
    }
  }

private:
  thread_local static int this_thread_id;
  std::vector<std::string> event_names_;
  std::unordered_map<std::string, int> event_indices_;
  std::vector<std::vector<Event>> events_;
  std::shared_mutex rw_mtx_;
  int thread_num_;
};

class ProfileGuard {
public:
  ProfileGuard(Profiler *profiler, const std::string &event_name)
      : profiler_(profiler), event_name_(event_name) {
    profiler_->event_begin(event_name_);
  }
  ~ProfileGuard() {
    profiler_->event_end(event_name_);
  }
  void pause() {
    profiler_->event_pause(event_name_);
  }
  void resume() {
    profiler_->event_resume(event_name_);
  }
  void add_input_row_count(int num) {
    profiler_->add_input_row_count(event_name_, num);
  }
  void add_output_row_count(int num) {
    profiler_->add_output_row_count(event_name_, num);
  }

private:
  Profiler *profiler_;
  std::string event_name_;
};

thread_local int Profiler::this_thread_id = INT32_MIN;
Profiler *global_profiler = nullptr;