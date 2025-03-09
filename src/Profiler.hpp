#pragma once



#include "fmt/base.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class Profiler {
  struct Event {
    const char *name = nullptr;
    int thread_id = INT32_MIN;
    int64_t begin_time = INT64_MIN; // us
    int64_t end_time = INT64_MIN; // us
    int64_t begin_cputime = INT64_MIN; // us
    int64_t end_cputime = INT64_MIN; // us
  };

public:
  Profiler(int thread_num) {
    thread_num_ = thread_num;
    events_.resize(thread_num);
  }

  void set_thread_id(int thread_id) {
    if (thread_id >= thread_num_) {
      throw std::runtime_error("thread_id is incorrect");
    }
    this_thread_id = thread_id;
  }

  void event_start(const std::string &event_name) {
    int eveid = 0;
    if ((eveid = find_event_index(event_name)) == INT32_MIN) {
      eveid = append_new_event_type(event_name);
    }
    std::shared_lock lock(rw_mtx_);
    events_[get_current_id()][eveid].begin_time = get_current_time();
    events_[get_current_id()][eveid].begin_cputime = get_current_cpu_time();
  }

  void event_end(const std::string &event_name) {
    int eveid = 0;
    if ((eveid = find_event_index(event_name)) == INT32_MIN) {
      throw std::runtime_error("event not esxists");
    }
    std::shared_lock lock(rw_mtx_);
    events_[get_current_id()][eveid].end_time = get_current_time();
    events_[get_current_id()][eveid].end_cputime = get_current_cpu_time();
  }

  void print_profiles() {
    fmt::print("| thread id\t |");
    for (int i = 0; i < event_names_.size(); i ++) {
      fmt::print(" {}\t |", event_names_[i]);
    }
    fmt::println("");
    for (int i = 0 ; i < thread_num_ ; i ++) {
      fmt::print("| {}\t |", i);
      for (Event &event : events_[i]) {
        int64_t time_use = 0;
        if (event.begin_time == INT64_MIN || event.end_time == INT64_MIN) {
          fmt::print(" N/A\t |");
          continue;
        }
        time_use = event.end_time - event.begin_time;
        if (time_use < 10 * 1000) {
          fmt::print(" {}us\t |", time_use);
        } else if (time_use < 10 * 1000 * 1000) {
          fmt::print(" {}ms\t |", time_use / 1000);
        } else {
          fmt::print(" {}s\t |", time_use / 1000 / 1000);
        }
      }
      fmt::println("");
    }
  }

private:
  int find_event_index(const std::string &event_name) {
    for (int i = 0; i < event_names_.size() ; i ++) {
      const std::string& name = event_names_[i];
      if (event_name == name) {
        // 双重检查
        return i;
      }
    }
    return INT32_MIN;
  }

  int append_new_event_type(const std::string &event_name) {
    std::lock_guard lock(rw_mtx_);
    int eventid = 0;
    if ((eventid = find_event_index(event_name)) != INT32_MIN) {
      return eventid;
    }
    
    event_names_.push_back(event_name);
    for (int i = 0; i < events_.size(); i ++) {
      auto &events = events_[i];
      events.emplace_back();
      events.back().name = event_name.c_str();
      events.back().thread_id = i;
    }

    return event_names_.size() - 1;
  }

  int64_t get_current_cpu_time() {
    return std::clock() / (CLOCKS_PER_SEC / 1000 / 1000);
  }

  int64_t get_current_time() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
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
  std::vector<std::vector<Event>> events_;
  std::shared_mutex rw_mtx_;
  int thread_num_;
};

thread_local int Profiler::this_thread_id = INT32_MIN;
