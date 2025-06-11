#pragma once

#include <condition_variable>
#include <thread>
#include <queue>

#include <pthread.h>

#include "pipeline.hpp"

namespace Contest::MorselLite {

class Scheduler {
  struct TaskNode {
    Pipeline                            *pipeline{nullptr};
    std::vector<Pipeline::ExecutorArgs>  task_args;
    std::atomic<ssize_t>                 undispatched{0};
    std::atomic<ssize_t>                 finished{0};
    TaskNode                            *next{nullptr};

    TaskNode() = default;

    TaskNode(const TaskNode &other)
      : pipeline(other.pipeline), task_args(other.task_args),
        undispatched(other.undispatched.load()),
        finished(other.finished.load()),
        next(other.next) { }
    
    TaskNode(TaskNode &&other)
      : pipeline(std::exchange(other.pipeline, nullptr)), task_args(std::move(other.task_args)),
        undispatched(other.undispatched.load()),
        finished(other.finished.load()),
        next(std::exchange(other.next, nullptr)) { }
  };
public:
  Scheduler(ExecutionContext &execution_ctx, size_t num_threads)
    : ctx_(std::addressof(execution_ctx)) {
    init_worker_threads(num_threads);
  }

  ~Scheduler() {
    {
      std::unique_lock<std::mutex> lk(task_queue_mtx_);
      task_queue_stop_ = true;
    }
    task_queue_cv_.notify_all();
    for (auto &t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  void set_plan(const Plan &plan, size_t partition_size) {
    reset();

    auto &root_pipeline = *pipelines_.emplace_back(std::make_unique<ScanProbeTabularPipeline>(0));
    do_decompose_pipeline(plan, plan.nodes[plan.root], root_pipeline);

    // complete root pipeline
    {
      const auto &root_node_output_attrs = plan.nodes[plan.root].output_attrs;

      TabularDesc tabular_desc;
      tabular_desc.output_columns.resize(root_node_output_attrs.size());
      output_data_types_.resize(root_node_output_attrs.size());

      for (size_t i = 0; i < root_node_output_attrs.size(); ++i) {
        tabular_desc.output_columns[i].index = std::get<0>(root_node_output_attrs[i]);
        tabular_desc.output_columns[i].data_type = std::get<1>(root_node_output_attrs[i]);
        output_data_types_[i] = tabular_desc.output_columns[i].data_type;
      }
    }

    task_node_buf_.resize(pipelines_.size());
    for (size_t i = 0; i < pipelines_.size(); ++i) {
      task_node_buf_[i].pipeline = pipelines_[i].get();
      task_node_buf_[i].task_args = pipelines_[i]->create_partition_args(partition_size, *ctx_);
      task_node_buf_[i].undispatched.store(task_node_buf_[i].task_args.size(), std::memory_order_relaxed);
    }
  }

  void reset() {
    ctx_->reset();
    pipelines_.clear();
    task_node_buf_.clear();
    hash_table_alloc_id_ = 0;
    output_data_types_.clear();
    serial_task_queue_ = std::queue<TaskNode *>();

    std::unique_lock<std::mutex> lk(task_queue_mtx_);

    task_queue_ = std::queue<TaskNode *>();
    num_executed_.store(0);
  }

  void serial_run() {
    for (size_t i = 0; i < pipelines_.size(); ++i) {
      Pipeline *pipeline = pipelines_[i].get();
      if (pipeline->num_dependicies() == 0) {
        serial_task_queue_.push(std::addressof(task_node_buf_[i]));
      }
    }
  
    while (!serial_task_queue_.empty()) {
      TaskNode *task_node = serial_task_queue_.front();
      for (size_t i = 0; i < task_node->undispatched; ++i) {
        task_node->pipeline->execute(0, *ctx_, task_node->task_args[i]);
      }
      serial_task_queue_.pop();
    
      auto next_pipelines = task_node->pipeline->get_next_pipelines();
      for (size_t i = 0; i < next_pipelines.size(); ++i) {
        if (next_pipelines[i]->descrease_dependency() == 1) {
          serial_task_queue_.push(std::addressof(task_node_buf_[next_pipelines[i]->id()]));
        }
      }
    }
  }

  void run() {
    {
      std::unique_lock<std::mutex> lk(task_queue_mtx_);
      for (size_t i = 0; i < pipelines_.size(); ++i) {
        Pipeline *pipeline = pipelines_[i].get();
        if (pipeline->num_dependicies() == 0) {
          task_queue_.push(std::addressof(task_node_buf_[i]));
        }
      }
    }
    task_queue_cv_.notify_all();

    std::unique_lock<std::mutex> wait_lk(work_wait_mtx_);
    work_wait_cv_.wait(wait_lk, [&]() { return num_executed_.load() == pipelines_.size(); } );
  }

  [[nodiscard]] Span<const DataType> output_data_types() const noexcept { return output_data_types_; }
  
private:
  ExecutionContext                      *ctx_;
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
  size_t                                 hash_table_alloc_id_{0};
  std::vector<DataType>                  output_data_types_;

  std::vector<TaskNode>                  task_node_buf_;
  std::queue<TaskNode *>                 task_queue_;
  std::mutex                             task_queue_mtx_;
  std::condition_variable                task_queue_cv_;
  bool                                   task_queue_stop_{false};
  std::atomic<size_t>                    num_executed_;
  
  std::queue<TaskNode *>                 serial_task_queue_;
  std::vector<std::thread>               threads_;

  std::mutex                             work_wait_mtx_;
  std::condition_variable                work_wait_cv_;

  void init_worker_threads(size_t num_threads) {
    threads_.reserve(num_threads);

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
      threads_.emplace_back([&, thread_id]() {
        while (true) {
          TaskNode *task_node = nullptr;
          ssize_t old_undispatched = 0;
          {
            std::unique_lock<std::mutex> lk(task_queue_mtx_);
            task_queue_cv_.wait(lk, [&]() { return !task_queue_.empty() || task_queue_stop_; });
            if (task_queue_stop_) {
              return;
            }
            task_node = task_queue_.front();

            old_undispatched = task_node->undispatched.fetch_sub(1, std::memory_order_relaxed);
            if (old_undispatched == 1) {
              task_queue_.pop();
            } else if (old_undispatched <= 0) {
              continue;
            }
          }

          const size_t num_total_tasks = task_node->task_args.size();

          task_node->pipeline->execute(thread_id, *ctx_, task_node->task_args[num_total_tasks - old_undispatched]);
          
          {
            size_t old_finished = task_node->finished.fetch_add(1, std::memory_order_relaxed);
            if (old_finished == num_total_tasks - 1) {

              auto next_pipelines = task_node->pipeline->get_next_pipelines();

              for (size_t i = 0; i < next_pipelines.size(); ++i) {
                if (next_pipelines[i]->descrease_dependency() == 1) {
                  TaskNode *next_task_node = std::addressof(task_node_buf_[next_pipelines[i]->id()]);
                  {
                    std::unique_lock<std::mutex> lk(task_queue_mtx_);
                    task_queue_.push(next_task_node);
                  }
                  task_queue_cv_.notify_all();   
                }
              }
              if (num_executed_.fetch_add(1) == pipelines_.size() - 1) {
                work_wait_cv_.notify_one();
              }
            }
          }
        }
      });
    }
  }

  void do_decompose_pipeline(const Plan &plan, const PlanNode &node, Pipeline &pipeline) {
    if (node.data.index() == 0) { // scan
      const auto &scan_node = std::get<0>(node.data);
      TableScanDesc desc;
      desc.table = std::addressof(plan.inputs[scan_node.base_table_id]);
      desc.output_columns.resize(node.output_attrs.size());
      for (size_t i = 0; i < desc.output_columns.size(); ++i) {
        desc.output_columns[i].index = std::get<0>(node.output_attrs[i]);
        desc.output_columns[i].data_type = std::get<1>(node.output_attrs[i]);

        const auto &col = desc.table->columns[desc.output_columns[i].index];
        
        if (desc.output_columns[i].data_type == DataType::VARCHAR) {
          ctx_->init_varchar_column_page_locator(std::addressof(col));
        } else {
          ctx_->init_fixed_column_page_locator(std::addressof(col));
        }
      }
      pipeline.set_scan_desc(std::move(desc));
      std::reverse(pipeline.hash_probe_descs_.begin(), pipeline.hash_probe_descs_.end());
    } else { // join
      const auto &join_node = std::get<1>(node.data);
      const auto &left_node = plan.nodes[join_node.left];
      const auto &right_node = plan.nodes[join_node.right];

      HashBuildDesc build_desc;
      HashProbeDesc probe_desc;

      build_desc.hash_table_id = hash_table_alloc_id_++;
      probe_desc.hash_table_id = build_desc.hash_table_id;
      ctx_->init_hash_table<ChainedHashMap>(build_desc.hash_table_id);

      if (join_node.build_left) {
        build_desc.key_column_desc.index = join_node.left_attr;
        build_desc.key_column_desc.data_type = DataType::INT32;
        
        probe_desc.key_column_desc.index = join_node.right_attr;
        probe_desc.key_column_desc.data_type = DataType::INT32;

        for (size_t i = 0; i < node.output_attrs.size(); ++i) {
          auto [ref_index, data_type] = node.output_attrs[i];
          if (ref_index < left_node.output_attrs.size()) {
            probe_desc.joined_output_column_descs.push_back(ProjectColumnDesc{ref_index, i,  data_type});
          } else {
            probe_desc.probed_output_column_descs.push_back(
              ProjectColumnDesc{ref_index - left_node.output_attrs.size(), i, data_type});
          }
        }
      } else {
        build_desc.key_column_desc.index = join_node.right_attr;
        build_desc.key_column_desc.data_type = DataType::INT32;

        probe_desc.key_column_desc.index = join_node.left_attr;
        probe_desc.key_column_desc.data_type = DataType::INT32;

        for (size_t i = 0; i < node.output_attrs.size(); ++i) {
          auto [ref_index, data_type] = node.output_attrs[i];
          if (ref_index < left_node.output_attrs.size()) {
            probe_desc.probed_output_column_descs.push_back(ProjectColumnDesc{ref_index, i,  data_type});
          } else {
            probe_desc.joined_output_column_descs.push_back(
              ProjectColumnDesc{ref_index - left_node.output_attrs.size(), i, data_type});
          }
        }
      }

      pipeline.add_hash_probe_desc(std::move(probe_desc));

      size_t new_pipeline_id = pipelines_.size();
      auto &new_pipeline = *pipelines_.emplace_back(std::make_unique<ScanProbeBuildPipeline>(new_pipeline_id));
      auto &spb_pipeline = static_cast<ScanProbeBuildPipeline &>(new_pipeline);
      spb_pipeline.set_hash_build_desc(std::move(build_desc));

      pipeline.increase_dependency();
      new_pipeline.add_next_pipeline(std::addressof(pipeline));

      if (join_node.build_left) {
        do_decompose_pipeline(plan, left_node, spb_pipeline);
        do_decompose_pipeline(plan, right_node, pipeline);
      } else {
        do_decompose_pipeline(plan, left_node, pipeline);
        do_decompose_pipeline(plan, right_node, spb_pipeline);
      }
    }
  }

};

}