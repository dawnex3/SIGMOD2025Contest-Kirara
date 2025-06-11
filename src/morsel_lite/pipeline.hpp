#pragma once

#include <bitset>

#include "column_partition.hpp"
#include "data_chunck.hpp"
#include "execution_context.hpp"

namespace Contest::MorselLite {

struct TableScanDesc {
  const ColumnarTable         *table{nullptr};
  std::vector<ColumnAttrDesc>  output_columns;
};

struct HashProbeDesc {
  ColumnAttrDesc                 key_column_desc;
  size_t                         hash_table_id{INVALID_INDEX};
  std::vector<ProjectColumnDesc> probed_output_column_descs; // probed side
  std::vector<ProjectColumnDesc> joined_output_column_descs; // hash table side
};

struct HashBuildDesc {
  ColumnAttrDesc key_column_desc;
  size_t         hash_table_id{INVALID_INDEX};
};

struct TabularDesc {
  std::vector<ColumnAttrDesc> output_columns;
};

class Pipeline {
  friend class Scheduler;
public:
  struct ExecutorArgs {
    std::vector<ColumnPartitionDesc> partition_descs;
    size_t                           num_rows;
  };

public:
  virtual ~Pipeline() = default;

  virtual void execute(uint32_t thread_id, ExecutionContext &ctx, const ExecutorArgs &args) const = 0;

  [[nodiscard]] size_t id() const noexcept { return id_; }

  size_t increase_dependency() { return num_dependicies_.fetch_add(1); }
  size_t descrease_dependency() { return num_dependicies_.fetch_sub(1); }

  [[nodiscard]] size_t num_dependicies() const noexcept { return num_dependicies_.load(); }

  [[nodiscard]] std::vector<ExecutorArgs> 
  create_partition_args(size_t partition_size, const ExecutionContext &ctx) {
    const ColumnarTable *table = scan_desc_.table;

    const size_t num_full_partitions = table->num_rows / partition_size;
    const size_t size_partial_partition = table->num_rows % partition_size;

    std::bitset<64> col_masks;
    for (size_t i = 0; i < scan_desc_.output_columns.size(); ++i) {
      col_masks[scan_desc_.output_columns[i].index] = true;
    }

    auto descs = ColumnPartitionDesc::from_table(*table, partition_size, 
      ctx.get_fixed_column_page_locators(), ctx.get_varchar_column_page_locators(), col_masks);

    std::vector<ExecutorArgs> args(num_full_partitions + static_cast<size_t>(size_partial_partition != 0));
    for (size_t i = 0; i < num_full_partitions; ++i) {
      args[i].partition_descs = std::move(descs[i]);
      args[i].num_rows = partition_size;
    }

    if (size_partial_partition != 0) {
      args.back().partition_descs = std::move(descs.back());
      args.back().num_rows = size_partial_partition;
    }

    return args;
  }

  void set_next_pipelines(std::vector<Pipeline *> pipelines) { next_pipelines_ = std::move(pipelines); }
  void add_next_pipeline(Pipeline *pipeline) { next_pipelines_.push_back(pipeline); }
  [[nodiscard]] Span<Pipeline *> get_next_pipelines() noexcept { return next_pipelines_; }

  void set_scan_desc(TableScanDesc desc) { scan_desc_ = std::move(desc); }
  [[nodiscard]] const TableScanDesc &get_scan_desc() const noexcept { return scan_desc_; }
  [[nodiscard]] TableScanDesc &get_scan_desc() noexcept { return scan_desc_; }

  void set_hash_probe_descs(std::vector<HashProbeDesc> descs) { hash_probe_descs_ = std::move(descs); }
  void add_hash_probe_desc(HashProbeDesc desc) { hash_probe_descs_.push_back(std::move(desc)); }
  [[nodiscard]] Span<const HashProbeDesc> get_hash_probe_descs() const { return hash_probe_descs_; }
  [[nodiscard]] Span<HashProbeDesc> get_hash_probe_descs() { return hash_probe_descs_; }

protected:
  void execute_scan(uint32_t thread_id, ExecutionContext &ctx, const TableScanDesc &scan_desc, 
                    const ExecutorArgs &args, DataChunck &data_chunck) const {
    const ColumnarTable *table = scan_desc.table;
    std::vector<ColumnVector> cols(scan_desc.output_columns.size());
    for (size_t i = 0; i < scan_desc.output_columns.size(); ++i) {
      const auto &partition_desc = args.partition_descs[scan_desc.output_columns[i].index];
      const auto &col = table->columns[scan_desc.output_columns[i].index];
      
      if (scan_desc.output_columns[i].data_type == DataType::VARCHAR) {
        cols[i] = RowIdColumnVector(col);
        cols[i].as_rowid_vector().resize(args.num_rows);

        auto col_view = cols[i].as_rowid_vector().view();
        partition_desc.materialize<std::string>(col, col_view.begin()); 
      } else {
        cols[i] = FlatColumnVector(scan_desc.output_columns[i].data_type);
        cols[i].as_flat_vector().resize(args.num_rows);

        auto impl = [&](auto _ty) {
          using T = ML_TYPE_OF_TAG(_ty);
          auto view = cols[i].as_flat_vector().view_as<T>();
          partition_desc.materialize<T>(col, view.begin());
        };

        switch (scan_desc.output_columns[i].data_type) {
        default              : unreachable_branch();
        case DataType::INT32 : impl(type_tag<int32_t>); break;
        case DataType::INT64 : impl(type_tag<int64_t>); break;
        case DataType::FP64  : impl(type_tag<double>); break;
        }
      }
    }
    data_chunck = DataChunck{std::move(cols)};
  }
  
  void execute_hash_probe(uint32_t thread_id, ExecutionContext &ctx, 
                          const HashProbeDesc &hash_probe_desc, DataChunck &data_chunck) const {
    const auto &key_column = data_chunck.column(hash_probe_desc.key_column_desc.index);
    const auto &ht = ctx.get_hash_table(hash_probe_desc.hash_table_id);

    ht.lookup(thread_id, ctx.get_data_chuncks(), key_column.as_flat_vector(), data_chunck, 
              hash_probe_desc.probed_output_column_descs, hash_probe_desc.joined_output_column_descs);
  }

protected:
  size_t                     id_{INVALID_INDEX};
  std::vector<Pipeline *>    next_pipelines_;
  TableScanDesc              scan_desc_;
  std::vector<HashProbeDesc> hash_probe_descs_;
  std::atomic<size_t>        num_dependicies_{0};

  Pipeline(size_t id)
    : id_(id) { }

  Pipeline(size_t id, TableScanDesc scan_desc, std::vector<HashProbeDesc> hash_probe_descs)
    : id_(id), scan_desc_(std::move(scan_desc)), hash_probe_descs_(std::move(hash_probe_descs)) { }

  Pipeline(size_t id, TableScanDesc scan_desc)
    : id_(id), scan_desc_(std::move(scan_desc)) { }
};

class ScanProbeTabularPipeline : public Pipeline {
public:
  ScanProbeTabularPipeline(size_t id)
    : Pipeline(id) { }

  ~ScanProbeTabularPipeline() override = default;

  ScanProbeTabularPipeline(size_t id, TableScanDesc scan_desc, std::vector<HashProbeDesc> probe_descs, TabularDesc tabular_desc)
    : Pipeline(id, std::move(scan_desc), std::move(probe_descs)), tabular_desc_(std::move(tabular_desc)) { }

  ScanProbeTabularPipeline(size_t id, TableScanDesc scan_desc, TabularDesc tabular_desc)
    : Pipeline(id, std::move(scan_desc)), tabular_desc_(std::move(tabular_desc)) { }

  void execute(uint32_t thread_id, ExecutionContext &ctx, const ExecutorArgs &args) const override {
    DataChunck data_chunck;

    execute_scan(thread_id, ctx, scan_desc_, args, data_chunck);

    for (size_t i = 0; i < hash_probe_descs_.size(); ++i) {
      if (data_chunck.num_rows() == 0) {
        return;
      }
      execute_hash_probe(thread_id, ctx, hash_probe_descs_[i], data_chunck);
    }

    if (data_chunck.num_rows() != 0) {
      ctx.add_result_partition(std::move(data_chunck));
    }
  }

  void set_tabular_desc(TabularDesc desc) { tabular_desc_ = std::move(desc); }
  const TabularDesc &get_tabular_desc() const noexcept { return tabular_desc_; }

private:
  TabularDesc tabular_desc_;
};

class ScanProbeBuildPipeline : public Pipeline {
public:
  ScanProbeBuildPipeline(size_t id)
    : Pipeline(id) { }

  ~ScanProbeBuildPipeline() = default;

  ScanProbeBuildPipeline(size_t id, TableScanDesc scan_desc, std::vector<HashProbeDesc> probe_descs, HashBuildDesc hash_build_desc)
    : Pipeline(id, std::move(scan_desc), std::move(probe_descs)), hash_build_desc_(std::move(hash_build_desc)) { }

  ScanProbeBuildPipeline(size_t id, TableScanDesc scan_desc, HashBuildDesc hash_build_desc)
    : Pipeline(id, std::move(scan_desc)), hash_build_desc_(std::move(hash_build_desc)) { }

  void execute(uint32_t thread_id, ExecutionContext &ctx, const ExecutorArgs &args) const override {
    DataChunck init_data_chunck;
    execute_scan(thread_id, ctx, scan_desc_, args, init_data_chunck);
    
    for (size_t i = 0; i < hash_probe_descs_.size(); ++i) {
      if (init_data_chunck.num_rows() == 0) {
        return;
      }
      execute_hash_probe(thread_id, ctx, hash_probe_descs_[i], init_data_chunck);
    }

    if (init_data_chunck.num_rows() == 0) {
      return;
    }

    auto &ht = ctx.get_hash_table(hash_build_desc_.hash_table_id);

    size_t data_chunck_id = ctx.manage_data_chunck(std::move(init_data_chunck));
    const auto &data_chunck = ctx.get_data_chunck(data_chunck_id);
    const auto &key_column = data_chunck.column(hash_build_desc_.key_column_desc.index);
    ML_ASSERT(key_column.is_flat_vector() && key_column.data_type() == DataType::INT32);
    ht.insert(thread_id, key_column.as_flat_vector(), data_chunck_id);
  }

  void set_hash_build_desc(HashBuildDesc desc) { hash_build_desc_ = std::move(desc); }
  [[nodiscard]] const HashBuildDesc &get_hash_build_desc() const noexcept { return hash_build_desc_; }

private:
  HashBuildDesc hash_build_desc_;
};

}