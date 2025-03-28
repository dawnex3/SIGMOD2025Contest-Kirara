#include <list>
#include "plan.h"
#include "HashMap.hpp"
#include "DataStructure.hpp"

//#define CACHE_LOG
#define NO_CACHE

namespace Contest {

// Boost库的hash_combine实现
inline std::size_t hash_combine(std::size_t seed, std::size_t hash_value) {
    // 0x9e3779b9是常用的魔数（来自黄金分割比例），用于带来良好的混合效果
    return seed ^ (hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief 计算范围 [first, last) 内元素的组合哈希值
 * @tparam InputIt 输入迭代器类型
 * @param first 范围起始迭代器
 * @param last 范围结束迭代器
 * @param seed 初始哈希种子 (默认为0)
 * @return 整个范围的组合哈希值
 *
 * @details
 * - 遍历范围时，每个元素的哈希值通过 hash_combine 合并到累积哈希中
 * - 使用 std::hash 计算单个元素的哈希值
 * - 空范围直接返回初始 seed
 * - 支持连续或非连续容器 (只要迭代器有效)
 */
template <typename InputIt>
std::size_t hash_range(InputIt first, InputIt last, std::size_t seed = 0) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    std::hash<value_type> hasher;  // 元素哈希计算器

    for (; first != last; ++first) {
        const auto& element = *first;
        const std::size_t element_hash = hasher(element);
        seed = hash_combine(seed, element_hash);
    }

    return seed;
}


// 一个执行过的计划。包括复制下来的plan的nodes，以及plan的input表的一些采样数据。
// 保存了执行中的哈希表。
class QueryCache{
public:
    // 对一个column的采样
    class ColumnSample{
    public:
        DataType                type_;
        size_t                  page_num_;
        std::vector<uint64_t>   samples_;   // 对列的采样
        uint64_t                hash_;      // 哈希值

        inline static const size_t SAMPLE_SIZE = 20;    // 采样数目（最大）。实际采样数目是2倍SAMPLE_SIZE

        // 采样一个列，生成哈希值。如果这个列是VARCHAR类型，不用采样了，哈希值设为无效。
        ColumnSample(const Column& column, size_t row_num) : type_(column.type), page_num_(column.pages.size()){
            if(type_!=DataType::INT32){
                hash_ = INVALID_HASH;
            }
            // 等间距的采样出SAMPLE_SIZE个数据。如果page_num_>=SAMPLE_SIZE，被采样的每个page都不重复，此时采样pages的开头位置
            if(page_num_ >= SAMPLE_SIZE){
                double page_step = static_cast<double>(page_num_ - 1) / (SAMPLE_SIZE - 1);
                for(size_t i=0; i<SAMPLE_SIZE; i++){
                    size_t page_index = static_cast<size_t>(std::round(i * page_step));
                    const Page* page = column.pages[page_index];
                    uint16_t non_null_count = getNonNullCount(page);
                    if(non_null_count==0){
                        samples_.push_back(*(uint32_t*)(page->data));
                    } else if(non_null_count==1){
                        samples_.push_back(*(uint64_t*)(page->data));
                    } else {
                        samples_.push_back(*(uint64_t*)(page->data+4));
                    }
                }
            } else {
                // 如果page_num_<SAMPLE_SIZE，那么有些page中需要采样多个数据。确定要采样的行号，对这些行做采样。
                size_t sample_size = std::min(row_num, SAMPLE_SIZE);
                if(sample_size==1){
                    samples_.push_back(*(uint64_t*)(column.pages[0]->data));
                    hash_ = hash_range(samples_.begin(),samples_.end());
                    return;
                }

                double row_step = static_cast<double>(row_num - 1) / (sample_size - 1);
                std::vector<uint32_t> sample_row_idx(2*sample_size+1,0);
                for(size_t i=0; i<sample_size; i++){
                    sample_row_idx[2*i] = i==0 ? 0 : static_cast<size_t>(std::round(i * row_step)) - 1;
                    sample_row_idx[2*i + 1] = static_cast<size_t>(std::round(i * row_step));
                }

                samples_.resize(sample_size,0);
                gatherContinuousColWithIndex(std::make_tuple(&column, 0, 0), 2*sample_size,
                    (uint8_t*)samples_.data(), sizeof(uint32_t), sample_row_idx.data());
            }

            hash_ = hash_range(samples_.begin(),samples_.end());
        }

        // 相等比较运算符
        bool operator==(const ColumnSample &other) const {
            return type_ == other.type_
                && page_num_ == other.page_num_
                && samples_ == other.samples_
                && hash_ == other.hash_;
        }

        bool operator!=(const ColumnSample &other) const {
            return !(*this == other);
        }
    };

    // 对一个表中的多个列的采样
    class TableSample{
    public:
        size_t                          num_rows_{0};
        std::map<size_t,ColumnSample>   column_samples_;
    };

    // DONT_CACHE: 不要缓存。可能该节点是scan，或者该join的构建侧包含VARCAHR，或者深度过深，或者被释放掉了
    // NEED_CACHE: 需要缓存。表明该节点希望得到缓存。经过查找缓存和实际执行后，变为USE_CACHE或者OWN_CACHE
    // USE_CACHE: 使用缓存。此时对应的hashmaps_中存储的是被缓存的哈希表
    // OWN_CACHE: 拥有缓存。此时对应的hashmaps_中存储的是该语句执行中生成的哈希表
    enum CacheType{DONT_CACHE, NEED_CACHE, USE_CACHE, OWN_CACHE};

    QueryCache(const Plan& plan)
    : nodes_(plan.nodes), inputs_sample_(plan.inputs.size()), root(plan.root), hashes_(plan.nodes.size(),INVALID_HASH),
    cache_types_(plan.nodes.size(),DONT_CACHE), node_depth_(plan.nodes.size(),0), hashmaps_(plan.nodes.size(),nullptr){
        for(size_t i=0; i<plan.inputs.size(); i++){
            inputs_sample_[i].num_rows_=plan.inputs[i].num_rows;
        }
#ifdef NO_CACHE
        return;
#endif
        // 计算各节点哈希值
        calculateNodeHash(plan,plan.root);
    }

    void print() const{
        // 打印节点得到哈希值和类型
        for(size_t i=0; i<nodes_.size(); i++){
            if(isJoinNode(i)){
                printf("Join %zu ", i);
            } else {
                printf("Scan %zu ", i);
            }
            if(cache_types_[i]==DONT_CACHE){
                printf("hash %lu DONT_CACHE\n", hashes_[i]);
            } else if(cache_types_[i]==NEED_CACHE){
                printf("hash %lu NEED_CACHE\n", hashes_[i]);
            } else if(cache_types_[i]==USE_CACHE){
                printf("hash %lu USE_CACHE\n", hashes_[i]);
            } else {
                printf("hash %lu OWN_CACHE\n", hashes_[i]);
            }
        }

        // 打印基表采样结果
        for(size_t i=0; i<inputs_sample_.size(); i++){
            const TableSample& table_sample = inputs_sample_[i];
            for(const auto& [col_id, col_sample]:table_sample.column_samples_){
                if(col_sample.type_==DataType::INT32){
                    printf("table %zu int col %lu hash %lu samples [",i,col_id,col_sample.hash_);
                } else {
                    printf("table %zu str col %lu hash %lu samples [",i,col_id,col_sample.hash_);
                }
                for(uint64_t s:col_sample.samples_){
                    printf("%lu, ",s);
                }
                printf("]\n");
            }
        }
    }

    // 生成一个哈希表
    inline void generateCache(size_t node_id){
        cache_types_[node_id] = OWN_CACHE;
        hashmaps_[node_id] = new Hashmap();
    }

    // 生成一个哈希表，但是这个哈希表不要保留为缓存
    inline void generateTmpCache(size_t node_id){
        cache_types_[node_id] = DONT_CACHE;
        hashmaps_[node_id] = new Hashmap();
    }

    // 使用哈希表缓存
    inline void setCache(size_t node_id, Hashmap* hashmap){
        cache_types_[node_id] = USE_CACHE;
        hashmaps_[node_id] = hashmap;
    }

    // 无效化该节点上的缓存（如果有）
    inline void invalidCache(size_t node_id){
        cache_types_[node_id] = DONT_CACHE;
        if(hashmaps_[node_id]!= nullptr){
            delete hashmaps_[node_id];
            hashmaps_[node_id] = nullptr;
        }
    }

    inline bool isNaiveJoin(size_t node_id){
        if(!isJoinNode(node_id)){
            return false;
        }
        const JoinNode &join = std::get<JoinNode>(nodes_[node_id].data);
        size_t build_node = join.build_left ? join.left : join.right;
        if(isJoinNode(build_node)){
            return false;
        }
        const ScanNode &scan = std::get<ScanNode>(nodes_[build_node].data);
        if(inputs_sample_[scan.base_table_id].num_rows_==1){
            return true;
        }
        return false;
    }

    inline CacheType getCacheType(size_t node_id) const{
        return cache_types_[node_id];
    }

    inline uint64_t getNodeHash(size_t node_id) const{
        return hashes_[node_id];
    }

    inline uint64_t getBuildSideHash(size_t node_id) const{
        if(!isJoinNode(node_id)){
            return INVALID_HASH;
        }
        const JoinNode &join = std::get<JoinNode>(nodes_[node_id].data);
        size_t build_node = join.build_left ? join.left : join.right;
        return hashes_[build_node];
    }

    inline Hashmap* getHashmap(size_t node_id){
        return hashmaps_[node_id];
    }

    inline bool isJoinNode(size_t node_id) const{
        return std::holds_alternative<JoinNode>(nodes_[node_id].data);
    }

    // 判断两个query中的指定节点（的输出）是否完全相同
    bool isSame(size_t node_id, const QueryCache* other_query, size_t other_node_id) const{
        uint64_t this_hash = hashes_[node_id];
        uint64_t other_hash = other_query->hashes_[other_node_id];
        if(this_hash!=other_hash || this_hash==INVALID_HASH){
            return false;
        }

        const PlanNode& this_node = nodes_[node_id];
        const PlanNode& other_node = other_query->nodes_[other_node_id];

        if (std::holds_alternative<ScanNode>(this_node.data)) {
            if(!std::holds_alternative<ScanNode>(other_node.data)){
                return false;
            }
            const ScanNode& this_scan = std::get<ScanNode>(this_node.data);
            const ScanNode& other_scan = std::get<ScanNode>(other_node.data);

            // 比较两个scan是否完全相同
            if(this_node.output_attrs.size()!=other_node.output_attrs.size()){
                return false;
            }
            for(size_t i=0; i<this_node.output_attrs.size(); i++){
                if(getColumnSample(this_scan.base_table_id,std::get<0>(this_node.output_attrs[i]))
                    != other_query->getColumnSample(other_scan.base_table_id,std::get<0>(other_node.output_attrs[i]))){
                    return false;
                }
            }
        } else {
            if(!std::holds_alternative<JoinNode>(other_node.data)){
                return false;
            }
            const JoinNode& this_join = std::get<JoinNode>(this_node.data);
            const JoinNode& other_join = std::get<JoinNode>(other_node.data);

            if(this_node.output_attrs!=other_node.output_attrs){
                return false;
            }
            if(this_join.left_attr!=other_join.left_attr || this_join.right_attr!=other_join.right_attr){
                return false;
            }
            // 递归比较左右两侧的节点
            if(!isSame(this_join.left,other_query,other_join.left)){
                return false;
            }
            if(!isSame(this_join.right,other_query,other_join.right)){
                return false;
            }
        }
        return true;
    }

    // 判断两个query中的指定Join节点的构建侧是否完全相同
    bool isBuildSideSame(size_t node_id, const QueryCache* other_query, size_t other_node_id) const{
        if(!isJoinNode(node_id) || !other_query->isJoinNode(other_node_id)){
            return false;
        }
        const JoinNode &join = std::get<JoinNode>(nodes_[node_id].data);
        size_t build_node = join.build_left ? join.left : join.right;
        const JoinNode &other_join = std::get<JoinNode>(other_query->nodes_[other_node_id].data);
        size_t other_build_node = other_join.build_left ? other_join.left : other_join.right;

        return isSame(build_node, other_query, other_build_node);
    }

    // 获取本query保存的哈希表的大小
    size_t getCacheSize() const{
        size_t total_size=0;
        for(size_t i=0; i<cache_types_.size(); i++){
            if(cache_types_[i]==OWN_CACHE && hashmaps_[i]!=nullptr){
                total_size += hashmaps_[i]->getMemSize();
            }
        }
        return total_size;
    }

    size_t getNodeNum() const{
        return nodes_.size();
    }

private:

    std::vector<PlanNode> nodes_;
    std::vector<TableSample> inputs_sample_;
    size_t root;

    inline static const uint64_t INVALID_HASH=1;   // 无效的哈希值
    std::vector<uint64_t> hashes_;          // 每个节点（输出）的哈希值

    std::vector<CacheType> cache_types_;    // 每个节点是否需要缓存哈希表
    std::vector<Hashmap*> hashmaps_;        // 每个（join）节点（构建侧）的哈希表（可能是空的）

    static const size_t MAX_NODE_DEPTH=3;   // 深度超过MAX_NODE_DEPTH的节点不缓存
    std::vector<size_t> node_depth_;        // 每个节点的深度。scan节点深度为0，join节点深度为其构建侧深度+1

    // 对table_id的col_id列做采样并存储
    inline void addColumnSamples(size_t table_id, size_t col_id, const Column& column){
        if(inputs_sample_[table_id].column_samples_.find(col_id) == inputs_sample_[table_id].column_samples_.end()){
            inputs_sample_[table_id].column_samples_.emplace(col_id, ColumnSample(column, getTableSize(table_id)));
        }
    }

    inline uint64_t getColumnHash(size_t table_id, size_t col_id){
        return inputs_sample_[table_id].column_samples_.at(col_id).hash_;
    }

    const ColumnSample& getColumnSample(size_t table_id, size_t col_id) const{
        return inputs_sample_[table_id].column_samples_.at(col_id);
    }

    inline size_t getTableSize(size_t table_id){
        return inputs_sample_[table_id].num_rows_;
    }

    // 计算nodes_中node_id节点的哈希值，基于当前QueryCache中已经计算的节点的哈希值
    uint64_t calculateNodeHash(const Plan& plan, size_t node_id){
        if(hashes_[node_id] != INVALID_HASH){
            return hashes_[node_id];
        }

        const PlanNode& node = nodes_[node_id];
        const std::vector<std::tuple<size_t, DataType>>& node_output = node.output_attrs;

        return std::visit([&](auto&& node_data) -> uint64_t {
            using T = std::decay_t<decltype(node_data)>;
            if constexpr (std::is_same_v<T, ScanNode>) {
                // 如果表中含有VARCHAR列，不再采样，直接设为无效哈希值
                for(auto [_, col_type] : node_output){
                    if(col_type!=DataType::INT32){
                        hashes_[node_id] = INVALID_HASH;
                        node_depth_[node_id] = 0;
                        return INVALID_HASH;
                    }
                }

                // 对基表做采样
                size_t table_id = node_data.base_table_id;
                std::vector<uint64_t> col_hashes;
                for(auto [col_id, _] : node_output){
                    // 对该列做采样，存储到TableSample中
                    const Column& column = plan.inputs[table_id].columns[col_id];
                    addColumnSamples(table_id, col_id, column);
                    col_hashes.push_back(getColumnHash(table_id, col_id));
                }

                // 依据基表各列的哈希值，以及基表的总行数，生成scan节点的哈希值
                uint64_t scan_hash = hash_range(col_hashes.begin(),col_hashes.end(),getTableSize(table_id));
                hashes_[node_id] = scan_hash;
                node_depth_[node_id] = 0;
                return scan_hash;
            } else if constexpr (std::is_same_v<T, JoinNode>) {
                // 分获取左右算子的哈希值，如果任意一侧无效，则设置该节点哈希值为无效
                uint64_t left_hash = calculateNodeHash(plan,node_data.left);
                uint64_t right_hash = calculateNodeHash(plan,node_data.right);
                node_depth_[node_id] = node_data.build_left ? node_depth_[node_data.left] : node_depth_[node_data.right];
                node_depth_[node_id] ++;
                // 判断本join的哈希表是否需要被被缓存
                if(node_depth_[node_id] <= MAX_NODE_DEPTH &&
                    ((node_data.build_left && left_hash!=INVALID_HASH) || (!node_data.build_left && right_hash!=INVALID_HASH))){
                    cache_types_[node_id]=NEED_CACHE;
                }
                if(left_hash==INVALID_HASH || right_hash==INVALID_HASH){
                    hashes_[node_id] = INVALID_HASH;
                    return INVALID_HASH;
                }
                std::vector<uint64_t> datas;
                datas.push_back(left_hash);
                datas.push_back(right_hash);
                // 加入构建侧、键值列号信息
                datas.push_back(node_data.build_left);
                datas.push_back(node_data.left_attr);
                datas.push_back(node_data.right_attr);
                // 加入输出行号信息
                for(auto [col_id, _] : node_output){
                    datas.push_back(col_id);
                }
                uint64_t join_hash = hash_range(datas.begin(),datas.end());
                hashes_[node_id] = join_hash;
                return join_hash;
            }
        }, node.data);
    }
};


// 负责哈希表缓存的查找与淘汰
class CacheManager{
    // 保存的所有QueryCache。最后面的是最新被使用的，淘汰时淘汰最前面的。
    std::vector<QueryCache*> queries_;

    // 使用 list 存储哈希表缓存，最前面的是使用最少的哈希表，最后面的是最新被使用的哈希表。
    // pair<queries_的下标，QueryCache.hashmaps_的下标>
    std::list<std::pair<size_t, size_t>> hashmap_caches_;

    // unordered_map 用于将节点哈希值映射到 list 中对应位置的迭代器，方便查找和移动操作。
    std::unordered_map<uint64_t, std::list<std::pair<size_t, size_t>>::iterator> caches_map_;

    inline static const size_t MAX_CACHE_SIZE = 1024ULL * 1024 * 1024 * 20;    // 缓存的最大字节数，最多20G
    size_t cache_size_{0};     // 记录了hashmap_caches_中保存的所有哈希表的总字节数目

public:
    // 生成一条语句。为这条语句使用可能的缓存。
    QueryCache* getQuery(const Plan& plan){
        QueryCache* query = new QueryCache(plan);

        // 检查query的每个join节点的哈希表是否可以缓存
        for(size_t i=0; i<plan.nodes.size(); i++){
            // 这是为了naive join的适配
            if(query->isNaiveJoin(i)){
                query->invalidCache(i);
                continue;
            }
            if(query->getCacheType(i)==QueryCache::NEED_CACHE){
                // 获取该join的构建侧节点的哈希值
                uint64_t node_hash = query->getBuildSideHash(i);
                if(caches_map_.find(node_hash)!=caches_map_.end()){
                    auto it = caches_map_.at(node_hash);
                    auto [query_id, node_id] = *it;
                    QueryCache* cached_query = queries_[query_id];
                    if(query->isBuildSideSame(i,cached_query,node_id)){
#ifdef CACHE_LOG
                        printf("use cached hashmap: query %lu node %lu\n", query_id, node_id);
#endif
                        hashmap_caches_.splice(hashmap_caches_.end(), hashmap_caches_, it);     // 将复用的哈希表移到末尾
                        query->setCache(i,cached_query->getHashmap(node_id));
                        continue;
                    }
                }
                // 没有缓存的情况下，为该节点生成一个哈希表
                query->generateCache(i);
            } else if(query->isJoinNode(i)){
                // 就算不需要缓存，Join节点也需要生成哈希表
                query->generateTmpCache(i);
            }
        }

#ifdef CACHE_LOG
        query->print();
#endif
        return query;
    }

    // 将已经执行完毕的query，加入到缓存当中
    void cacheQuery(QueryCache* query){
#ifdef NO_CACHE
        // 释放query中的所有hashmap
        for(size_t i=0; i<query->getNodeNum(); i++){
            query->invalidCache(i);
        }
        return;
#endif
        queries_.push_back(query);
        size_t query_size = query->getCacheSize();

        if(query_size > MAX_CACHE_SIZE){
            return;     // 这可能吗？
        }

        while(cache_size_ + query_size > MAX_CACHE_SIZE){
            // 空间已满，需要释放一些缓存
            auto [query_id, node_id] = hashmap_caches_.front();
            hashmap_caches_.pop_front();
            caches_map_.erase(queries_[query_id]->getBuildSideHash(node_id));
            assert(cache_size_ >= queries_[query_id]->getHashmap(node_id)->getTotalSize());
            cache_size_ -= queries_[query_id]->getHashmap(node_id)->getMemSize();
            queries_[query_id]->invalidCache(node_id);
#ifdef CACHE_LOG
            printf("free cached hashmap: query %lu node %lu\n", query_id, node_id);
#endif
        }

        // 将本查询中的OWN_CACHE的节点加入到缓存中。清理不需要缓存的哈希表
        for(size_t i=0; i<query->getNodeNum(); i++){
            if(query->isJoinNode(i)){
                QueryCache::CacheType type = query->getCacheType(i);
                Hashmap *hashmap = query->getHashmap(i);
                if(type==QueryCache::OWN_CACHE && hashmap!=nullptr){
                    hashmap_caches_.emplace_back(queries_.size()-1,i);
                    caches_map_.emplace(query->getBuildSideHash(i),--hashmap_caches_.end());
#ifdef CACHE_LOG
                    printf("add cached hashmap: query %lu node %lu\n", queries_.size()-1, i);
#endif
                } else if(type==QueryCache::DONT_CACHE){
                    query->invalidCache(i);
                }
            }
        }


        cache_size_ += query->getCacheSize();
#ifdef CACHE_LOG
        printf("cached hashmap total size: %lu\n", cache_size_);
#endif
    }

};

}