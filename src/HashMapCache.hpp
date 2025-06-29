/**
* @file HashMapCache.hpp
* @brief Defines a caching system for hash join tables to accelerate recurring subqueries.
*
* @warning THIS ENTIRE MODULE IS DISABLED BY THE `NO_CACHE` MACRO AND IS NOT USED
* IN THE FINAL EXECUTION LOGIC. It represents a proof-of-concept for a cache that
* was ultimately not used, primarily because its method for identifying identical
* subqueries relies on data sampling. While this approach is often effective, it is
* probabilistic and does not provide a 100% guarantee of correctness, which may be
* unacceptable for a database system.
*
* The intended design was to:
* 1.  **Identify recurring sub-plans:** `QueryCache` would analyze a query plan,
*     sample its input data, and generate a structural and data-based hash for each node.
* 2.  **Cache hash tables:** `CacheManager` would act as a global, LRU-based cache.
*     When a query's sub-plan hash matched a previously cached entry, the pre-built
*     hash table (`Hashmap`) would be reused, skipping the expensive build phase of a hash join.
* 3.  **Probabilistic Matching:** The correctness of the cache hit depended on comparing
*     the plan structure and a small sample of the underlying data.
*/

#include <list>
#include "plan.h"
#include "HashMap.hpp"
#include "DataStructure.hpp"

// The comments below describe the intended functionality of this disabled module.
namespace Contest {

/// @brief Combines a seed with a new hash value to produce a new seed.
inline std::size_t hash_combine(std::size_t seed, std::size_t hash_value) {
    // 0x9e3779b9 is a commonly used magic number (derived from the golden ratio) used to bring good mixing effects
    return seed ^ (hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Calculates a combined hash value for a range of elements.
 * @tparam InputIt The type of the input iterator.
 * @param first The start iterator of the range.
 * @param last The end iterator of the range.
 * @param seed An initial seed for the hash calculation.
 * @return The combined hash value for the entire range.
 */
template <typename InputIt>
std::size_t hash_range(InputIt first, InputIt last, std::size_t seed = 0) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    std::hash<value_type> hasher;  // hash calculator

    for (; first != last; ++first) {
        const auto& element = *first;
        const std::size_t element_hash = hasher(element);
        seed = hash_combine(seed, element_hash);
    }

    return seed;
}


/**
 * @class QueryCache
 * @brief Represents a single query's potential for caching.
 *
 * This class analyzes a `Plan`, samples its input data to create a statistical
 * fingerprint, and computes a hash for each node. This information is then used by
 * the `CacheManager` to determine if any sub-plans (specifically, the build sides
 * of hash joins) are identical to previously executed queries and can be reused.
 */
class QueryCache{
public:
    /**
     * @class ColumnSample
     * @brief A statistical fingerprint of a single column, based on a small sample of its data.
     */
    class ColumnSample{
    public:
        DataType                type_;
        size_t                  page_num_;
        std::vector<uint64_t>   samples_;   // Samples from the column.
        uint64_t                hash_;      // Hash value.

        inline static const size_t SAMPLE_SIZE = 20;    // Maximum number of samples. The actual number of samples is 2 * SAMPLE_SIZE.

        /**
         * @brief Constructs a sample by reading a few data points from a column.
         *
         * The sampling is a heuristic. It does not sample VARCHAR columns. For INT32,
         * it takes a few values spread across the column's pages to generate a hash.
         * @param column The column to sample.
         * @param row_num The total number of rows in the column.
         */
        ColumnSample(const Column& column, size_t row_num) : type_(column.type), page_num_(column.pages.size()){
            if(type_!=DataType::INT32 || page_num_ == 0){
                hash_ = INVALID_HASH;
                return;
            }
            // Sample SAMPLE_SIZE data points at equal intervals. If page_num_ >= SAMPLE_SIZE, each sampled page is unique, and we sample the beginning of these pages.
            if(page_num_ >= SAMPLE_SIZE){
                double page_step = static_cast<double>(page_num_ - 1) / (SAMPLE_SIZE - 1);
                for(size_t i=0; i<SAMPLE_SIZE; i++){
                    size_t page_index = static_cast<size_t>(std::round(i * page_step));
                    const Page* page = column.pages[page_index];
                    uint16_t non_null_count = *(uint16_t*)(page->data+2);
                    if(non_null_count==0){
                        samples_.push_back(*(uint32_t*)(page->data));
                    } else if(non_null_count==1){
                        samples_.push_back(*(uint64_t*)(page->data));
                    } else {
                        samples_.push_back(*(uint64_t*)(page->data+4));
                    }
                }
            } else {
                // If page_num_ < SAMPLE_SIZE, multiple data points need to be sampled from some pages. Determine the row numbers to be sampled and sample these rows.
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
                ContinuousColumn(&column, 0, 0).gather(2*sample_size,
                    (uint8_t*)samples_.data(), sizeof(uint32_t), sample_row_idx.data());
            }

            hash_ = hash_range(samples_.begin(),samples_.end());
        }

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

    /**
     * @class TableSample
     * @brief A container for `ColumnSample`s from a single table.
     */
    class TableSample{
    public:
        size_t                          num_rows_{0};
        std::map<size_t,ColumnSample>   column_samples_;
    };

    /**
     * @enum CacheType
     * @brief Describes the caching status of a join node within this query.
     */
    enum CacheType {
        DONT_CACHE, /// Do not cache. The node might be a scan, the build side of the join might contain VARCHAR, the depth might be too large, or it might have been released.
        NEED_CACHE, /// A candidate for caching, pending lookup in the global cache. Indicates that this node desires to be cached. After cache lookup and actual execution, it becomes USE_CACHE or OWN_CACHE.
        USE_CACHE,  /// Cache hit. This node will use a pre-built hash table from the cache.
        OWN_CACHE   /// Cache miss. This node will build a new hash table and offer it to the cache. The corresponding hashmaps_ entry stores the hash table generated during the execution of this statement.
    };

    /**
     * @brief Analyzes a plan to prepare it for caching logic.
     * @param plan The logical query plan.
     * @param input An optional alternative set of input tables.
     */
    QueryCache(const Plan& plan, const std::vector<ColumnarTable>* input=nullptr)
    : nodes_(plan.nodes), root(plan.root), hashes_(plan.nodes.size(),INVALID_HASH),
    cache_types_(plan.nodes.size(),DONT_CACHE), node_depth_(plan.nodes.size(),0), hashmaps_(plan.nodes.size(),nullptr){
        if(input==nullptr){
            input = &plan.inputs;
        }
        inputs_sample_.resize(input->size());
        for(size_t i=0; i<input->size(); i++){
            inputs_sample_[i].num_rows_=(*input)[i].num_rows;
        }
#ifdef NO_CACHE
        return;
#endif
        // Calculate the hash values for each node.
        calculateNodeHash(plan,plan.root,input);
    }

    ~QueryCache(){
        // Release all hash tables.
        for(size_t i=0; i<nodes_.size(); i++){
            if(cache_types_[i]!=USE_CACHE && hashmaps_[i]!=nullptr){
                delete hashmaps_[i];
            }
        }
    }

    /// @brief Prints the caching information for this query for debugging.
    void print() const{
        // Print the hash value and type of the node.
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

        // Print the sampling results of the base tables.
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

    /// @brief Marks a node to build a new hash table that will be owned by the cache.
    inline void generateCache(size_t node_id){
        cache_types_[node_id] = OWN_CACHE;
        hashmaps_[node_id] = new Hashmap();
    }

    /// @brief Marks a node to build a temporary hash table that will not be cached.
    inline void generateTmpCache(size_t node_id){
        cache_types_[node_id] = DONT_CACHE;
        hashmaps_[node_id] = new Hashmap();
    }

    /// @brief Marks a node to use an existing hash table from the cache.
    inline void setCache(size_t node_id, Hashmap* hashmap){
        cache_types_[node_id] = USE_CACHE;
        hashmaps_[node_id] = hashmap;
    }

    /// @brief Invalidates the cache status of a node and frees its hashmap if it's not from the global cache.
    inline void invalidCache(size_t node_id){
        if(cache_types_[node_id] != USE_CACHE && hashmaps_[node_id]!= nullptr){
            delete hashmaps_[node_id];
            hashmaps_[node_id] = nullptr;
        }
        cache_types_[node_id] = DONT_CACHE;
    }

    /// @brief Checks if a join node should be implemented as a NaiveJoin due to a 1-row build side.
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

    /**
     * @brief Recursively checks if a sub-plan in this query is identical to a sub-plan in another query.
     *
     * The check is probabilistic, comparing node hashes (which depend on data samples).
     * @param node_id The root of the sub-plan in this query.
     * @param other_query The other query to compare against.
     * @param other_node_id The root of the sub-plan in the other query.
     * @return `true` if the sub-plans are considered identical.
     */
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

            // Compare if two scans are identical.
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
            // Recursively compare the left and right side nodes.
            if(!isSame(this_join.left,other_query,other_join.left)){
                return false;
            }
            if(!isSame(this_join.right,other_query,other_join.right)){
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Checks if the build sides of two join nodes are identical.
     * This is the key check for determining if a hash table can be reused.
     */
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

    /// @brief Get the size of the hash tables saved by this query.
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

    inline static const uint64_t INVALID_HASH=1;   // Invalid hash value.
    std::vector<uint64_t> hashes_;          // Hash value of each node (output).

    std::vector<CacheType> cache_types_;    // Whether each node needs to cache its hash table.
    std::vector<Hashmap*> hashmaps_;        // Hash table for each (join) node (on its build side) (can be empty).

    static const size_t MAX_NODE_DEPTH=3;   // Nodes with depth exceeding MAX_NODE_DEPTH are not cached.
    std::vector<size_t> node_depth_;        // Depth of each node. Scan node depth is 0, join node depth is its build side depth + 1.

    // Sample and store the col_id column of table_id.
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

    /// @brief Calculate the hash value of the node_id in nodes_, based on the already calculated hash values in the current QueryCache.
    uint64_t calculateNodeHash(const Plan& plan, size_t node_id, const std::vector<ColumnarTable>* input){
        if(hashes_[node_id] != INVALID_HASH){
            return hashes_[node_id];
        }

        const PlanNode& node = nodes_[node_id];
        const std::vector<std::tuple<size_t, DataType>>& node_output = node.output_attrs;

        return std::visit([&](auto&& node_data) -> uint64_t {
            using T = std::decay_t<decltype(node_data)>;
            if constexpr (std::is_same_v<T, ScanNode>) {
                // If the table contains a VARCHAR column, do not sample and set the hash value to invalid.
                for(auto [_, col_type] : node_output){
                    if(col_type!=DataType::INT32){
                        hashes_[node_id] = INVALID_HASH;
                        node_depth_[node_id] = 0;
                        return INVALID_HASH;
                    }
                }

                // Sample the base table.
                size_t table_id = node_data.base_table_id;
                std::vector<uint64_t> col_hashes;
                for(auto [col_id, _] : node_output){
                    // Sample this column and store it in TableSample.
                    const Column& column = (*input)[table_id].columns[col_id];
                    addColumnSamples(table_id, col_id, column);
                    col_hashes.push_back(getColumnHash(table_id, col_id));
                }

                // Generate the hash value for the scan node based on the hash values of each column of the base table and the total number of rows.
                uint64_t scan_hash = hash_range(col_hashes.begin(),col_hashes.end(),getTableSize(table_id));
                hashes_[node_id] = scan_hash;
                node_depth_[node_id] = 0;
                return scan_hash;
            } else if constexpr (std::is_same_v<T, JoinNode>) {
                // Get the hash values of the left and right operators respectively. If either side is invalid, set this node's hash value to invalid.
                uint64_t left_hash = calculateNodeHash(plan,node_data.left,input);
                uint64_t right_hash = calculateNodeHash(plan,node_data.right,input);
                node_depth_[node_id] = node_data.build_left ? node_depth_[node_data.left] : node_depth_[node_data.right];
                node_depth_[node_id] ++;
                // Determine if the hash table of this join needs to be cached.
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
                // Add information about the build side and key column numbers.
                datas.push_back(node_data.build_left);
                datas.push_back(node_data.left_attr);
                datas.push_back(node_data.right_attr);
                // Add output row number information.
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


/**
 * @class CacheManager
 * @brief Manages the global cache of hash tables.
 *
 * This class is responsible for storing, looking up, and evicting cached `Hashmap`s
 * from completed queries. It uses a Least Recently Used (LRU) policy for eviction
 * when the total cache size exceeds a defined limit.
 */
class CacheManager{
    /// @brief A history of all processed `QueryCache` objects.
    std::vector<QueryCache*> queries_;

    /// @brief An LRU list of cached hashmaps. The back is most-recently-used.
    /// The pair stores `{index_into_queries_, node_index_within_that_query}`.
    std::list<std::pair<size_t, size_t>> hashmap_caches_;

    /// @brief A map for O(1) lookup of cached hashmaps by their build-side hash.
    std::unordered_map<uint64_t, std::list<std::pair<size_t, size_t>>::iterator> caches_map_;

    inline static const size_t MAX_CACHE_SIZE = 1024ULL * 1024 * 1024 * 20;    // Maximum cache size in bytes, up to 20GB.
    size_t cache_size_{0};     // Records the total byte size of all hash tables saved in hashmap_caches_.

public:
    ~CacheManager(){
        // Free all allocated memory.
        for(auto query:queries_){
            delete query;
        }
    }

    /**
     * @brief Factory method to create a `QueryCache` object for a new plan.
     *
     * This method orchestrates the cache lookup process. It creates a `QueryCache`
     * and then, for each join node that is a caching candidate, it checks if a
     * matching hash table already exists in the manager.
     * @param plan The query plan to process.
     * @return A new `QueryCache` object, configured to either use existing cache
     *         entries or generate new ones.
     */
    QueryCache* getQuery(const Plan& plan, const std::vector<ColumnarTable>* input=nullptr){
        QueryCache* query = new QueryCache(plan,input);

        // Check if the hash table of each join node in the query can be cached.
        for(size_t i=0; i<query->getNodeNum(); i++){
            // This is for adapting to naive joins.
            if(query->isNaiveJoin(i)){
                query->invalidCache(i);
                continue;
            }
            if(query->getCacheType(i)==QueryCache::NEED_CACHE){
                // Get the hash value of the build side node of this join.
                uint64_t node_hash = query->getBuildSideHash(i);
                if(caches_map_.find(node_hash)!=caches_map_.end()){
                    auto it = caches_map_.at(node_hash);
                    auto [query_id, node_id] = *it;
                    QueryCache* cached_query = queries_[query_id];
                    if(query->isBuildSideSame(i,cached_query,node_id)){
#ifdef CACHE_LOG
                        printf("use cached hashmap: query %lu node %lu\n", query_id, node_id);
#endif
                        hashmap_caches_.splice(hashmap_caches_.end(), hashmap_caches_, it);     // Move the reused hash table to the end.
                        query->setCache(i,cached_query->getHashmap(node_id));
                        continue;
                    }
                }
                // If there is no cache, generate a hash table for this node.
                query->generateCache(i);
#ifdef CACHE_LOG
                printf("generate cached hashmap: query %lu node %lu\n", queries_.size(), i);
#endif
            } else if(query->isJoinNode(i)){
                // Even if caching is not needed, a Join node still needs to generate a hash table.
                query->generateTmpCache(i);
#ifdef CACHE_LOG
                printf("generate tmp hashmap: query %lu node %lu\n", queries_.size(), i);
#endif
            }
        }

#ifdef CACHE_LOG
        query->print();
#endif
        return query;
    }

    // Add an executed query to the cache.
    void cacheQuery(QueryCache* query){
        queries_.push_back(query);

#ifdef NO_CACHE
        // Release all hashmaps in the query.
        for(size_t i=0; i<query->getNodeNum(); i++){
            query->invalidCache(i);
        }
        return;
#endif

        size_t query_size = query->getCacheSize();

        if(query_size > MAX_CACHE_SIZE){
            return;     // Is this possible?
        }

        while(cache_size_ + query_size > MAX_CACHE_SIZE){
            // The space is full, need to free some cache.
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

        // Add the OWN_CACHE nodes from this query to the cache. Clean up hash tables that do not need to be cached.
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
#ifdef CACHE_LOG
                    printf("free tmp hashmap: query %lu node %lu\n", queries_.size()-1, i);
#endif
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