#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <algorithm>
#include <thread>
#include "plan.h"
#include "DataStructure.hpp"

//#define OPTIMIZE_LOG

namespace Contest {

class SpaceSaving {
public:
    // 构造函数，指定候选集合容量（通常是 top-k 数量）
    SpaceSaving(size_t capacity) : capacity(capacity), total(0) { }

    // 存储候选元素的结构体
    struct Candidate {
        int count;  // 估计的出现次数
        int error;  // 替换时引入的误差上界
    };

    // 处理数据流中的每个元素
    void process(int32_t value) {
        total ++;
        auto it = candidates.find(value);
        if (it != candidates.end()) {
            // 如果元素已在候选集合中，更新计数
            int oldCount = it->second.count;
            // 从有序 set 中移除旧的记录
            order.erase({oldCount, value});
            // 更新计数
            it->second.count++;
            // 重新插入 set，使用新的计数
            order.insert({it->second.count, value});
        } else {
            if (candidates.size() < capacity) {
                // 候选集合未满，直接插入新元素，计数为1，误差为0
                candidates.emplace(value, Candidate{1, 0});
                order.insert({1, value});
            } else {
                // 候选集合已满，快速找到计数最小的候选项
                auto minIt = order.begin(); // set 按 count 升序排列
                int minCount = minIt->first;
                int32_t minKey = minIt->second;

                // 从 set 和 map 中移除计数最小的候选项
                order.erase(minIt);
                candidates.erase(minKey);

                // 用当前元素替换，设置新计数为 minCount + 1，误差为 minCount
                candidates.emplace(value, Candidate{minCount + 1, minCount});
                order.insert({minCount + 1, value});
            }
        }
    }

    // 返回当前候选集合（无排序），格式为 vector<pair<key, Candidate>>
    std::vector<std::pair<int32_t, Candidate>> getCandidates() const {
        std::vector<std::pair<int32_t, Candidate>> result;
        for (const auto &p : candidates) {
            result.push_back(p);
        }
        return result;
    }

    // 返回按计数降序排序的候选集合
    std::vector<std::pair<int32_t, Candidate>> getSortedCandidates() const {
        auto cand = getCandidates();
        std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
            return a.second.count > b.second.count;
        });
        return cand;
    }

    size_t getTotal() const {return total;}

private:
    size_t total;     // 总计数
    size_t capacity;  // 保存候选个数（top-k）
    // 使用 unordered_map 快速查找候选项
    std::unordered_map<int32_t, Candidate> candidates;
    // 使用 set 按 (count, key) 排序，以便快速获得计数最小项
    std::set<std::pair<int, int32_t>> order;
};


class SpaceSavingSmall {
public:
    // 构造函数指定候选集合容量（一般设为 k）
    SpaceSavingSmall(size_t capacity) : capacity(capacity), total(0) { }

    // 用于存储候选项的结构体
    struct Candidate {
        int32_t key;   // 值
        int count;     // 估计的出现次数
        int error;     // 误差（由于替换操作引入的错误上界）
    };

    // 处理数据流中的每个元素
    void process(int32_t value) {
        total++;
        // ① 如果 value 已经在候选集合中，计数直接加 1
        for (auto &cand : candidates) {
            if (cand.key == value) {
                cand.count++;
                return;
            }
        }

        // ② 如果候选集合未满，则新增候选项
        if (candidates.size() < capacity) {
            candidates.push_back({value, 1, 0});
        } else {
            // ③ 候选集合已满，找到计数最小的候选项
            auto min_it = std::min_element(candidates.begin(), candidates.end(),
                [](const Candidate &a, const Candidate &b) {
                    return a.count < b.count;
                });
            int min_count = min_it->count;
            // 用新元素替换该候选，计数设置为 min_count + 1，误差记录为 min_count
            *min_it = {value, min_count + 1, min_count};
        }
    }

    // 返回当前候选集合（无排序）
    std::vector<Candidate> getCandidates() const {
        return candidates;
    }

    // 返回按计数从高到低排序的候选集合
    std::vector<Candidate> getSortedCandidates() const {
        auto sorted = candidates;
        std::sort(sorted.begin(), sorted.end(), [](const Candidate &a, const Candidate &b) {
            return a.count - a.error > b.count - b.error;
        });
        return sorted;
    }

    size_t getTotal() const {return total;}

    void reset(){
        total = 0;
        candidates.clear();
    }

    // 判断当前已经处理的 total 个元素是否都互不重复
    // 根据空间节省算法的理论：
    // (1) 如果 total < capacity：所有候选的 count 应该均为 1, error = 0，总数等于 total。
    // (2) 如果 total >= capacity：设 q = total / capacity, r = total % capacity，
    //     则候选集合中应有 r 个项的 count 为 q+1，(capacity - r) 个项的 count 为 q，
    //     并且每个候选的 error 应该等于 count - 1.
    bool allDistinct() const {
        if (total == 0)
            return true;  // 无数据认为全不同

        if (total < capacity) {
            if (candidates.size() != total)
                return false;
            for (const auto &cand : candidates) {
                if (cand.count != 1 || cand.error != 0)
                    return false;
            }
            return true;
        } else {
            // total >= capacity，候选集合应已满 (size == capacity)
            if (candidates.size() != capacity)
                return false;
            int q = total / capacity;
            int r = static_cast<int>(total % capacity);
            int countPlus = 0;   // 计数为 q+1 的个数
            int countNormal = 0; // 计数为 q 的个数
            for (const auto &cand : candidates) {
                if (cand.error + 1 != cand.count)
                    return false;
                if (cand.count == q + 1) {
                    countPlus++;
                } else if (cand.count == q) {
                    countNormal++;
                } else {
                    return false;
                }
            }
            if (countPlus != r || countNormal != static_cast<int>(capacity) - r)
                return false;
            return true;
        }
    }

private:
    size_t total;     // 总计数
    size_t capacity;                 // 候选集合的容量，即最多保留的候选数（Top-k）
    std::vector<Candidate> candidates;  // 候选数据集合
};


class CountMinSketch {
    // 这是一个大素数，用于哈希函数的模运算
    static constexpr unsigned long LONG_PRIME = 4294967311ul;

    // w 表示二维计数器数组的宽度，即每一行拥有的计数器数量；d 表示数组的深度，也即使用的哈希函数个数
    unsigned int w,d;

    // 总计数
    unsigned int total;

    // 保存计数器的二维数组，大小为 d × w
    int **C;

    // 保存用于哈希函数参数的二维数组。每个哈希函数用一对 (aj, bj) 表示，这些值均从 Z_p（模 LONG_PRIME 的域）中随机选取
    int **hashes;

    // 记录最高的频数（估计值）
    unsigned int max_frequency;

    // 为第 i 个哈希函数生成参数。
    void genajbj(int** hashes, int i) {
        hashes[i][0] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
        hashes[i][1] = int(float(rand())*float(LONG_PRIME)/float(RAND_MAX) + 1);
    }

    // num_buckets为桶数目
    // num_hashes为哈希函数个数
    void init(size_t num_buckets, size_t num_hashes){
        w = num_buckets;
        d = num_hashes;
        //printf("cms w=%d d=%d\n",w,d);
        total = 0;
        // 初始化二维计数器数组
        C = new int *[d];
        unsigned int i, j;
        for (i = 0; i < d; i++) {
            C[i] = new int[w];
            for (j = 0; j < w; j++) {
                C[i][j] = 0;
            }
        }
        // 初始化 d 个哈希函数
        srand(time(NULL));
        hashes = new int* [d];
        for (i = 0; i < d; i++) {
            hashes[i] = new int[2];
            genajbj(hashes, i);
        }

        max_frequency = 0;
    }

public:

    // num_buckets为桶数目
    // num_hashes为哈希函数个数
    CountMinSketch(size_t num_buckets, size_t num_hashes){
        init(num_buckets, num_hashes);
    }

//    // ep 为误差参数， 0.01 < ep < 1
//    // gamma 表示保证结果精度的概率参数，0 < gamm < 1
//    CountMinSketch(float eps, float gamma) {
//        assert(0.009 <= eps && eps < 1);
//        assert(0 < gamma && gamma < 1);
//        init(ceil(exp(1)/eps), ceil(log(1/gamma)));
//    }

    // 将给定元素 item 更新 c 次
    void update(int item, int c=1) {
        total += c;
        unsigned int hashval = 0;
        int minval = std::numeric_limits<int>::max();
        for (unsigned int j = 0; j < d; j++) {
            // 计算哈希值：先对 LONG_PRIME 取模，再对宽度 w 取模
            hashval = (((long)hashes[j][0] * item + hashes[j][1]) % LONG_PRIME) % w;
            C[j][hashval] += c;
            if (C[j][hashval] < minval)
                minval = C[j][hashval];
        }
        // 更新全局最大频次（记录最高的估计值）
        if (minval > max_frequency)
            max_frequency = minval;
    }

    // 估计 item 出现的次数
    unsigned int estimate(int item) {
        int minval = std::numeric_limits<int>::max();
        unsigned int hashval = 0;
        for (unsigned int j = 0; j < d; j++) {
            hashval = ((long)hashes[j][0] * item + hashes[j][1]) % LONG_PRIME % w;
            minval = minval < C[j][hashval] ? minval : C[j][hashval];
        }
        return minval;
    }

    unsigned int getTotal() const {return total;}

    unsigned int getMaxFrequency() const {return max_frequency;}

    ~CountMinSketch() {
        unsigned int i;
        for (i = 0; i < d; i++) {
            delete[] C[i];
        }
        delete[] C;

        for (i = 0; i < d; i++) {
            delete[] hashes[i];
        }
        delete[] hashes;
    }

};


// 统计列的topK频繁出现的元素，及其出现次数
// SpaceSavingTopK统计的很准确，但是耗时无法接受。单线程统计a1的数据，需要600ms
std::vector<std::pair<int32_t, SpaceSaving::Candidate>> spaceSavingTopK(const Column& col, float sample_rate, size_t k){
    // 按照采样比例选择一个page中的部分元素来统计
    assert(col.type==DataType::INT32);
    SpaceSaving space_saving(k);
    for(const Page* page:col.pages){
        const int32_t* base = getPageData<int32_t>(page);
        //size_t page_rows = getRowCount(page);
        size_t page_nonnull_rows = getNonNullCount(page);
        size_t k = (size_t)(page_nonnull_rows * sample_rate);
        if(k==0) k=1;       // 至少要采样一行。
        // 采样k行（不超过page_nonnull_rows），不采样空值。
        std::unordered_set<size_t> selected;
        for (size_t i = page_nonnull_rows - k + 1; i <= page_nonnull_rows; i++) {
            size_t r = rand() % i;
            if (selected.find(r) == selected.end()){
                selected.insert(r);
                space_saving.process(base[r]);
            } else {
                selected.insert(i - 1);
                space_saving.process(base[i - 1]);
            }
        }
    }

    auto topk = space_saving.getSortedCandidates();
//    for(auto [key,statistic]:topk){
//        printf("key %d count %d error %d\n",key,statistic.count,statistic.error);
//    }
    printf("max %.3f error %d\n",(float)topk.front().second.count/space_saving.getTotal(),topk.front().second.error);
    return topk;
}


// 统计列的topK频繁出现的元素，及其出现次数
// SpaceSavingTopK统计的很准确，但是耗时无法接受。单线程统计a1的数据，需要600ms
std::vector<SpaceSavingSmall::Candidate> spaceSavingSmallTopK(const Column& col, float sample_rate, size_t k, size_t start_page=0, size_t page_step=1){
    // 按照采样比例选择一个page中的部分元素来统计
    assert(col.type==DataType::INT32);
    assert(start_page<col.pages.size());
    SpaceSavingSmall space_saving(k);
    for(size_t p=start_page;p<col.pages.size();p+=page_step){
        const Page* page = col.pages[p];
        const int32_t* base = getPageData<int32_t>(page);
        //size_t page_rows = getRowCount(page);
        size_t page_nonnull_rows = getNonNullCount(page);
        size_t k = (size_t)(page_nonnull_rows * sample_rate);
        if(k==0) k=1;       // 至少要采样一行。
        // 采样k行（不超过page_nonnull_rows），不采样空值。
        std::unordered_set<size_t> selected;
        for (size_t i = page_nonnull_rows - k + 1; i <= page_nonnull_rows; i++) {
            size_t r = rand() % i;
            if (selected.find(r) == selected.end()){
                selected.insert(r);
                space_saving.process(base[r]);
            } else {
                selected.insert(i - 1);
                space_saving.process(base[i - 1]);
            }
        }
    }

    auto topk = space_saving.getSortedCandidates();
    //    for(auto [key,statistic]:topk){
    //        printf("key %d count %d error %d\n",key,statistic.count,statistic.error);
    //    }
    printf("max %.3f error %d\n",(float)topk.front().count/space_saving.getTotal(),topk.front().error);
    return topk;
}


// 抽取page中连续的元素来统计，减少随机访问
std::vector<SpaceSavingSmall::Candidate> spaceSavingSmallTopKContinue(const Column& col, float sample_rate, size_t k, size_t start_page=0, size_t page_step=1){
    // 按照采样比例选择一个page中的部分元素来统计
    assert(col.type==DataType::INT32);
    assert(start_page<col.pages.size());
    SpaceSavingSmall space_saving(k);
    for(size_t p=start_page;p<col.pages.size();p+=page_step){
        const Page* page = col.pages[p];
        //size_t page_rows = getRowCount(page);
        size_t page_nonnull_rows = getNonNullCount(page);
        size_t sample_num = (size_t)(page_nonnull_rows * sample_rate);
        if(sample_num==0) sample_num=1;       // 至少要采样一行。
        if(sample_num>page_nonnull_rows) sample_num=page_nonnull_rows;
        // 采样连续的k行（不超过page_nonnull_rows），不采样空值。
        size_t pos = rand() % (page_nonnull_rows - sample_num + 1);
        const int32_t* base = getPageData<int32_t>(page) + pos;
        for(size_t j = 0; j<sample_num; j++){
            space_saving.process(base[j]);
        }
    }

    auto topk = space_saving.getSortedCandidates();
    //    for(auto [key,statistic]:topk){
    //        printf("key %d count %d error %d\n",key,statistic.count,statistic.error);
    //    }
    printf("max %.3f error %d\n",(float)topk.front().count/space_saving.getTotal(),topk.front().error);
    return topk;
}


// 抽取page中连续的元素来统计，并且在可能为主键列时扩大抽样步长，减少抽样次数
std::vector<SpaceSavingSmall::Candidate> spaceSavingSmallTopKFast(const Column& col, float sample_rate, size_t k, size_t start_page=0, size_t page_step=1){
    // 按照采样比例选择一个page中的部分元素来统计
    assert(col.type==DataType::INT32);
    assert(start_page<col.pages.size());
    SpaceSavingSmall space_saving(k);
    int maybe_primary_key = 0;
    for(size_t p=start_page;p<col.pages.size();p+=page_step){
        const Page* page = col.pages[p];
        //size_t page_rows = getRowCount(page);
        size_t page_nonnull_rows = getNonNullCount(page);
        size_t sample_num = (size_t)(page_nonnull_rows * sample_rate);
        if(sample_num==0) sample_num=1;       // 至少要采样一行。
        if(sample_num>page_nonnull_rows) sample_num=page_nonnull_rows;
        // 采样连续的k行（不超过page_nonnull_rows），不采样空值。
        size_t pos = rand() % (page_nonnull_rows - sample_num + 1);
        const int32_t* base = getPageData<int32_t>(page) + pos;
        for(size_t j = 0; j<sample_num; j++){
            space_saving.process(base[j]);
        }

        if(maybe_primary_key>=0){
            maybe_primary_key ++;
            // 判断有没有可能不是主键列了
            if(maybe_primary_key % 4==0){
                if(space_saving.allDistinct()){
                    page_step *= 2;             // 如果还有可能是主键列，那么每4次步长翻倍
                    //printf("page_step %zu\n",page_step);
                } else {
                    maybe_primary_key = -1;     // 该列不可能是主键，以后都不用再判断了
                }
            }
        }

    }

    auto topk = space_saving.getSortedCandidates();
    //    for(auto [key,statistic]:topk){
    //        printf("key %d count %d error %d\n",key,statistic.count,statistic.error);
    //    }

    if(maybe_primary_key > 0){
        printf("max %.3f error %d maybe pk\n",(float)topk.front().count/space_saving.getTotal(),topk.front().error);
    } else {
        printf("max %.3f error %d mustbe fk\n",(float)topk.front().count/space_saving.getTotal(),topk.front().error);
    }
    return topk;
}


// 估计 top1 频数（注意：返回的是一个上界估计，可能会高于真实值）
int countMinSketchTop1(const Column& col, size_t num_rows, float sample_rate) {
    // 按照采样比例选择一个page中的部分元素来统计
    assert(col.type==DataType::INT32);
    CountMinSketch cms(1000,1);
    for(const Page* page:col.pages){
        const int32_t* base = getPageData<int32_t>(page);
        //size_t page_rows = getRowCount(page);
        size_t page_nonnull_rows = getNonNullCount(page);
        size_t k = (size_t)(page_nonnull_rows * sample_rate);
        if(k==0) k=1;       // 至少要采样一行。
        // 采样k行（不超过page_nonnull_rows），不采样空值。
        std::unordered_set<size_t> selected;
        for (size_t i = page_nonnull_rows - k + 1; i <= page_nonnull_rows; i++) {
            size_t r = rand() % i;
            if (selected.find(r) == selected.end()){
                selected.insert(r);
                cms.update(base[r]);
            } else {
                selected.insert(i - 1);
                cms.update(base[i - 1]);
            }
        }
    }

    //printf("max frequency %d samples %d rows %d\n",cms.getMaxFrequency(),cms.getTotal(),num_rows);
    printf("max %.3f\n",(float)cms.getMaxFrequency()/cms.getTotal());
    return cms.getMaxFrequency();
}


// 递归查找节点 nodeIndex 的输出属性 attrIndex 对应的基础列。返回 pair {base_table_id, column_id}。
std::pair<size_t, size_t> getBaseColumn(const Plan& plan, size_t nodeIndex, size_t attrIndex) {
    const PlanNode& node = plan.nodes[nodeIndex];
    // 若扫描节点，则直接返回扫描节点中的列信息
    if (std::holds_alternative<ScanNode>(node.data)) {
        const ScanNode& scan = std::get<ScanNode>(node.data);
        size_t colIndex = std::get<0>(node.output_attrs[attrIndex]);
        return { scan.base_table_id, colIndex };
    }
    // 若为 JoinNode，则按照输出属性的拼接规则：左子节点的输出在前，右子节点的输出紧随其后
    else if (std::holds_alternative<JoinNode>(node.data)) {
        const JoinNode& join = std::get<JoinNode>(node.data);
        size_t colIndex = std::get<0>(node.output_attrs[attrIndex]);
        const PlanNode& leftNode = plan.nodes[join.left];
        size_t leftCount = leftNode.output_attrs.size();
        if (colIndex < leftCount) {
            return getBaseColumn(plan, join.left, colIndex);
        } else {
            return getBaseColumn(plan, join.right, colIndex - leftCount);
        }
    }
    // 默认返回非法值（理论上不应到达这里）
    return { size_t(-1), size_t(-1) };
}


// 统计 plan 中所有参与 join 的列，并对每个 join 键列调用 getTopK 进行统计打印
void estimatePlanSingle(const Plan& plan, size_t k) {
    // 使用 set 存储唯一(join键)的 (表号, 列号) 对
    std::set<std::pair<size_t, size_t>> joinKeyColumns;  // pair: {base_table_id, column_id}

    // 遍历所有 PlanNode
    for (size_t nodeIndex = 0; nodeIndex < plan.nodes.size(); nodeIndex++) {
        const PlanNode& node = plan.nodes[nodeIndex];
        // 仅处理 JoinNode（仅 JoinNode 才会有 join 键）
        if (std::holds_alternative<JoinNode>(node.data)) {
            const JoinNode& join = std::get<JoinNode>(node.data);

            // 对左侧 join key：取得左子节点中对应的属性信息
            std::pair<size_t, size_t> leftKey = getBaseColumn(plan, join.left, join.left_attr);
            joinKeyColumns.insert(leftKey);

            // 对右侧 join key：取得右子节点中对应的属性信息
            std::pair<size_t, size_t> rightKey = getBaseColumn(plan, join.right, join.right_attr);
            joinKeyColumns.insert(rightKey);
        }
    }

    // 遍历唯一的 joinKeyColumns，统计打印每一列的 TopK
    for (const auto& key : joinKeyColumns) {
        size_t tableId = key.first;
        size_t colId = key.second;
        assert(tableId <  plan.inputs.size());
        const ColumnarTable& table = plan.inputs[tableId];
        assert(colId < table.columns.size());
        const Column& col = table.columns[colId];

        printf("TopK for join key (table %zu, column %zu, row num %zu)\n", tableId, colId, table.num_rows);



        auto topk = spaceSavingSmallTopKFast(col, 0.01, k);
        //auto topk = spaceSavingSmallTopKContinue(col, 0.01, k);
        //auto topk = spaceSavingTopK(col, table.num_rows, 1.0, k);
        //uint32_t max_frequency = countMinSketchTop1(col, table.num_rows, 0.01);

    }
}


// 定义每个任务的数据结构
struct TopKTask {
    size_t tableId;   // 对应 plan.inputs 中的表号
    size_t colId;     // 表中列号
    size_t startPage; // 任务从该页开始处理
    size_t step;      // 任务每次跳跃的步长
    std::vector<SpaceSavingSmall::Candidate> output;    // 任务处理后输出的局部统计结果
};


// 根据给定 plan 中的 join 键列，创建任务队列，启动 n 个线程处理任务，合并各任务结果得到每列的汇总数据。
void estimatePlan(const Plan& plan, size_t k, size_t threadCount) {
    // 遍历 plan.nodes，提取所有参与 join 的列
    std::set<std::pair<size_t, size_t>> joinKeyColumns; // {tableId, colId}
    for (size_t nodeIndex = 0; nodeIndex < plan.nodes.size(); nodeIndex++) {
        const PlanNode& node = plan.nodes[nodeIndex];
        // 仅处理 JoinNode（仅 JoinNode 才会有 join 键）
        if (std::holds_alternative<JoinNode>(node.data)) {
            const JoinNode& join = std::get<JoinNode>(node.data);
            // 对左侧 join key：取得左子节点中对应的属性信息
            std::pair<size_t, size_t> leftKey = getBaseColumn(plan, join.left, join.left_attr);
            joinKeyColumns.insert(leftKey);
            // 对右侧 join key：取得右子节点中对应的属性信息
            std::pair<size_t, size_t> rightKey = getBaseColumn(plan, join.right, join.right_attr);
            joinKeyColumns.insert(rightKey);
        }
    }

    // 构造任务队列
    std::vector<TopKTask> tasks;
    // 对每个 join key 列生成任务
    for (const auto& key : joinKeyColumns) {
        size_t tableId = key.first;
        size_t colId = key.second;
        assert(tableId < plan.inputs.size());
        const ColumnarTable& table = plan.inputs[tableId];
        assert(colId < table.columns.size());
        const Column& col = table.columns[colId];
        size_t numPages = col.pages.size();
        // 如果页数>= threadCount，生成 threadCount 个任务；否则生成 numPages 个任务
        size_t numTasks = (numPages >= threadCount) ? threadCount : numPages;
        for (size_t i = 0; i < numTasks; i++) {
            TopKTask task;
            task.tableId = tableId;
            task.colId = colId;
            task.startPage = i;
            task.step = numTasks;  // 如果页数足够则 step = threadCount，否则 step = numTasks
            tasks.push_back(task);
        }
    }

    // 用原子索引作为任务队列指针
    std::atomic<size_t> taskIndex(0);

    // 启动 threadCount 个线程
    std::vector<std::thread> threadPool;
    for (size_t i = 0; i < threadCount; i++) {
        threadPool.emplace_back([&tasks, &plan, k, &taskIndex, i]() {

            while (true) {
                size_t idx = taskIndex.fetch_add(1);
                if (idx >= tasks.size()) break;
                TopKTask& task = tasks[idx];
                // 处理此任务：获取对应列，并调用 processColumnPages()
                const ColumnarTable& table = plan.inputs[task.tableId];
                const Column& col = table.columns[task.colId];
                // 调用辅助函数处理从 task.startPage，每隔 task.step 个 page 的采样统计
                //task.output = spaceSavingSmallTopK(col, 0.01f, k, task.startPage, task.step);

                // 按照采样比例选择一个page中的部分元素来统计
                assert(col.type==DataType::INT32);
                assert(start_page<col.pages.size());
                task.output = spaceSavingSmallTopKContinue(col,0.01,k,task.startPage,task.step);
            }
        });
    }
    for (auto& th : threadPool) {
        th.join();
    }

    // 合并同一列任务的输出结果
    std::map<std::pair<size_t, size_t>, std::vector<SpaceSavingSmall::Candidate>> aggregated;
    for (const TopKTask& task : tasks) {
        std::pair<size_t, size_t> key = { task.tableId, task.colId };
        for (const auto& candidate : task.output) {
            // 找到aggregated[key]中匹配的Candidate，合并数据
            auto &vec = aggregated[key];
            bool found = false;
            // 遍历已合并的候选项，查找 key 相同的项
            for (auto &candi : vec) {
                if (candi.key == candidate.key) {
                    candi.count += candidate.count;
                    candi.error += candidate.error;
                    found = true;
                    break;
                }
            }
            // 若未找到，则插入新的候选项
            if (!found) {
                vec.push_back(candidate);
            }
        }
    }

    // 对每个列合并后的候选项排序并输出前 k 个（TopK）
    for (auto& entry : aggregated) {
        size_t tableId = entry.first.first;
        size_t colId = entry.first.second;
        printf("Final TopK for join key (table %zu, column %zu, rows %zu):\n", tableId, colId,plan.inputs[tableId].num_rows);

        std::sort(entry.second.begin(), entry.second.end(), [](const auto &a, const auto &b) {
            return a.count > b.count;
        });
        for (size_t i = 0; i < std::min(entry.second.size(), (size_t) k); i++) {
            printf("  key %d count %d error %d\n", entry.second[i].key, entry.second[i].count, entry.second[i].error);
        }
    }
}

// 递归查找节点node_id的输出属性col_id对应的Scan节点。返回 pair {scan_id, column_id}。
std::pair<size_t, size_t> getBaseScan(const Plan& plan, size_t node_id, size_t col_id) {
    const PlanNode& node = plan.nodes[node_id];
    // 若扫描节点，则直接返回扫描节点中的列信息
    if (std::holds_alternative<ScanNode>(node.data)) {
        return { node_id, col_id };
    }
    // 若为 JoinNode，则按照输出属性的拼接规则：左子节点的输出在前，右子节点的输出紧随其后
    else if (std::holds_alternative<JoinNode>(node.data)) {
        const JoinNode& join = std::get<JoinNode>(node.data);
        size_t input_col_id = std::get<0>(node.output_attrs[col_id]);
        const PlanNode& left_node = plan.nodes[join.left];
        size_t left_count = left_node.output_attrs.size();
        if (input_col_id < left_count) {
            return getBaseScan(plan, join.left, input_col_id);
        } else {
            return getBaseScan(plan, join.right, input_col_id - left_count);
        }
    }
    // 默认返回非法值（理论上不应到达这里）
    return { size_t(-1), size_t(-1) };
}

// 对一个query中各表连接关系的抽象图。
// 这个图必须是无向，无环，联通的图。环会导致两个表之间存在多个连接条件。
// 当没有环时，任意的基表或者中间表之间最多存在一个连接条件。
class JoinGraph{
public:
    // 一个表的一个列
    class TableCol{
    public:
        size_t table_id_;
        size_t col_id_;
        // 实际上我们并不关心一个列是主键还是外键，我们只关心列的最大频率。如果最大频率为1，它就是实际上的主键列。
        // 任何与最大频率为1的列的join，都不会增加结果大小
        size_t max_frequency_;
        std::vector<size_t> final_output_pos_;

        TableCol(size_t base_table_id, size_t col_id)
        : table_id_(base_table_id), col_id_(col_id), max_frequency_(0) {}
    };

    // 一个节点可以代表一个表，也可以代表中间结果
    enum NodeType{JOINED_TABLE, UNJOINED_TABLE, INTERM_TABLE};
    class Node{
    public:
        size_t table_size_;
        std::vector<TableCol> columns_;
        NodeType type_;
        std::vector<PlanNode> plan_tree_;
        size_t plan_root_;

        Node(NodeType type, size_t table_size) : table_size_(table_size), type_(type) { }

        Node(const PlanNode& plan_node, const std::vector<ColumnarTable>& inputs){
            const ScanNode& scan = std::get<ScanNode>(plan_node.data);
            size_t base_table_id = scan.base_table_id;
            table_size_ = inputs[base_table_id].num_rows;
            for(auto [col_id,_]:plan_node.output_attrs){
                columns_.emplace_back(base_table_id,col_id);
            }
            type_ = UNJOINED_TABLE;
            plan_tree_.emplace_back(plan_node);
            plan_root_=0;
        }
    };

    // 一条边代表两个表之间的连接条件
    class Edge{
    public:
        uint32_t t1_, t2_;
        uint32_t c1_, c2_;

        Edge(uint32_t t1, uint32_t t2, uint32_t c1, uint32_t c2)
        : t1_(t1), t2_(t2), c1_(c1), c2_(c2){}

        Edge() : t1_(std::numeric_limits<uint32_t>::max()), t2_(std::numeric_limits<uint32_t>::max()),
        c1_(std::numeric_limits<uint32_t>::max()), c2_(std::numeric_limits<uint32_t>::max()){}

        void reverseOnStartTable(uint32_t start_table_id){
            assert(start_table_id==t1_ || start_table_id==t2_);
            if(start_table_id==t2_){
                uint32_t t = t1_;
                t1_ = t2_;
                t2_ = t;

                uint32_t c = c1_;
                c1_ = c2_;
                c2_ = c;
            }
        }
    };

    // 维护节点集合，Key 为节点 id
    std::vector<Node> nodes_;

    // 邻接表，key 为节点 id，value 为与该节点关联的边集合
    std::unordered_map<uint32_t, std::vector<uint32_t>> adjacency_list_;

    // 存储边集。无向图中每条边仅存储一份，两个表上允许有多个边，但是他们的连接列不能相同
    std::vector<Edge> edges_;


    JoinGraph(const Plan& plan){
        std::unordered_map<size_t, uint32_t> planID_to_graphID; // 记录Plan中的scan节点的id到JoinGraph中基表id的映射
        for(size_t plan_id=0; plan_id<plan.nodes.size(); plan_id++){
            // 若扫描节点，则在JoinGraph中新增一个基表
            if (std::holds_alternative<ScanNode>(plan.nodes[plan_id].data)) {
#ifdef OPTIMIZE_LOG
                printf("add scan node %zu as node %zu\n",plan_id,nodes_.size());
#endif
                planID_to_graphID[plan_id] = nodes_.size();
                nodes_.emplace_back(plan.nodes[plan_id],plan.inputs);
            }
        }
        for(size_t plan_id=0; plan_id<plan.nodes.size(); plan_id++){
            // 若连接节点，则在JoinGraph中新增一个边
            if (std::holds_alternative<JoinNode>(plan.nodes[plan_id].data)) {
                // 先获取连接的左右列对应的scan节点
                const JoinNode& join = std::get<JoinNode>(plan.nodes[plan_id].data);
                auto [left_scan_id, left_col] = getBaseScan(plan,join.left,join.left_attr);
                auto [right_scan_id, right_col] = getBaseScan(plan,join.right,join.right_attr);
                left_scan_id = planID_to_graphID[left_scan_id];
                right_scan_id = planID_to_graphID[right_scan_id];
#ifdef OPTIMIZE_LOG
                printf("add join node %zu on node %zu col %zu and node %zu col %zu\n",
                    plan_id,left_scan_id,left_col,right_scan_id,right_col);
#endif
                // 构造一条边，更新邻接表
                uint32_t eid = edges_.size();
                edges_.emplace_back(left_scan_id,right_scan_id,left_col,right_col);
                adjacency_list_[left_scan_id].push_back(eid);
                adjacency_list_[right_scan_id].push_back(eid);
            }
        }

        // 填充final_outputs
        const PlanNode& root_plan_node = plan.nodes[plan.root];
        for(size_t i=0; i<root_plan_node.output_attrs.size(); i++){
            auto [scan_id, col] = getBaseScan(plan,plan.root,i);
            scan_id = planID_to_graphID[scan_id];
            nodes_[scan_id].columns_[col].final_output_pos_.push_back(i);
#ifdef OPTIMIZE_LOG
            printf("mark node %zu col %zu as final output %zu\n",
                scan_id,col,i+1);
#endif
        }

    }

    inline bool isPrimaryKey(uint32_t nid, uint32_t cid){
        return nodes_[nid].columns_[cid].max_frequency_ == 1;
    }

    inline size_t maxFrequency(uint32_t nid, uint32_t cid){
        return nodes_[nid].columns_[cid].max_frequency_;
    }

    inline size_t tableSize(uint32_t nid){
        return nodes_[nid].table_size_;
    }

    inline std::vector<std::tuple<size_t, DataType>> outputAttr(uint32_t nid){
        return nodes_[nid].plan_tree_[nodes_[nid].plan_root_].output_attrs;
    }

    // 用一个基表做为初始的中间表。返回中间表的id
    uint32_t initJoinNode(uint32_t start_table){
        // 要求start_table必须是未连接的表
        if(nodes_[start_table].type_ != UNJOINED_TABLE){
            throw std::runtime_error("Invalid Join");
        }

        // 构造中间表。直接复制first_table的内容
        nodes_.emplace_back(nodes_[start_table]);
        uint32_t root_node = nodes_.size() - 1;

        // 调整表的类型
        nodes_[start_table].type_ = JOINED_TABLE;
        nodes_[root_node].type_ = INTERM_TABLE;

        // 将原来在start_table的边迁移到root_node上面
        for(auto eid:adjacency_list_[start_table]){
            if(edges_[eid].t1_ == start_table){
                edges_[eid].t1_ = root_node;
            } else {
                edges_[eid].t2_ = root_node;
            }
        }
        adjacency_list_[root_node] = adjacency_list_[start_table];
        adjacency_list_.erase(start_table);
        return root_node;
    }

    // 按照边e，join两个节点，更新节点状态和邻接表。输出结果中间表的id
    uint32_t join(const Edge& e){
        // 要求e的左侧必须是中间表，右侧可以是中间表或者未连接的表，不能是已连接的表
        if(nodes_[e.t1_].type_ != INTERM_TABLE || nodes_[e.t2_].type_ == JOINED_TABLE){
            throw std::runtime_error("Invalid Join");
        }

        // 修改左侧表最大频率。
        uint32_t left_table = e.t1_;
        uint32_t right_table = e.t2_;
        uint32_t left_col = e.c1_;
        uint32_t right_col = e.c2_;
        size_t max_f_left = maxFrequency(left_table,left_col);
        size_t max_f_right = maxFrequency(right_table,right_col);
        bool left_table_small = tableSize(left_table) < tableSize(right_table);
        for(TableCol& column : nodes_[left_table].columns_){
            column.max_frequency_*=max_f_right;
        }
        // 左侧表的基数上限改变
        nodes_[left_table].table_size_ = std::min(nodes_[left_table].table_size_ * max_f_right,
            nodes_[right_table].table_size_ * max_f_left);

        // 修改右侧表最大频率。
        size_t left_col_num = nodes_[left_table].columns_.size();
        for(TableCol column : nodes_[right_table].columns_){
            column.max_frequency_*=max_f_left;
        }

        // 移除left_table与right_table之间的边，检查左侧表的连接键上是否还有其他连接
        std::vector<uint32_t> new_edge_list;
        bool keep_key_col = false;
        for(uint32_t eid : adjacency_list_[left_table]){
            edges_[eid].reverseOnStartTable(left_table);
            if(edges_[eid].t2_!=right_table){
                new_edge_list.push_back(eid);
                if(edges_[eid].c1_==left_col){
                    keep_key_col = true;
                }
            }
        }
        // 移除right_table与left_table之间的边，检查右侧表的连接键上是否还有其他连接
        for(uint32_t eid : adjacency_list_[right_table]){
            edges_[eid].reverseOnStartTable(right_table);
            if(edges_[eid].t2_!=left_table){
                new_edge_list.push_back(eid);
                if(edges_[eid].c1_==right_col){
                    keep_key_col = true;
                }
            }
        }
        // 检查被连接的列是否是最终输出
        auto& left_out = nodes_[left_table].columns_[left_col].final_output_pos_;
        auto& right_out = nodes_[right_table].columns_[right_col].final_output_pos_;
        if(!left_out.empty()){
            keep_key_col = true;
        }
        if(!right_out.empty()){
            keep_key_col = true;
            left_out.insert(left_out.end(),right_out.begin(),right_out.end());
        }

        // 更新连接的列号
        for(uint32_t eid : new_edge_list){
            Edge& edge = edges_[eid];
            if(edge.t1_==left_table && !keep_key_col && edge.c1_>left_col){
                edge.c1_--;
            } else if(edge.t1_==right_table){
                edge.t1_=left_table;
                if(edge.c1_<right_col){
                    edge.c1_ += left_col_num;
                } else if (edge.c1_==right_col){
                    edge.c1_ = left_col;
                } else {
                    edge.c1_ += left_col_num - 1;
                }
                if(!keep_key_col) edge.c1_--;
            }
        }
        adjacency_list_[left_table] = new_edge_list;
        adjacency_list_.erase(right_table);

        // 将右侧表的plan tree的根连接到左侧表plan tree的根上面。具体的join顺序取决于两侧表大小。
        std::vector<PlanNode>& left_plan = nodes_[left_table].plan_tree_;
        std::vector<PlanNode>& right_plan = nodes_[right_table].plan_tree_;
        // 构造一个新的JoinPlanNode
        JoinNode join_node{left_table_small,                    /* build_left */
            nodes_[left_table].plan_root_,                      /* left */
            nodes_[right_table].plan_root_ + left_plan.size(),  /* right */
            left_col,      /* left_attr */
            right_col      /* right_attr */};
        // 对于两表合并后的中间表的列，全部都需要输出，除了被连接的列
        std::vector<std::tuple<size_t, DataType>> output_attrs;
        std::vector<TableCol> table_columns;
        std::vector<std::tuple<size_t, DataType>> left_attrs  = outputAttr(left_table);
        for(size_t i=0; i<left_attrs.size(); i++){
            if(keep_key_col || i!=left_col){
                output_attrs.emplace_back(i,std::get<1>(left_attrs[i]));
                table_columns.emplace_back(nodes_[left_table].columns_[i]);
            }
        }
        std::vector<std::tuple<size_t, DataType>> right_attrs = outputAttr(right_table);
        for(size_t i=0; i<right_attrs.size(); i++){
            if(i!=right_col){
                output_attrs.emplace_back(i+left_attrs.size(),std::get<1>(right_attrs[i]));
                table_columns.emplace_back(nodes_[right_table].columns_[i]);
            }
        }
        nodes_[left_table].columns_ = std::move(table_columns);

        // 将右侧plan_tree的节点加到左侧
        size_t left_plan_size = left_plan.size();
        for(PlanNode plan_node : right_plan){
            std::visit([&](auto&& data) {
                using T = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<T, JoinNode>) {
                    data.left += left_plan_size;
                    data.right += left_plan_size;
                }
            }, plan_node.data);
            left_plan.emplace_back(plan_node);
        }
        left_plan.emplace_back(join_node,output_attrs);
        nodes_[left_table].plan_root_ = left_plan.size() - 1;

        return left_table;
    }

    uint32_t join(uint32_t left, uint32_t right){
        uint32_t join = joinPartnersFor(left,right);
        Edge e = edges_[join];
        e.reverseOnStartTable(left);
        return this->join(e);
    }

    // 尽可能深的将所有主键连接应用到root_node上面（root_node不在pkfk_tables当中）
    uint32_t joinAllDeepPk(uint32_t root_node, std::set<uint32_t>& pkfk_tables){
        if(nodes_[root_node].type_==UNJOINED_TABLE){
            if(joinPartnersFor(root_node,pkfk_tables,true,true).empty()){
                return root_node;
            }else{
                root_node = initJoinNode(root_node);
            }
        }
        for(uint32_t pk_join : joinPartnersFor(root_node,pkfk_tables,true,true)){
            Edge e = edges_[pk_join];
            e.reverseOnStartTable(root_node);
            pkfk_tables.erase(e.t2_);
            uint32_t pk_node = joinAllDeepPk(e.t2_,pkfk_tables);
            root_node = join(root_node,pk_node);
        }
        return root_node;
    }

    // 连接所有与root_node直接相连的pk（root_node不在pkfk_tables当中）
    uint32_t joinAllShallowPk(uint32_t root_node, std::set<uint32_t>& pkfk_tables){
        if(nodes_[root_node].type_==UNJOINED_TABLE){
            if(joinPartnersFor(root_node,pkfk_tables,true,true).empty()){
                return root_node;
            }else{
                root_node = initJoinNode(root_node);
            }
        }
        for(uint32_t pk_join : joinPartnersFor(root_node,pkfk_tables,true,true)){
            Edge e = edges_[pk_join];
            e.reverseOnStartTable(root_node);
            pkfk_tables.erase(e.t2_);
            root_node = join(root_node,e.t2_);
        }
        return root_node;
    }

    // 这个函数只应当在初始时调用。将基表分为两部分：
    // 一种是参与了任意n-m join的表id。
    // 另一种是其补集。
    std::pair<std::set<uint32_t>, std::set<uint32_t>> divideTables(){
        std::set<uint32_t> nm_tables;
        for(auto & edge : edges_){
            // 检查这条边是否为n-m join。是的话将两侧的表都加入到结果中
            if(!isPrimaryKey(edge.t1_,edge.c1_) && !isPrimaryKey(edge.t2_,edge.c2_)){
                nm_tables.insert(edge.t1_);
                nm_tables.insert(edge.t2_);
            }
        }

        std::set<uint32_t> pkfk_tables;
        for(uint32_t nid=0; nid<nodes_.size(); nid++){
            if(nm_tables.find(nid) == nm_tables.end()){
                pkfk_tables.insert(nid);
            }
        }
        return std::make_pair(nm_tables, pkfk_tables);
    }

    // 给定一个table，获取other_tables与table的连接
    uint32_t joinPartnersFor(uint32_t table, uint32_t other_tables){
        // 检查table的所有连接（假设他们都是有效的）
        for(uint32_t eid : adjacency_list_[table]){
            Edge e = edges_[eid];
            e.reverseOnStartTable(table);        // 强制将table设为起始表
            if(e.t2_==other_tables){          // 要求必须在other_tables中
                return eid;
            }
        }
        throw std::runtime_error("No join between two node");
    }

    // 给定一个table，获取other_tables中与table的连接，并按照基数从小到大排列
    std::vector<uint32_t> joinPartnersFor(uint32_t table, const std::set<uint32_t> & other_tables, bool pk_join=false, bool sort=false){
        std::vector<uint32_t> joins;

        // 检查table的所有连接（假设他们都是有效的）
        for(uint32_t eid : adjacency_list_[table]){
            Edge e = edges_[eid];
            e.reverseOnStartTable(table);        // 强制将table设为起始表
            // 要求必须在other_tables中
            if(other_tables.find(e.t2_)!=other_tables.end() && (!pk_join || isPrimaryKey(e.t2_,e.c2_))){
                joins.emplace_back(eid);
            }
        }

        if(sort){
            std::sort(joins.begin(),joins.end(),[this, &table](uint32_t a, uint32_t b) {
                return tableSize(edges_[a].t1_==table ? edges_[a].t2_ : edges_[a].t1_)
                     < tableSize(edges_[b].t1_==table ? edges_[b].t2_ : edges_[b].t1_);  // 按从小到大排序
            });
        }

        return joins;
    }

    Plan getFinalPlan(uint32_t root_node_id){
        Node& root_node = nodes_[root_node_id];
        Plan final_plan;
        final_plan.nodes = root_node.plan_tree_;
        final_plan.root = root_node.plan_root_;

        // 调整输出的顺序
        std::vector<std::tuple<size_t, DataType>> output_attrs;
        for(size_t i=0; i<root_node.columns_.size(); i++){
            for(size_t out_pos : root_node.columns_[i].final_output_pos_){
                if(output_attrs.size()<out_pos+1){
                    output_attrs.resize(out_pos+1);
                }
                output_attrs[out_pos] = root_node.plan_tree_[root_node.plan_root_].output_attrs[i];
            }
        }
        final_plan.nodes[final_plan.root].output_attrs = output_attrs;

        return final_plan;
    }
};


Plan UESJoinOrderOptimize(JoinGraph& join_graph){
    std::unordered_map<uint32_t, size_t> upper_bounds;  // 记录了表的上限
    uint32_t tree_root = std::numeric_limits<uint32_t>::max();     // join_graph的根节点

    auto [nm_tables, pkfk_tables] = join_graph.divideTables();
#ifdef OPTIMIZE_LOG
    printf("n-m tables: ");
    for(uint32_t nm_table:nm_tables){
        printf("%d, ",nm_table);
    }
    printf("\npk-fk tables: ");
    for(uint32_t pkfk_table:pkfk_tables){
        printf("%d, ",pkfk_table);
    }
    printf("\n");
#endif
    bool init = true;
    while (!nm_tables.empty()) {
#ifdef OPTIMIZE_LOG
        printf("\n optimize loop start\n");
#endif
        // 更新计算nm_table在应用所有可能的pkfk_join之后的基数上限
        for(uint32_t nid : nm_tables){
            size_t min_table_size = join_graph.tableSize(nid);
            for(uint32_t join_id : join_graph.joinPartnersFor(nid,pkfk_tables)){
                JoinGraph::Edge join = join_graph.edges_[join_id];
                join.reverseOnStartTable(nid);
                if(join_graph.isPrimaryKey(join.t2_, join.c2_)){
                    size_t estimate = join_graph.tableSize(join.t2_) * join_graph.maxFrequency(nid,join.c1_);
                    min_table_size = std::min(min_table_size, estimate);
#ifdef OPTIMIZE_LOG
                    printf("n-m table %u join pk table %u col %u estimate size %zu\n",
                        nid, join.t2_, join.c2_, estimate);
#endif
                }
            }
            upper_bounds[nid] = min_table_size;
#ifdef OPTIMIZE_LOG
            printf("n-m table %u upper bound %zu\n",nid, min_table_size);
#endif
        }

        // 初始化join 树
        if(init){
            // 以upper_bounds最小的表作为起点
            size_t lowest_bound = std::numeric_limits<size_t>::max();               // 目前的最小的基数
            uint32_t lowest_bound_table = std::numeric_limits<uint32_t>::max();     // 最小基数的nm表
            for(auto [table, bound] : upper_bounds){
                if(bound < lowest_bound){
                    lowest_bound = bound;
                    lowest_bound_table = table;
                }
            }
            tree_root = join_graph.initJoinNode(lowest_bound_table);
#ifdef OPTIMIZE_LOG
            printf("choose init n-m table %u as intermediate table %u\n", lowest_bound_table, tree_root);
#endif

            // 应用所有lowest_bound_table（现在的tree_root）的pk-fk join，基数最小的优先
            tree_root = join_graph.joinAllDeepPk(tree_root,pkfk_tables);

            //从nm_tables中移除该表
            nm_tables.erase(lowest_bound_table);
            upper_bounds[tree_root] = lowest_bound;
            init = false;
            continue;
        }

        size_t best_upper = std::numeric_limits<size_t>::max();
        JoinGraph::Edge next_nm_join;
        // 确定下一个最小的nm_table
        for(uint32_t join_id : join_graph.joinPartnersFor(tree_root,nm_tables)){
            JoinGraph::Edge nm_join = join_graph.edges_[join_id];
            nm_join.reverseOnStartTable(tree_root);
            size_t current_upper = std::min(upper_bounds[nm_join.t1_]*join_graph.maxFrequency(nm_join.t2_,nm_join.c2_),
                upper_bounds[nm_join.t2_]*join_graph.maxFrequency(nm_join.t1_,nm_join.c1_));
            if(current_upper < best_upper){
                best_upper = current_upper;
                next_nm_join = nm_join;
            }
        }
        uint32_t next_nm_table = next_nm_join.t2_;

        // 连接下一个nm_table
        if(upper_bounds[next_nm_table] < join_graph.tableSize(next_nm_table)){
            // 如果next_nm_table上的pk-fk连接能减少其大小，那么首先应用pk-fk连接，构造一颗子树
            size_t sub_tree_root = join_graph.initJoinNode(next_nm_table);
#ifdef OPTIMIZE_LOG
            printf("choose next n-m table %u as intermediate table %zu\n", next_nm_table, sub_tree_root);
#endif

            // 应用所有next_nm_table的pk-fk join，基数最小的优先
            //sub_tree_root = join_graph.joinAllDeepPk(sub_tree_root,pkfk_tables);
            sub_tree_root = join_graph.joinAllShallowPk(sub_tree_root,pkfk_tables);

            // 将子树连接到主树上。注意要更新next_nm_join，不要用之前的
            tree_root = join_graph.join(tree_root,sub_tree_root);
        } else {
#ifdef OPTIMIZE_LOG
            printf("choose next n-m table %u col %u and join on intermediate table %u col %u\n",
                next_nm_join.t2_, next_nm_join.c2_, next_nm_join.t1_, next_nm_join.c1_);
#endif
            tree_root = join_graph.join(next_nm_join);
            // 应用所有next_nm_table的pk-fk join，基数最小的优先
            //tree_root = join_graph.joinAllDeepPk(tree_root,pkfk_tables);
            tree_root = join_graph.joinAllShallowPk(tree_root,pkfk_tables);

        }
        nm_tables.erase(next_nm_table);
        upper_bounds[tree_root] = best_upper;
    }

    // 处理没有处理完的pkfk_tables。现在nm_tables一定处理完了。
    while (!pkfk_tables.empty()) {
#ifdef OPTIMIZE_LOG
        printf("\nprocess unjoined pkfk tables\n");
#endif
        if(init){
            // 如果还没有初始化，那很奇怪了，说明所有的连接都是主键连接。这种情况我们不优化了，相信pg
#ifdef OPTIMIZE_LOG
            printf("cannot determine root, stop\n");
#endif
            return {};
        }

        // 应用所有的pk join，基数最小的优先
        tree_root = join_graph.joinAllDeepPk(tree_root,pkfk_tables);


        // 应用所有的fk join，基数最小的优先
        for(uint32_t join_id : join_graph.joinPartnersFor(tree_root,pkfk_tables,false,true)){
            JoinGraph::Edge join = join_graph.edges_[join_id];
            join.reverseOnStartTable(tree_root);
#ifdef OPTIMIZE_LOG
            printf("join table %u col %u on intermediate table %u col %u\n", join.t2_, join.c2_, join.t1_, join.c1_);
#endif
            tree_root = join_graph.join(join);
            pkfk_tables.erase(join.t2_);
        }
    }

    // new_plan.inputs 尚未填充
    return join_graph.getFinalPlan(tree_root);
}


// 抽取page中连续的元素来统计，并且在可能为主键列时扩大抽样步长，减少抽样次数
// 获取一列的非空值的最大频数
size_t maxFrequency(const Column& col, size_t row_num, float sample_rate, size_t k, size_t start_page=0, size_t page_step=1){
    // 按照采样比例选择一个page中的部分元素来统计
    assert(col.type==DataType::INT32);
    assert(start_page<col.pages.size());
    SpaceSavingSmall space_saving(k);
    int maybe_primary_key = 0;
    for(size_t p=start_page;p<col.pages.size();p+=page_step){
        const Page* page = col.pages[p];
        //size_t page_rows = getRowCount(page);
        size_t page_nonnull_rows = getNonNullCount(page);
        size_t sample_num = (size_t)(page_nonnull_rows * sample_rate);
        if(sample_num==0) sample_num=1;       // 至少要采样一行。
        if(sample_num>page_nonnull_rows) sample_num=page_nonnull_rows;
        // 采样连续的k行（不超过page_nonnull_rows），不采样空值。
        size_t pos = rand() % (page_nonnull_rows - sample_num + 1);
        const int32_t* base = getPageData<int32_t>(page) + pos;
        for(size_t j = 0; j<sample_num; j++){
            space_saving.process(base[j]);
        }

        if(maybe_primary_key>=0){
            maybe_primary_key ++;
            // 判断有没有可能不是主键列了
            if(maybe_primary_key % 4==0){
                if(space_saving.allDistinct()){
                    page_step *= 2;             // 如果还有可能是主键列，那么每4次步长翻倍
                    //printf("page_step %zu\n",page_step);
                } else {
                    maybe_primary_key = -1;     // 该列不可能是主键，以后都不用再判断了
                }
            }
        }
    }

    auto topk = space_saving.getSortedCandidates();
    float max_f =  (float)(topk.front().count-topk.front().error) / space_saving.getTotal();
    //    for(auto [key,statistic]:topk){
    //        printf("key %d count %d error %d\n",key,statistic.count,statistic.error);
    //    }
#ifdef OPTIMIZE_LOG
    if(maybe_primary_key > 0){
        printf("top 1 count=%d error=%d rate=%.3f maybe pk\n",
            topk.front().count, topk.front().error, max_f);
    } else {
        printf("top 1 count=%d error=%d rate=%.3f mustbe fk\n",
            topk.front().count, topk.front().error, max_f);
    }
#endif

    if(maybe_primary_key > 0){
        return 1;
    } else {
        return std::ceil(max_f * row_num);
    }
}


void estimateJoinGraph(JoinGraph& join_graph, const std::vector<ColumnarTable>& inputs, size_t k) {
    // 遍历JoinGraph中的所有边，也就是join，对参与join的列做估计
    for(JoinGraph::Edge join : join_graph.edges_){
        // 估计左侧列
        JoinGraph::TableCol& left_col = join_graph.nodes_[join.t1_].columns_[join.c1_];
        if(left_col.max_frequency_==0){
            const ColumnarTable& table = inputs[left_col.table_id_];
            const Column& col = table.columns[left_col.col_id_];
            left_col.max_frequency_ = maxFrequency(col, table.num_rows, 0.01, k);
#ifdef OPTIMIZE_LOG
            printf("Max Frequency for join key (table %zu, column %zu, row num %zu) is %zu\n",
                left_col.table_id_, left_col.col_id_, table.num_rows, left_col.max_frequency_);
#endif
        }
        // 估计右侧列
        JoinGraph::TableCol& right_col = join_graph.nodes_[join.t2_].columns_[join.c2_];
        if(right_col.max_frequency_==0){
            const ColumnarTable& table = inputs[right_col.table_id_];
            const Column& col = table.columns[right_col.col_id_];
            right_col.max_frequency_ = maxFrequency(col, table.num_rows, 0.01, k);
#ifdef OPTIMIZE_LOG
            printf("Max Frequency for join key (table %zu, column %zu, row num %zu) is %zu\n",
                right_col.table_id_, right_col.col_id_, table.num_rows, right_col.max_frequency_);
#endif
        }
    }
}


Plan optimizePlan(const Plan& plan){
    JoinGraph join_graph(plan);
    estimateJoinGraph(join_graph,plan.inputs,5);
    Plan new_plan = UESJoinOrderOptimize(join_graph);
    //new_plan.inputs = plan.inputs;
    if(new_plan.nodes.empty()){
        new_plan.nodes=plan.nodes;
        new_plan.root=plan.root;
    }
    return new_plan;
}

}