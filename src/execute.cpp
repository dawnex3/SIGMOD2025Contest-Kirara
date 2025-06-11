#include <hardware.h>
#include <plan.h>
#include <table.h>

#include "morsel_lite/execution_context.hpp"
#include "morsel_lite/scheduler.hpp"

namespace Contest {

using ExecuteResult = std::vector<std::vector<Data>>;
using namespace std::chrono;

class Runtime {
public:
    MorselLite::ExecutionContext ctx;
    MorselLite::Scheduler scheduler;

    Runtime(size_t num_threads)
        : scheduler(ctx, num_threads) { }
};

ColumnarTable execute(const Plan& plan, void* context) {
    Runtime *rt = static_cast<Runtime *>(context);
    rt->scheduler.set_plan(plan, 10000);
    rt->scheduler.run();
    return rt->ctx.get_result_table(rt->scheduler.output_data_types());
}

void* build_context() {
    return new Runtime(24);
}

void destroy_context(void* context) {
    delete static_cast<Runtime *>(context);
}

} // namespace Contest