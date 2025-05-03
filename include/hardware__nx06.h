// Hardware information for Intel Xeon E7-4880 v2 node sidon.

// Architecture from `uname -srm`.
#define SPC__X86_64

// CPU from `/proc/cpuinfo`.
#define SPC__CPU_NAME "Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz"

// The servers might have multiple CPUs. We limit all benchmarks to a single node using numactl. The listed CPU numbers
// below are for a single CPU. The listed NUMA numbers are just meant to give you a rough idea of the system.
#define SPC__CORE_COUNT 32
#define SPC__THREAD_COUNT 64
#define SPC__NUMA_NODE_COUNT 2
#define SPC__NUMA_NODES_ACTIVE_IN_BENCHMARK 1

// Main memory per NUMA node (MB).
#define SPC__NUMA_NODE_DRAM_MB 257264

// Obtained from `lsb_release -a`.
#define SPC__OS "Ubuntu 24.04.2 LTS"

// Obtained from: `uname -srm`.
#define SPC__KERNEL "Linux 5.4.0-122-generic x86_64"

// Intel: possible options are AVX, AVX2, and AVX512. No Intel CPU older than Intel Xeon E7-4880 v2 will be used.
#define SPC__SUPPORTS_AVX
#define SPC__SUPPORTS_AVX2
#define SPC__SUPPORTS_AVX512

// Cache information from `getconf -a | grep CACHE`.
#define SPC__LEVEL1_ICACHE_SIZE                 32768
#define SPC__LEVEL1_ICACHE_ASSOC                8
#define SPC__LEVEL1_ICACHE_LINESIZE             64
#define SPC__LEVEL1_DCACHE_SIZE                 49152
#define SPC__LEVEL1_DCACHE_ASSOC                12
#define SPC__LEVEL1_DCACHE_LINESIZE             64
#define SPC__LEVEL2_CACHE_SIZE                  1310720
#define SPC__LEVEL2_CACHE_ASSOC                 20
#define SPC__LEVEL2_CACHE_LINESIZE              64
#define SPC__LEVEL3_CACHE_SIZE                  50331648
#define SPC__LEVEL3_CACHE_ASSOC                 12
#define SPC__LEVEL3_CACHE_LINESIZE              64
#define SPC__LEVEL4_CACHE_SIZE                  0
#define SPC__LEVEL4_CACHE_ASSOC                 0
#define SPC__LEVEL4_CACHE_LINESIZE              0
