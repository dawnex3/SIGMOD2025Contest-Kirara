// Hardware information for IBM Power10 node cp03.

// Architecture from `uname -srm`.
#define SPC__PPC64LE

// CPU from `/proc/cpuinfo`.
#define SPC__CPU_NAME "POWER10 (architected), altivec supported"

// The servers might have multiple CPUs. We limit all benchmarks to a single node using numactl. The listed CPU numbers
// below are for a single CPU. The listed NUMA numbers are just meant to give you a rough idea of the system.
#define SPC__CORE_COUNT 12
#define SPC__THREAD_COUNT 48
#define SPC__NUMA_NODE_COUNT 4
#define SPC__NUMA_NODES_ACTIVE_IN_BENCHMARK 1

// Main memory per NUMA node (MB).
#define SPC__NUMA_NODE_DRAM_MB 256931

// Obtained from `lsb_release -a`.
#define SPC__OS "Ubuntu 24.04 LTS"

// Obtained from: `uname -srm`.
#define SPC__KERNEL "Linux 5.15.0-107-generic ppc64le"

// IBM: possible options are VSX, VMX, and MMA. No IBM CPU older than Power8 will be used.
#define SPC__SUPPORTS_VSX
#define SPC__SUPPORTS_VMX
#define SPC__SUPPORTS_MMA

// Cache information from `getconf -a | grep CACHE`.
#define SPC__LEVEL1_ICACHE_SIZE                 49152
#define SPC__LEVEL1_ICACHE_ASSOC                6
#define SPC__LEVEL1_ICACHE_LINESIZE             128
#define SPC__LEVEL1_DCACHE_SIZE                 32768
#define SPC__LEVEL1_DCACHE_ASSOC                8
#define SPC__LEVEL1_DCACHE_LINESIZE             128
#define SPC__LEVEL2_CACHE_SIZE                  1048576
#define SPC__LEVEL2_CACHE_ASSOC                 8
#define SPC__LEVEL2_CACHE_LINESIZE              128
#define SPC__LEVEL3_CACHE_SIZE                  4194304
#define SPC__LEVEL3_CACHE_ASSOC                 16
#define SPC__LEVEL3_CACHE_LINESIZE              128
#define SPC__LEVEL4_CACHE_SIZE                  0
#define SPC__LEVEL4_CACHE_ASSOC                 0
#define SPC__LEVEL4_CACHE_LINESIZE              0
