// Host-only wrapper around NVIDIA's standalone occupancy calculator.
// Build with the include directory of CUDA Runtime 13.0.96; no CUDA driver is loaded.
#include <cuda_occupancy.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::fprintf(stderr, "Usage: occupancy_check SM REGISTERS STATIC_SHARED_BYTES BARRIERS\n");
        return 2;
    }
    const int sm = std::atoi(argv[1]);
    cudaOccDeviceProp device;
    device.computeMajor = sm / 10;
    device.computeMinor = sm % 10;
    device.maxThreadsPerBlock = 1024;
    device.regsPerBlock = 65536;
    device.regsPerMultiprocessor = 65536;
    device.warpSize = 32;
    device.numSms = 1;  // The result is per SM.
    device.sharedMemPerBlock = 48 * 1024;
    device.reservedSharedMemPerBlock = sm >= 80 ? 1024 : 0;
    // NVIDIA Programming Guide, Table 30 and Table 31, accessed 2026-10-07.
    // SM 8.8 is compiler-supported but has no published physical-limit row.
    switch (sm) {
    case 53:
    case 62:
        device.regsPerBlock = 32768;
        [[fallthrough]];
    case 50:
    case 60:
        device.maxThreadsPerMultiprocessor = 2048;
        device.sharedMemPerMultiprocessor = 64 * 1024;
        break;
    case 52:
    case 61:
    case 70:
    case 72:
        device.maxThreadsPerMultiprocessor = 2048;
        device.sharedMemPerMultiprocessor = 96 * 1024;
        break;
    case 75:
        device.maxThreadsPerMultiprocessor = 1024;
        device.sharedMemPerMultiprocessor = 64 * 1024;
        break;
    case 80:
        device.maxThreadsPerMultiprocessor = 2048;
        device.sharedMemPerMultiprocessor = 164 * 1024;
        break;
    case 87:
        device.maxThreadsPerMultiprocessor = 1536;
        device.sharedMemPerMultiprocessor = 164 * 1024;
        break;
    case 86:
    case 89:
    case 120:
    case 121:
        device.maxThreadsPerMultiprocessor = 1536;
        device.sharedMemPerMultiprocessor = 100 * 1024;
        break;
    case 90:
    case 100:
    case 103:
        device.maxThreadsPerMultiprocessor = 2048;
        device.sharedMemPerMultiprocessor = 228 * 1024;
        break;
    case 110:
        device.maxThreadsPerMultiprocessor = 1536;
        device.sharedMemPerMultiprocessor = 228 * 1024;
        break;
    default:
        std::fprintf(stderr, "No verified physical-limit specification for SM %d\n", sm);
        return 2;
    }
    device.sharedMemPerBlockOption = sm < 70 ? 48 * 1024
                                          : device.sharedMemPerMultiprocessor - device.reservedSharedMemPerBlock;
    cudaOccFuncAttributes function;
    function.maxThreadsPerBlock = 256;
    function.numRegs = std::atoi(argv[2]);
    function.sharedSizeBytes = std::strtoull(argv[3], nullptr, 10);
    function.numBlockBarriers = std::atoi(argv[4]);
    cudaOccDeviceState state;
    cudaOccResult result = {};
    const cudaOccError error = cudaOccMaxActiveBlocksPerMultiprocessor(&result, &device, &function, &state, 256, 0);
    if (error != CUDA_OCC_SUCCESS) {
        std::fprintf(stderr, "NVIDIA occupancy calculator returned %d\n", int(error));
        return 1;
    }
    int granularity = 0;
    int partitions = 0;
    cudaOccRegAllocationGranularity(&granularity, &device);
    cudaOccSubPartitionsPerMultiprocessor(&partitions, &device);
    std::printf(
        "{\"sm\":%d,\"block_size\":256,\"registers\":%d,\"max_warps_per_sm\":%d,"
        "\"active_blocks\":%d,\"active_warps\":%d,\"occupancy_pct\":%.9g,"
        "\"registers_per_warp_granularity\":%d,\"register_partitions\":%d,"
        "\"registers_allocated_per_block\":%d,\"shared_allocated_per_block\":%zu,"
        "\"limits\":{\"registers\":%d,\"shared\":%d,\"warps\":%d,\"blocks\":%d,\"barriers\":%d}}\n",
        sm, function.numRegs, device.maxThreadsPerMultiprocessor / 32,
        result.activeBlocksPerMultiprocessor, result.activeBlocksPerMultiprocessor * 8,
        result.activeBlocksPerMultiprocessor * 25600.0 / device.maxThreadsPerMultiprocessor,
        granularity, partitions, result.allocatedRegistersPerBlock, result.allocatedSharedMemPerBlock,
        result.blockLimitRegs, result.blockLimitSharedMem, result.blockLimitWarps,
        result.blockLimitBlocks, result.blockLimitBarriers
    );
}
