#include <miner.h>
#include "argon2ref/argon2.h"
#include "argon2d_kernel.h"
#include "cuda_helper.h"
#include <cuda_runtime.h>

#define NBN 2

static const size_t INPUT_BYTES = 80;
static const size_t OUTPUT_BYTES = 32;
static const unsigned int DEFAULT_ARGON2_FLAG = 2;

static uint32_t *d_resNonces[MAX_GPUS];
static uint32_t throughputs[MAX_GPUS] = {0};
static bool init[MAX_GPUS] = {0};
uint8_t* memory[MAX_GPUS];


void argon2d_dyn_hash( void *output, const void *input )
{
    argon2_context context;
    context.out = (uint8_t *)output;
    context.outlen = (uint32_t)OUTPUT_BYTES;
    context.pwd = (uint8_t *)input;
    context.pwdlen = (uint32_t)INPUT_BYTES;
    context.salt = (uint8_t *)input; //salt = input
    context.saltlen = (uint32_t)INPUT_BYTES;
    context.secret = NULL;
    context.secretlen = 0;
    context.ad = NULL;
    context.adlen = 0;
    context.allocate_cbk = NULL;
    context.free_cbk = NULL;
    context.flags = DEFAULT_ARGON2_FLAG; // = ARGON2_DEFAULT_FLAGS
    // main configurable Argon2 hash parameters
    context.m_cost = 500;  // Memory in KiB (512KB)
    context.lanes = 8;     // Degree of Parallelism
    context.threads = 1;   // Threads
    context.t_cost = 2;    // Iterations
    context.version = ARGON2_VERSION_10;

    argon2_ctx( &context, Argon2_d );
}

__host__
void ar_set_throughput(int thr_id){
    int avail_mem = cuda_available_memory(thr_id);
    uint32_t throughput = (avail_mem * 1024 * 0.75) / ALGO_TOTAL_BLOCKS;
    throughput = cuda_default_throughput(thr_id, throughput);
    throughput = (throughput / 16) * 16;

    throughputs[thr_id] = throughput;
}

__host__
void argon2d_init(int thr_id){

    size_t mem_size = (size_t)throughputs[thr_id] * ALGO_TOTAL_BLOCKS * ARGON2_BLOCK_SIZE;

    gpulog(LOG_INFO, thr_id,
            "batchsize: %u, trying to allocate %u MB of memory",
            throughputs[thr_id],  mem_size / (1024 * 1024));

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_resNonces[thr_id], NBN * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &memory[thr_id], mem_size));

}


__host__ void argon2d_hash_cuda(int thr_id, uint32_t throughput, uint32_t startNonce, uint32_t target,uint32_t* resNonces){

    struct block_g *memory_blocks=(struct block_g *)memory[thr_id];
    const dim3 blocks = dim3(1, 1, throughput);
    const dim3 th_1 = dim3(16, 16, 1);
    const dim3 th_2 = dim3(THREADS_PER_LANE, ALGO_LANES, 1);
    const dim3 th_3 = dim3(4, 16, 1);


    CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)));

    argon2_initialize<<<throughput/16, th_1>>>((block*) memory[thr_id],startNonce);

    argon2_fill<<<blocks, th_2>>>(memory_blocks, ALGO_PASSES, ALGO_LANES, ALGO_SEGMENT_BLOCKS);

    argon2_finalize<<<throughput/16, th_3, 16 * 258 * sizeof(uint32_t)>>>((block*) memory[thr_id], startNonce, target, d_resNonces[thr_id]);

    cudaThreadSynchronize();

    CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (resNonces[0] == resNonces[1]) {
        resNonces[1] = UINT32_MAX;
    }

}

int scanhash_argon2d( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    uint32_t _ALIGN(64) endiandata[20];
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t throughput = 0;

    if (!init[thr_id])
    {
        cudaSetDevice(device_map[thr_id]);
        if (opt_cudaschedule == -1 && gpu_threads == 1) {
            cudaDeviceReset();
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            CUDA_LOG_ERROR();
        }

        ar_set_throughput(thr_id);

        argon2d_init(thr_id);

        init[thr_id] = true;
    }

    throughput = throughputs[thr_id];

    for (int k=0; k < 20; k++)
        be32enc(&endiandata[k], pdata[k]);

    set_data(endiandata);

    do {

        argon2d_hash_cuda(thr_id, throughput, pdata[19], ptarget[7], work->nonces);

        *hashes_done = pdata[19] - first_nonce + throughput;

        pdata[19] += throughput;

        if (work->nonces[0] != UINT32_MAX)
        {

            uint32_t _ALIGN(64) vhash[8];
            const uint32_t Htarg = ptarget[7];
            be32enc(&endiandata[19], work->nonces[0]);
            argon2d_dyn_hash( vhash, endiandata );

            if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);

                if (work->nonces[1] != UINT32_MAX) {
                    be32enc(&endiandata[19], work->nonces[1]);
                    argon2d_dyn_hash(vhash, endiandata);
                    if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                        bn_set_target_ratio(work, vhash, 1);
                        work->valid_nonces++;
                    }
                }

                return work->valid_nonces;

            }

        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
            pdata[19] = max_nonce;
            break;
        }

    } while (!work_restart[thr_id].restart && !abort_flag);

    *hashes_done = pdata[19] - first_nonce;
    return 0;

}

extern "C" void free_argon2d(int thr_id)
{
    if (!init[thr_id])
        return;

    cudaThreadSynchronize();

    cudaFree(memory[thr_id]);

    cudaFree(d_resNonces[thr_id]);

    init[thr_id] = false;

    cudaDeviceSynchronize();

    cudaDeviceReset();
}
