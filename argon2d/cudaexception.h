#ifndef ARGON2_CUDA_CUDAEXCEPTION_H
#define ARGON2_CUDA_CUDAEXCEPTION_H


#define TRY(call)                                                            \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        fprintf(stderr, "Error: %s:%d \n", __FILE__, __LINE__);              \
        fprintf(stderr, "code: %d, reason: %s \n", error,                    \
                cudaGetErrorString(error));                                  \
        exit(1);                                                             \
    }                                                                        \
}


#endif
