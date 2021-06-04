#include <cuda_runtime.h>
#include <omp.h>
#include "common.h"

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) C[tid] = A[tid] + B[tid];
}

int main(int argc, char **argv) {
    int nElem = 1 << 28;
    if (argc > 1) nElem = 1 << atoi(argv[1]);
    size_t nBytes = nElem * sizeof(float);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    printf("Vector size: %d\n\n", nElem);

    // malloc host and unified memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef, nBytes));

    initialData(A, nElem);
    initialData(B, nElem);

    sumArraysOnHost(A, B, hostRef, nElem);

    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);

    // warmup
    sumArraysOnGPU<<<grid, block>>>(A, B, gpuRef, nElem);
    CHECK(cudaDeviceSynchronize());

    double dtime = - omp_get_wtime();
    sumArraysOnGPU<<<grid, block>>>(A, B, gpuRef, nElem);
    CHECK(cudaDeviceSynchronize());
    dtime += omp_get_wtime();
    printf("\"sumArraysOnGPU\" with <<<grid %d, block %d>>>\n", grid.x, block.x);
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, COST * nElem / dtime / 1.0e+9);
    checkResult(hostRef, gpuRef, nElem);

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    CHECK(cudaDeviceReset());

    return 0;
}