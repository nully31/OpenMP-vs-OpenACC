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

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);

    // warmup
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());

    double dtime = - omp_get_wtime();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    dtime += omp_get_wtime();
    printf("\"sumArraysOnGPU\" with <<<grid %d, block %d>>>\n", grid.x, block.x);
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, COST * nElem / dtime / 1.0e+9);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    CHECK(cudaDeviceReset());

    return 0;
}