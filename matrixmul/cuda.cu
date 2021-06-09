#include <cuda_runtime.h>
#include <omp.h>
#include "../common.h"

__global__ void mulMatrixOnGPU(float *A, float *B, float *C, const int N) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * N + ix;

    if (ix < N && iy < N) {
        float temp = 0.0;
        for (int i = 0; i < N; i++) {
            temp += A[iy * N + i] * B[i * N + ix];
        }
        C[idx] = temp;
    }
}

int main(int argc, char **argv) {
    int nElem = 1 << 10;
    if (argc > 1) nElem = 1 << atoi(argv[1]);
    size_t nBytes = nElem * nElem * sizeof(float);

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

    if (nElem < (1 << 13)) mulMatrixOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid((nElem + block.x - 1) / block.x);

    // warmup
    mulMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());

    double dtime = - omp_get_wtime();
    mulMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    dtime += omp_get_wtime();
    printf("\"mulMatrixOnGPU\" with <<<grid (%d, %d), block (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
    printf("Elapsed time: %.3f msec\n\n", dtime / 1.0e-3);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (nElem < (1 << 13)) checkResult(hostRef, gpuRef, nElem * nElem);

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