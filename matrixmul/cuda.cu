#include <cuda_runtime.h>
#include <omp.h>
#include "matrixmul.h"
#include "cublas_v2.h"

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
    int nxy = nElem * nElem;
    size_t nBytes = nxy * sizeof(float);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    printf("Matrix size: %d * %d\n\n", nElem, nElem);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    call_cblas_sgemm(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    if (argc > 3) {
        block.x = atoi(argv[2]);
        block.y = atoi(argv[3]);
    }
    dim3 grid((nElem + block.x - 1) / block.x, (nElem + block.y - 1) / block.y);

    // warmup
    mulMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());

    double dtime = - omp_get_wtime();
    mulMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    dtime += omp_get_wtime();
    printf("\"mulMatrixOnGPU\" with <<<grid (%d, %d), block (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);

    // cuBLAS
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(nElem, nElem, sizeof(*h_A), h_A, nElem, d_A, nElem);
    stat = cublasSetMatrix(nElem, nElem, sizeof(*h_B), h_B, nElem, d_B, nElem);
    CHECK(cudaMemset(d_A, 0, nBytes));

    float alpha = 1.0f;
    float beta = 1.0f;

    dtime = - omp_get_wtime();
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nElem, nElem, nElem, &alpha, d_A, nElem, d_B, nElem, &beta, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    dtime += omp_get_wtime();
    printf("\"cublasSgemm\"\n");
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));

    stat = cublasGetMatrix(nElem, nElem, sizeof(*gpuRef), d_C, nElem, gpuRef, nElem);
    checkResult(hostRef, gpuRef, nxy);


    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    cublasDestroy(handle);

    CHECK(cudaDeviceReset());

    return 0;
}