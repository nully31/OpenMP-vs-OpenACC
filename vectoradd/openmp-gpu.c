#include <omp.h>
#include "../common.h"

void sumArraysOnGPUOMP(float *A, float *B, float *C, const int N) {
    #pragma omp target teams distribute \
                parallel for
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char const *argv[])
{
    int nElem = 1 << 28;
    if (argc > 1) nElem = 1 << atoi(argv[1]);
    size_t nBytes = nElem * sizeof(float);

    printf("Vector size: %d\n\n", nElem);

    float *A, *B, *C, *D;
    A = (float *)malloc(nBytes);
    B = (float *)malloc(nBytes);
    C = (float *)malloc(nBytes);
    D = (float *)malloc(nBytes);

    initialData(A, nElem);
    initialData(B, nElem);

    sumArraysOnHost(A, B, C, nElem);

    // warmup
    #pragma omp target data map(to:A[0:nElem]) map(to:B[0:nElem]) map(from:D[0:nElem])
    {
    sumArraysOnGPUOMP(A, B, D, nElem);

    double dtime = - omp_get_wtime();
    sumArraysOnGPUOMP(A, B, D, nElem);
    dtime += omp_get_wtime();
    printf("\"sumArraysOnGPUOMP\"\n");
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, COST * nElem / dtime / 1.0e+9);
    }
    checkResult(C, D, nElem);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
