#include <omp.h>
#include "vectoradd.h"

void sumArraysOnGPUOMP(float *A, float *B, float *C, const int N) {
    #pragma omp target teams distribute \
                parallel for num_threads(128)
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
    #pragma omp target enter data map(to:A[0:nElem]) map(to:B[0:nElem]) map(alloc:D[0:nElem])
    //sumArraysOnGPUOMP(A, B, D, nElem);

    double dtime = - omp_get_wtime();
    for (int i = 0; i < 1000; i++) sumArraysOnGPUOMP(A, B, D, nElem);
    dtime += omp_get_wtime();
    #pragma omp target exit data map(from:D[0:nElem])
    printf("\"sumArraysOnGPUOMP\"\n");
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, calcVaddGFLOPS(nElem, dtime));

    checkResult(C, D, nElem);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
