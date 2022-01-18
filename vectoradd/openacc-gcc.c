#include <omp.h>
#include "vectoradd.h"

void sumArraysOnACC(float *A, float *B, float *C, const int N) {
    #pragma acc data copyin(A[0:N]) copyin(B[0:N]) copyout(C[0:N])
    #pragma acc parallel loop independent
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
    #pragma acc enter data copyin(A[0:nElem]) copyin(B[0:nElem]) create(D[0:nElem])
    sumArraysOnACC(A, B, D, nElem);

    double dtime = - omp_get_wtime();
    for (int i = 0; i < 1000; i++) sumArraysOnACC(A, B, D, nElem);
    dtime += omp_get_wtime();
    #pragma acc exit data copyout(D[0:nElem])
    printf("\"sumArraysOnACC\"\n");
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, calcVaddGFLOPS(nElem, dtime));
    checkResult(C, D, nElem);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
