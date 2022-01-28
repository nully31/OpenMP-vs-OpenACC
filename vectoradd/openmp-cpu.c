#include <omp.h>
#include "vectoradd.h"

void sumArraysOnHostOMP(float *A, float *B, float *C, const int N) {
    #pragma omp parallel for simd
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char const *argv[])
{
    char* compiler =
#ifdef __NVCOMPILER
    "nvc";
#elif __clang__
    "clang";
#elif __GNUC__
    "gcc";
#endif

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

    // warmup
    //sumArraysOnHost(A, B, C, nElem);

    printf("\033[1mVector Addition on CPU (Sequential) using %s\033[0m\n", compiler);
    double dtime = - omp_get_wtime();
    for (int i = 0; i < 1000; i++) sumArraysOnHost(A, B, C, nElem);
    dtime += omp_get_wtime();
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, calcVaddGFLOPS(nElem, dtime));

    printf("\033[1mVector Addition on CPU with %d threads using OpenMP&%s\033[0m\n", omp_get_max_threads(), compiler);
    dtime = - omp_get_wtime();
    for (int i = 0; i < 1000; i++) sumArraysOnHostOMP(A, B, D, nElem);
    dtime += omp_get_wtime();
    printf("Elapsed time: %.3f sec, %lf GFLOPS\n\n", dtime, calcVaddGFLOPS(nElem, dtime));
    checkResult(C, D, nElem);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
