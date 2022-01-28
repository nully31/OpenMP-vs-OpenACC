/*
 * On clang 14.0, if the function `mulMatrixOnACC()` is absent from the code even if it's not used anywhere, the comiler crashes (like what??? how???).
 * Otherwise, you have to disable optimization flags completely.
 * Really weird bug of clang 14.0 with GPU offloading.
 */

#include <omp.h>
#include <cblas.h>
#include "matrixmul.h"

void mulMatrixOnACC(float *A, float *B, float *C, const int N) {
    #pragma omp target teams distribute \
            parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0;
            for (int k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = temp;
        }
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

    int nElem = 1 << 10;
    if (argc > 1) nElem = 1 << atoi(argv[1]);
    int nxy = nElem * nElem;
    size_t nBytes = nxy * sizeof(float);

    printf("Matrix size: %d x %d\n\n", nElem, nElem);

    float *A, *B, *C, *D;
    A = (float *)malloc(nBytes);
    B = (float *)malloc(nBytes);
    C = (float *)malloc(nBytes);
    D = (float *)malloc(nBytes);

    initialData(A, nxy);
    initialData(B, nxy);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nElem, nElem, nElem, 1.0, A, nElem, B, nElem, 1.0, C, nElem);

    #pragma omp target enter data map(to:A[0:nxy]) map(to:B[0:nxy]) map(alloc:D[0:nxy])

    printf("\033[1mMatrix Multiplication on GPU using OpenMP&%s with optimization flag\033[0m\n", compiler);
    double dtime = - omp_get_wtime();
    #pragma omp target teams distribute \
            parallel for collapse(2)
    for (int i = 0; i < nElem; i++) {
        for (int j = 0; j < nElem; j++) {
            float temp = 0.0;
            for (int k = 0; k < nElem; k++) {
                temp += A[i * nElem + k] * B[k * nElem + j];
            }
            D[i * nElem + j] = temp;
        }
    }
    dtime += omp_get_wtime();
    #pragma omp target exit data map(from:D[0:nxy])
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
    checkResult(C, D, nxy);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}