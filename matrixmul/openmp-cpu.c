#include <omp.h>
#include <cblas.h>
#include "matrixmul.h"

void mulMatrixOnHost(float *A, float *B, float *C, const int N) {
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

void mulMatrixOnHostOMP(float *A, float *B, float *C, const int N) {
    #pragma omp parallel for collapse(2)
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

    double dtime = - omp_get_wtime();
    printf("\033[1mCBLAS using %s\033[0m\n", compiler);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nElem, nElem, nElem, 1.0, A, nElem, B, nElem, 1.0, C, nElem);
    dtime += omp_get_wtime();
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));

    if (nElem < 1 << 11) {
        printf("\033[1mMatrix Multiplication on CPU (Sequential) using %s\033[0m\n", compiler);
        dtime = - omp_get_wtime();
        mulMatrixOnHost(A, B, D, nElem);
        dtime += omp_get_wtime();
        printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
        checkResult(C, D, nxy);
    }

    printf("\033[1mMatrix Multiplication on CPU with %d threads using OpenMP&%s\033[0m\n", omp_get_max_threads(), compiler);
    dtime = - omp_get_wtime();
    mulMatrixOnHostOMP(A, B, D, nElem);
    dtime += omp_get_wtime();
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
    checkResult(C, D, nxy);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
