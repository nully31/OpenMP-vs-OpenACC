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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nElem, nElem, nElem, 1.0, A, nElem, B, nElem, 1.0, C, nElem);
    dtime += omp_get_wtime();
    printf("\"CBLAS\"\n");
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));

    dtime = - omp_get_wtime();
    mulMatrixOnHost(A, B, D, nElem);
    dtime += omp_get_wtime();
    printf("\"mulMatrixOnHost\"\n");
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
    checkResult(C, D, nxy);

    dtime = - omp_get_wtime();
    mulMatrixOnHostOMP(A, B, D, nElem);
    dtime += omp_get_wtime();
    printf("\"mulMatrixOnHostOMP\" with %d threads\n", omp_get_max_threads());
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
    checkResult(C, D, nxy);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
