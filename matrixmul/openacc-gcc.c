#include <omp.h>
#include <cblas.h>
#include "matrixmul.h"

void mulMatrixOnACC(float *A, float *B, float *C, const int N) {
    #pragma acc data copyin(A[0:N*N]) copyin(B[0:N*N]) copyout(C[0:N*N])
    #pragma acc parallel loop independent collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0;
            #pragma acc loop seq
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

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nElem, nElem, nElem, 1.0, A, nElem, B, nElem, 1.0, C, nElem);

    // warmup
    #pragma acc enter data copyin(A[0:nxy]) copyin(B[0:nxy]) create(D[0:nxy])
    mulMatrixOnACC(A, B, D, nElem);

    double dtime = - omp_get_wtime();
    mulMatrixOnACC(A, B, D, nElem);
    dtime += omp_get_wtime();
    #pragma acc exit data copyout(D[0:nxy])
    printf("\"mulMatrixOnACC\"\n");
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
    checkResult(C, D, nxy);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
