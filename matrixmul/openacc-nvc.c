/*
 * As of Jan 29, my whole PC crashes if cblas library is involved when matrix size is above 1024.
 * Due to this problem, validation check is temporally disabled.
 */

#include <omp.h>
#include <cblas.h>
#include "matrixmul.h"

void mulMatrixOnACC(float *A, float *B, float *D, int nElem)
{
    #pragma acc data present(A[0:nElem*nElem], B[0:nElem*nElem], D[0:nElem*nElem])
    #pragma acc kernels loop independent collapse(2)
    for (int i = 0; i < nElem; i++) {
        for (int j = 0; j < nElem; j++) {
            float temp = 0.0;
            #pragma acc loop seq
            for (int k = 0; k < nElem; k++) {
                temp += A[i * nElem + k] * B[k * nElem + j];
            }
            D[i * nElem + j] = temp;
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

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nElem, nElem, nElem, 1.0, A, nElem, B, nElem, 1.0, C, nElem);

    #pragma acc enter data copyin(A[0:nxy]) copyin(B[0:nxy]) create(D[0:nxy])

    printf("\033[1mMatrix Multiplication on GPU using OpenACC&%s\033[0m\n", compiler);
    double dtime = - omp_get_wtime();
    mulMatrixOnACC(A, B, D, nElem);
    dtime += omp_get_wtime();
    #pragma acc exit data copyout(D[0:nxy])
    printf("\"mulMatrixOnACC\"\n");
    printf("Elapsed time: %.3f sec, %.4f TFLOPS\n\n", dtime, calcMmulTFLOPS(nElem, dtime));
    // checkResult(C, D, nxy);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
