#include <cblas.h>

void call_cblas_sgemm(float *h_A, float *h_B, float *hostRef, const int nElem) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nElem, nElem, nElem, 1.0, h_A, nElem, h_B, nElem, 1.0, hostRef, nElem);
}