#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#ifndef _COMMON_H_
#define _COMMON_H_

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s: %d,\t", __FILE__, __LINE__);                     \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0e-1;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f\tgpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

double calcMmulTFLOPS(int nElem, double dtime) {
    double cost = 2.0 * nElem;
    return cost * nElem * nElem / dtime / 1.0e+12;
}

#ifdef __cplusplus
extern "C" {
#endif
    void call_cblas_sgemm(float *, float *, float *, const int);
#ifdef __cplusplus
}
#endif
#endif
