#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifndef _COMMON_H_
#define _COMMON_H_

#define COST_VA 1
#define COST_MM (2 * nElem)

#define BLOCKSIZE 64
#define UNROLL 16

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
    double epsilon = 1.0e-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
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

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

static void do_block(float *A, float *B, float *C, const int n,
			const int si, const int sj, const int sk) {
    #pragma omp parallel for collapse(2)
	for (int i = si; i < si + BLOCKSIZE; i++) {
		for (int j = sk; j < sk + BLOCKSIZE; j++) {
			float cij = C[i*n+j]; /* cij = C[i][j] */
			for (int k = sj; k < sj + BLOCKSIZE; k++)
				cij += A[i*n+k] * B[k*n+j]; /* cij = A[i][k] * B[k][j] */
			C[i*n+j] = cij; /* C[i][j] = cij */
		}
	}
}

void mulMatrixOnHost(float *A, float *B, float *C, const int n) {
    #pragma omp parallel for collapse(3)
	for (int si = 0; si < n; si += BLOCKSIZE) {
		for (int sj = 0; sj < n; sj += BLOCKSIZE) {
			for (int sk = 0; sk < n; sk += BLOCKSIZE) {
				do_block(A, B, C, n, si, sj, sk);
			}
		}
	}
}
#endif
