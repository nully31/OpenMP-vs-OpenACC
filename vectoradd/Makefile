all: openmp-cpu openmp-gpu-gcc openmp-gpu-clang openmp-gpu-nvc openacc-gcc openacc-nvc cuda

openmp-cpu: openmp-cpu.c
	gcc $< -o $@ -march=native -fopenmp -O2 -g -lm

openmp-gpu-gcc: openmp-gpu.c
	gcc $< -o $@ -fopenmp -foffload=nvptx-none -O2

openmp-gpu-clang: openmp-gpu.c
	clang $< -o $@ -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_86 -O2 -g -lm

openmp-gpu-nvc: openmp-gpu.c
	nvc $< -o $@ -mp=gpu -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast

openacc-gcc: openacc-gcc.c
	gcc $< -o $@ -fopenacc -O2 -g -lm

openacc-nvc: openacc-nvc.c
	nvc $< -o $@ -acc -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast

cuda: cuda.cu
	nvcc $< -o $@ -arch=sm_86 -Xcompiler -fopenmp -O2 -g -lm -ccbin gcc-9