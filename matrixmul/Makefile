all: openmp-cpu openmp-gpu-gcc openmp-gpu-clang openmp-gpu-clang-opt openmp-gpu-nvc openacc-gcc openacc-nvc cuda

openmp-cpu: openmp-cpu.c
	gcc $< -o $@ -march=native -fopenmp -O2 -g -lm -lblas -lcblas

openmp-gpu-gcc: openmp-gpu.c
	gcc $< -o $@ -fopenmp -foffload=nvptx-none -O2 -lblas -lcblas

openmp-gpu-clang: openmp-gpu.c
	clang $< -o $@ -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_86 -g -lm -lblas -lcblas

openmp-gpu-clang-opt: openmp-gpu-clang.c
	clang $< -o $@ -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_86 -g -lm -lblas -lcblas -O2

openmp-gpu-nvc: openmp-gpu.c
	nvc $< -o $@ -mp=gpu -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast -lblas -lcblas

openacc-gcc: openacc-gcc.c
	gcc $< -o $@ -fopenacc -O2 -g -lm -lblas -lcblas

openacc-nvc: openacc-nvc.c
	nvc $< -o $@ -acc -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast -lblas -lcblas

cuda: cuda.o link.o cblaswrapper.o
	gcc cuda.o link.o cblaswrapper.o -o $@ -lblas -lcblas -fopenmp -O2 -g -L/usr/local/cuda-11.5/lib64 -lcudart -lcublas

cblaswrapper.o: cblaswrapper.c
	gcc -c $< -o $@ -O2 -g

cuda.o: cuda.cu matrixmul.h
	nvcc -dc $< -arch=sm_86 -O2 -g -lcublas -ccbin gcc-9

link.o: cuda.o
	nvcc $< -o $@ -arch=sm_86 -dlink -ccbin gcc-9