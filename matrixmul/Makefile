COMMONFLAGS=-fopenmp -O3 -g
STDFLAGS=-std=c11
GCCFLAGS=-fopenacc -O3 -g
BLAS=-lblas -lcblas

all: openmp-cpu.o openmp-gpu-gcc.o openmp-gpu-clang.o openmp-gpu-nvc.o openacc-gcc.o openacc-nvc.o cudao.o

openmp-cpu.o: openmp-cpu.c
	gcc $< -o $@ -march=native -fopenmp -O2 -g -lm -lblas -lcblas

openmp-gpu-gcc.o: openmp-gpu.c
	gcc $< -o $@ -fopenmp -foffload=nvptx-none -O2 -lblas -lcblas

openmp-gpu-clang.o: openmp-gpu.c
	clang $< -o $@ -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_86 -O2 -g -lm

openmp-gpu-nvc.o: openmp-gpu.c
	nvc $< -o $@ -mp=gpu -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast -lblas -lcblas

openacc-gcc.o: openacc-gcc.c
	gcc $< -o $@ -fopenacc -O2 -g -lm -lblas -lcblas

openacc-nvc.o: openacc-nvc.c
	nvc $< -o $@ -acc -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast

cudao.o: cuda.o link.o cblaswrapper.o
	gcc cuda.o link.o cblaswrapper.o -o $@ $(BLAS) $(COMMONFLAGS) -L/usr/local/cuda-11.5/lib64 -lcudart -lcublas

cblaswrapper.o: cblaswrapper.c
	gcc -c $< -o $@ -O3 -g

cuda.o: cuda.cu matrixmul.h
	nvcc -dc $< -arch=sm_86 -O3 -g -lcublas -ccbin gcc-9

link.o: cuda.o
	nvcc $< -o $@ -arch=sm_86 -dlink -ccbin gcc-9


run: all
	./openmp-cpu.o $(size)
	./openmp-gpu-gcc.o $(size)
	./openmp-gpu-nvc.o $(size)
	./cudao.o $(size)
	./openacc-gcc.o $(size)
	./openacc-nvc.o $(size)

clean:
	rm -f *.o *.out

.PHONY: all run clean