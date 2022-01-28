COMMONFLAGS=-fopenmp -O3 -g -lm
STDFLAGS=-std=c11
GCCFLAGS=-fopenacc -O3 -g -lm

all: openmp-cpu.o openmp-gpu-gcc.o openmp-gpu-clang.o openmp-gpu-nvc.o openacc-gcc.o openacc-nvc.o cuda.o

openmp-cpu.o: openmp-cpu.c
	gcc $< -o $@ -march=native -fopenmp -O2 -g -lm

openmp-gpu-gcc.o: openmp-gpu.c
	gcc $< -o $@ -fopenmp -foffload=nvptx-none -O2

openmp-gpu-clang.o: openmp-gpu.c
	clang $< -o $@ -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_86 -O2 -g -lm

openmp-gpu-nvc.o: openmp-gpu.c
	nvc $< -o $@ -mp=gpu -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast

openacc-gcc.o: openacc-gcc.c
	gcc $< -o $@ -fopenacc -O2 -g -lm

openacc-nvc.o: openacc-nvc.c
	nvc $< -o $@ -acc -Minfo=accel -gpu=cc86 -tp=host -g -lm -fast

cuda.o: cuda.cu
	nvcc $< -o $@ -arch=sm_86 -Xcompiler -fopenmp -O2 -g -lm -ccbin gcc-9

run: all
	./openmp-cpu.o $(size)
	./openmp-gpu-gcc.o $(size)
	./openmp-gpu-clang.o $(size)
	./openmp-gpu-nvc.o $(size)
	./openacc-gcc.o $(size)
	./openacc-nvc.o $(size)
	./cuda.o $(size)

clean:
	rm -f *.o

.PHONY: all run clean