COMMONFLAGS=-fopenmp -O3 -g
STDFLAGS=-std=c11
GCCFLAGS=-fopenacc -O3 -g
BLAS=-lblas -lcblas

all: openmp-cpu.out openmp-gpu.out openacc-gcc.out openacc-pgi.out cuda.out

openmp-cpu.out: openmp-cpu.c
	gcc $< -o $@ $(COMMONFLAGS) $(STDFLAGS) $(BLAS)

openmp-gpu.out: openmp-gpu.c
	~/offload/install/bin/gcc $< -o $@ $(COMMONFLAGS) $(STDFLAGS) $(BLAS)

openacc-gcc.out: openacc-gcc.c
	~/offload/install/bin/gcc $< -o $@ $(GCCFLAGS) $(STDFLAGS) $(BLAS)

openacc-pgi.out: openacc-pgi.c
	nvc $< -o $@ -acc -Minfo=accel -gpu=cc86 -tp=host $(BLAS) $(COMMONFLAGS) $(STDFLAGS)

cuda.out: cuda.o link.o cblaswrapper.o
	gcc cuda.o link.o cblaswrapper.o -o $@ $(BLAS) $(COMMONFLAGS) -L/usr/local/cuda-11.3/lib64 -lcudart -lcublas

cblaswrapper.o: cblaswrapper.c
	gcc -c $< -o $@ -O3 -g

cuda.o: cuda.cu matrixmul.h
	nvcc -dc $< -arch=sm_86 -O3 -g -lcublas

link.o: cuda.o
	nvcc $< -o $@ -arch=sm_86 -dlink

test: all
	./openmp-cpu.out $(size)
	./openmp-gpu.out $(size)
	./cuda.out $(size)
	./openacc-gcc.out $(size)
	./openacc-pgi.out $(size)

clean:
	rm -f *.out *.o

.PHONY: all test clean