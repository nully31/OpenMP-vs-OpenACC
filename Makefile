COMMONFLAGS=-fopenmp -O3 -g
STDFLAGS=-std=c11
GCCFLAGS=-fopenacc -O3 -g

all: openmp-cpu.out openmp-gpu.out openacc-gcc.out openacc-pgi.out cuda.out

openmp-cpu.out: openmp-cpu.c
	gcc $< -o $@ $(COMMONFLAGS) $(STDFLAGS)

openmp-gpu.out: openmp-gpu.c
	~/offload/install/bin/gcc $< -o $@ $(COMMONFLAGS) $(STDFLAGS)

openacc-gcc.out: openacc-gcc.c
	~/offload/install/bin/gcc $< -o $@ $(GCCFLAGS) $(STDFLAGS)

openacc-pgi.out: openacc-pgi.c
	nvc $< -o $@ -acc -Minfo=accel -gpu=cc86 -tp=host $(COMMONFLAGS) $(STDFLAGS)

cuda.out: cuda.cu
	nvcc $< -o $@ -arch=sm_86 -Xcompiler $(COMMONFLAGS)

test: all
	./openmp-cpu.out
	./openmp-gpu.out
	./cuda.out
	./openacc-gcc.out
	./openacc-pgi.out

clean:
	rm -f *.out

.PHONY: all test clean