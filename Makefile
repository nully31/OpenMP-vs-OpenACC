COMMONFLAGS=-fopenmp -O3
STDFLAGS=-std=c11

all: openmp-cpu.out openacc.out cuda.out

openmp-cpu.out: openmp-cpu.c
	gcc $< -o $@ $(COMMONFLAGS) $(STDFLAGS)

openacc.out: openacc.c
	pgcc $< -o $@ -acc -Minfo=accel $(COMMONFLAGS) $(STDFLAGS)

cuda.out: cuda.cu
	nvcc $< -o $@ -arch=sm_86 -Xcompiler $(COMMONFLAGS)

test: all
	./openmp-cpu.out
	./cuda.out
	./openacc.out

clean:
	rm -f *.out

.PHONY: all test clean