BASE_DIR := $(shell pwd)

VADD_BIN_DIR := $(BASE_DIR)/bin/vectoradd
MMUL_BIN_DIR := $(BASE_DIR)/bin/matrixmul

all: VADD MMUL

VADD:
	cd vectoradd;	make all;\
	mv openmp-cpu $(VADD_BIN_DIR);\
	mv openmp-gpu-gcc $(VADD_BIN_DIR);\
	mv openmp-gpu-clang $(VADD_BIN_DIR);\
	mv openmp-gpu-nvc $(VADD_BIN_DIR);\
	mv openacc-gcc $(VADD_BIN_DIR);\
	mv openacc-nvc $(VADD_BIN_DIR);\
	mv cuda $(VADD_BIN_DIR)

MMUL:
	cd matrixmul;	make all;\
	mv openmp-cpu $(MMUL_BIN_DIR);\
	mv openmp-gpu-gcc $(MMUL_BIN_DIR);\
	mv openmp-gpu-clang $(MMUL_BIN_DIR);\
	mv openmp-gpu-nvc $(MMUL_BIN_DIR);\
	mv openacc-gcc $(MMUL_BIN_DIR);\
	mv openacc-nvc $(MMUL_BIN_DIR);\
	mv cuda $(MMUL_BIN_DIR)

run: VADD_run MMUL_run

VADD_run:
	$(info You may specify size of vectors by `VECTOR_SIZE=` argument)
	$(VADD_BIN_DIR)/openmp-cpu $(VECTOR_SIZE)
	$(VADD_BIN_DIR)/openmp-gpu-gcc $(VECTOR_SIZE)
	$(VADD_BIN_DIR)/openmp-gpu-clang $(VECTOR_SIZE)
	$(VADD_BIN_DIR)/openmp-gpu-nvc $(VECTOR_SIZE)
	$(VADD_BIN_DIR)/openacc-gcc $(VECTOR_SIZE)
	$(VADD_BIN_DIR)/openacc-nvc $(VECTOR_SIZE)
	$(VADD_BIN_DIR)/cuda $(VECTOR_SIZE)

MMUL_run:
	$(info You may specify size of matrices by `MATRIX_SIZE=` argument)
	$(MMUL_BIN_DIR)/openmp-cpu $(MATRIX_SIZE)
	$(MMUL_BIN_DIR)/openmp-gpu-gcc $(MATRIX_SIZE)
	$(MMUL_BIN_DIR)/openmp-gpu-nvc $(MATRIX_SIZE)
	$(MMUL_BIN_DIR)/openmp-gpu-clang $(MATRIX_SIZE)
	$(MMUL_BIN_DIR)/openacc-gcc $(MATRIX_SIZE)
	$(MMUL_BIN_DIR)/openacc-nvc $(MATRIX_SIZE)
	$(MMUL_BIN_DIR)/cuda $(MATRIX_SIZE)

clean: VADD_clean MMUL_clean

VADD_clean:
	cd $(VADD_BIN_DIR);	rm -f *

MMUL_clean:
	cd matrixmul; rm -f *.o
	cd $(MMUL_BIN_DIR); rm -f *
