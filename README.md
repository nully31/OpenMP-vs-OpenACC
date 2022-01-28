# OpenMP-vs-OpenACC
GPU offloading performance comparison between OpenMP and OpenACC directives (this also includes CPU sequential and multithreaded codes, as well as CUDA kernels, for reference).
As of now, this repository contains two simple calculations: vector addition and matrix multiplication (SGEMM).

## Prerequisites
* `gcc`: The basic GNU C Compiler.
* `gcc` with NVIDIA PTX: You have to build gcc with this feature enabled in home directory, otherwise change the path in Makefile accordingly. For details, see https://kristerw.blogspot.com/2017/04/building-gcc-with-support-for-nvidia.html.
* `clang` with NVIDIA PTX: Same as `gcc`, you have to build it manually for now. For details, see https://gist.github.com/anjohan/9ee746295ea1a00d9ca69415f40fafc9.
* `nvc` and `nvcc`: These come with NVIDIA HPC Software Development Kit along with CUDA toolkit. For details, see https://developer.nvidia.com/hpc-sdk.
* BLAS and CBLAS libraries: It is used for SGEMM validation. For details, see http://www.netlib.org/blas/.

## Directory structure
* `OpenMP-vs-OpenACC/bin`: binary executables.
* `OpenMP-vs-OpenACC/vectoradd`: source code for vector addition.
* `OpenMP-vs-OpenACC/matrixmul`: source code for matrix multiplication.


## Usage
Use Makefile in the root directory (`OpenMP-vs-OpenACC/Makefile`). `make run` can be used to run all the binaries.
You may specify the size of vectors for vector addition or matrices for matrix multiplication by using `VECTOR_SIZE=` or `MATRIX_SIZE=` command line arguments.
Both numbers should be the exponent of 2.
