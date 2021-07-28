# OpenMP-vs-OpenACC
GPU offloading performance comparison between OpenMP and OpenACC directives.
This contains two simple calculations: vector addition and matrix multiplication (SGEMM).

## Files
`openmp-cpu`: Contains sequential and multi-threaded codes by OpenMP. Compiled with basic gcc.
`openmp-gpu`: Contains a GPU offloading code using OpenMP directives (OpenMP 4.5 <). Needs gcc with NVIDIA PTX GPU offloading feature enabled.
`openacc-gcc`: Contains a GPU offloading code using OpenACC directives. Needs gcc with NVIDIA PTX GPU offloading feature enabled.
`openacc-pgi`: Contains a GPU offloading code using OpenACC directives. Compiled with NVIDIA PGI Compiler (also known as NVC).
`cuda`: Contains a CUDA version of the code. Compiled with nvcc.

## Usage
Use Makefile in each directory. Run `make test` to execute every single version, you can also specify the array size using by `make test size=(a power of 2)`.

## Prerequisites
`gcc`: The basic GNU C Compiler.
`gcc` with NVIDIA PTX: You have to build gcc with this feature enabled in home directory, otherwise change the path in Makefile accordingly. For details, see (https://kristerw.blogspot.com/2017/04/building-gcc-with-support-for-nvidia.html).
`PGI Compiler` and `nvcc`: These come with NVIDIA HPC Software Development Kit. For details, see (https://developer.nvidia.com/hpc-sdk).
