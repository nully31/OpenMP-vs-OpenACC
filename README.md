# OpenMP-vs-OpenACC
GPU offloading performance comparison between OpenMP and OpenACC directives.
This contains two simple calculations: vector addition and matrix multiplication (SGEMM).

## Files
1. `openmp-cpu`: Contains sequential and multi-threaded codes by OpenMP. Compiled with basic gcc.
2. `openmp-gpu`: Contains a GPU offloading code using OpenMP directives (OpenMP 4.5 <). Compiled with gcc with NVIDIA PTX GPU offloading feature enabled.
3. `openacc-gcc`: Contains a GPU offloading code using OpenACC directives. Compiled with gcc with NVIDIA PTX GPU offloading feature enabled.
4. `openacc-pgi`: Contains a GPU offloading code using OpenACC directives. Compiled with NVIDIA PGI Compiler (also known as NVC).
5. `cuda`: Contains a CUDA version of the code. Compiled with nvcc.

## Usage
Use Makefile in each directory. Run `make test` to execute every single version, you can also specify the array size using <code>make test size=*(a power of 2)*</code>.

## Prerequisites
* `gcc`: The basic GNU C Compiler.
* `gcc` with NVIDIA PTX: You have to build gcc with this feature enabled in home directory, otherwise change the path in Makefile accordingly. For details, see https://kristerw.blogspot.com/2017/04/building-gcc-with-support-for-nvidia.html.
* `PGI Compiler` and `nvcc`: These come with NVIDIA HPC Software Development Kit. For details, see https://developer.nvidia.com/hpc-sdk.
* `BLAS` and `CBLAS` libraries: It is used for SGEMM validation. For details, see http://www.netlib.org/blas/.

## Misc.
These OpenMP and OpenACC offloading codes should work on the LLVM/Clang compiler (Although the OpenACC implementation is still in progress and it's translated to OpenMP directives internally [1], therefore might cause some issues), but I haven't tried it yet.

[1] Clacc: OpenACC support for Clang and LLVM. https://www.openacc.org/sites/default/files/inline-images/events/F2F20%20presentations/BoF-clacc.pdf
