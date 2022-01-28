# OpenMP-vs-OpenACC
GPU offloading performance comparison between OpenMP and OpenACC directives.
This contains two simple calculations: vector addition and matrix multiplication (SGEMM).

## Prerequisites
* `gcc`: The basic GNU C Compiler.
* `gcc` with NVIDIA PTX: You have to build gcc with this feature enabled in home directory, otherwise change the path in Makefile accordingly. For details, see https://kristerw.blogspot.com/2017/04/building-gcc-with-support-for-nvidia.html.
* `clang` with NVIDIA PTX: Same as `gcc`, you have to build it manually for now. For details, see https://gist.github.com/anjohan/9ee746295ea1a00d9ca69415f40fafc9.
* `nvc` and `nvcc`: These come with NVIDIA HPC Software Development Kit along with CUDA toolkit. For details, see https://developer.nvidia.com/hpc-sdk.
* BLAS and CBLAS libraries: It is used for SGEMM validation. For details, see http://www.netlib.org/blas/.

## Files
1. `openmp-cpu`: Contains sequential and multi-threaded codes by OpenMP. The sequential kernel is skipped when `size > 2^10`. Compiled with basic gcc.
2. `openmp-gpu`: Contains a GPU offloading code using OpenMP directives (OpenMP 4.5 <). Compiled with gcc with NVIDIA PTX GPU offloading feature enabled.
3. `openacc-gcc`: Contains a GPU offloading code using OpenACC directives. Compiled with gcc with NVIDIA PTX GPU offloading feature enabled.
4. `openacc-pgi`: Contains a GPU offloading code using OpenACC directives. Compiled with NVIDIA PGI Compiler (also known as NVC).
5. `cuda`: Contains a CUDA version of the code and also the `cublasSgemm` routine. Compiled with nvcc and linked to the CBLAS wrapper using gcc.

## Usage
Use Makefile in each directory. Run `make run` to execute every single version, you can also specify the array size using <code>make run size=*(a power of 2)*</code>.

## Misc.
* `openacc-pgi` seems to incur a relatively large error compared to others. I'm not certain about the reason, but maybe it could be the gang / worker settings which is automatically set by the compiler.
* These OpenMP and OpenACC offloading codes should work on the LLVM/Clang compiler (Although the OpenACC implementation is still in progress and it's translated to OpenMP directives internally [1], therefore might cause some issues), but I haven't tried it yet. Also you may have to build it to use the GPU offloading feature, just like gcc. 

[1] Clacc: OpenACC support for Clang and LLVM. https://www.openacc.org/sites/default/files/inline-images/events/F2F20%20presentations/BoF-clacc.pdf
