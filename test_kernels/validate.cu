// Kernels with analytically known branch behavior, used to validate
// GCFG_Weight divergence numbers against theory.
// Launch: N = 1<<20 threads, 256/block -> 4096 blocks, 32768 warps.
//
// Each branch side contains a 4-iteration unroll-1 loop: loops cannot be
// if-converted, so ptxas must emit real branches and distinct basic blocks
// (a plain FMA body just gets predicated/FSEL-merged, see git history).
#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 20)
#define LOOP4(v, c)                          \
    _Pragma("unroll 1")                      \
    for (int k = 0; k < 4; ++k)              \
        v = fmaf(v, 1.01f, c);

// warp-uniform branch: whole block takes one side.
// Expect per side: loop body 16384 warps * 4 iters = 65536 execs, 32 lanes.
__global__ void k_uniform(const float *a, float *o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[i];
    if (blockIdx.x % 2 == 0) {
        LOOP4(v, 1.0f)
    } else {
        LOOP4(v, -1.0f)
    }
    o[i] = v;
}

// half-warp split: lanes 0-15 vs 16-31 of EACH warp (& 31 gives the lane id;
// a plain `threadIdx.x < 16` would only split warp 0 of each block).
// Expect per side: 32768 warps * 4 iters = 131072 execs, 16 lanes.
__global__ void k_half(const float *a, float *o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[i];
    if ((threadIdx.x & 31) < 16) {
        LOOP4(v, 2.0f)
    } else {
        LOOP4(v, -2.0f)
    }
    o[i] = v;
}

// quarter split: lanes with (tid&3)==0 (8 of 32) vs the rest (24 of 32).
// Expect: taken body 131072 execs @ 8 lanes, else body 131072 @ 24 lanes.
__global__ void k_quarter(const float *a, float *o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[i];
    if ((threadIdx.x & 3) == 0) {
        LOOP4(v, 3.0f)
    } else {
        LOOP4(v, -3.0f)
    }
    o[i] = v;
}

// data-dependent trip count 1..4 per lane.
// Expect: body 4*32768 = 131072 execs, avg lanes (32+24+16+8)/4 = 20.0.
__global__ void k_loop(const float *a, float *o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = a[i];
    int trips = (threadIdx.x & 3) + 1;
#pragma unroll 1
    for (int k = 0; k < trips; ++k)
        v = fmaf(v, 1.01f, 0.5f);
    o[i] = v;
}

int main() {
    float *a, *o;
    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&o, N * sizeof(float));
    cudaMemset(a, 0x3f, N * sizeof(float));
    dim3 b(256), g(N / 256);
    k_uniform<<<g, b>>>(a, o);
    k_half<<<g, b>>>(a, o);
    k_quarter<<<g, b>>>(a, o);
    k_loop<<<g, b>>>(a, o);
    cudaError_t err = cudaDeviceSynchronize();
    printf("validate done: %s\n", cudaGetErrorString(err));
    cudaFree(a); cudaFree(o);
    return err == cudaSuccess ? 0 : 1;
}
