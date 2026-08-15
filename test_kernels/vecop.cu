// Sample kernel with branches, a loop, and warp divergence,
// used to validate the CFG + NCU per-PC metric pipeline.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecop(const float *a, const float *b, float *c, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = a[i];
    float y = b[i];

    // divergent branch: half the lanes of each warp take each side
    if ((i & 16) == 0) {
        x = x * y + 1.0f;
    } else {
        x = x - y * 0.5f;
    }

    // loop with data-dependent trip count -> divergence across lanes
    int trips = iters + (i % 7);
    for (int k = 0; k < trips; ++k) {
        x = fmaf(x, 1.0001f, y);
    }

    c[i] = x;
}

int main(int argc, char **argv) {
    int n = 1 << 20;
    int iters = 32;
    size_t bytes = n * sizeof(float);

    float *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);
    cudaMemset(a, 0x3f, bytes);
    cudaMemset(b, 0x3e, bytes);

    dim3 block(256), grid((n + block.x - 1) / block.x);
    vecop<<<grid, block>>>(a, b, c, n, iters);
    cudaError_t err = cudaDeviceSynchronize();
    printf("vecop done: %s\n", cudaGetErrorString(err));

    cudaFree(a); cudaFree(b); cudaFree(c);
    return err == cudaSuccess ? 0 : 1;
}
