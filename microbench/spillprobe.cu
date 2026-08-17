// Black-box probes of ptxas spill-cost heuristics (ptxas is closed source).
// Build with a tight -maxrregcount so spilling is forced, then inspect where
// STL/LDL land in the SASS.
//
// Probe 1 (k_loopdepth): values L0..L7 used only inside a loop, values
// S0..S7 used only after it. A loop-depth-weighted allocator must spill
// the S group (all LDL after the loop, none inside).
//
// Probe 2 (k_trapcold): symmetric data-dependent branches; T group used
// only in a branch ending in __trap(), E group only in the normal branch.
// A coldness-heuristic allocator should prefer spilling the T group.
#include <cuda_runtime.h>

#define DERIVE(v, off, c) \
    float v = fmaf(a[i + off], a[i + off + 1], c); v = fmaf(v, v, c);

extern "C" __global__ void k_loopdepth(const float *a, float *o, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    DERIVE(l0, 0, 1.f) DERIVE(l1, 2, 2.f) DERIVE(l2, 4, 3.f) DERIVE(l3, 6, 4.f) DERIVE(l4, 8, 5.f) DERIVE(l5, 10, 6.f) DERIVE(l6, 12, 7.f) DERIVE(l7, 14, 8.f) DERIVE(l8, 16, 9.f) DERIVE(l9, 18, 10.f)
    DERIVE(s0, 20, 1.f) DERIVE(s1, 22, 2.f) DERIVE(s2, 24, 3.f) DERIVE(s3, 26, 4.f) DERIVE(s4, 28, 5.f) DERIVE(s5, 30, 6.f) DERIVE(s6, 32, 7.f) DERIVE(s7, 34, 8.f) DERIVE(s8, 36, 9.f) DERIVE(s9, 38, 10.f) DERIVE(s10, 40, 11.f) DERIVE(s11, 42, 12.f) DERIVE(s12, 44, 13.f) DERIVE(s13, 46, 14.f)

    // early store consuming every value: pins all defs before this point, and
    // makes re-loading from a[] unsafe afterwards (o may alias a) — the
    // values must now survive in registers or be spilled.
    o[i] = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13;

    float acc = 0.f;
#pragma unroll 1
    for (int k = 0; k < iters; ++k) {
        acc = fmaf(acc, l0, l1); acc = fmaf(acc, l2, l3); acc = fmaf(acc, l4, l5); acc = fmaf(acc, l6, l7); acc = fmaf(acc, l8, l9);
    }
    // straight-line uses after the loop
    acc = fmaf(acc, s0, s1); acc = fmaf(acc, s2, s3); acc = fmaf(acc, s4, s5); acc = fmaf(acc, s6, s7); acc = fmaf(acc, s8, s9); acc = fmaf(acc, s10, s11); acc = fmaf(acc, s12, s13);
    o[i] = acc;
}

extern "C" __global__ void k_trapcold(const float *a, float *o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    DERIVE(t0, 0, 1.f) DERIVE(t1, 2, 2.f) DERIVE(t2, 4, 3.f) DERIVE(t3, 6, 4.f) DERIVE(t4, 8, 5.f) DERIVE(t5, 10, 6.f) DERIVE(t6, 12, 7.f) DERIVE(t7, 14, 8.f) DERIVE(t8, 16, 9.f) DERIVE(t9, 18, 10.f)
    DERIVE(e0, 20, 1.f) DERIVE(e1, 22, 2.f) DERIVE(e2, 24, 3.f) DERIVE(e3, 26, 4.f) DERIVE(e4, 28, 5.f) DERIVE(e5, 30, 6.f) DERIVE(e6, 32, 7.f) DERIVE(e7, 34, 8.f) DERIVE(e8, 36, 9.f) DERIVE(e9, 38, 10.f)

    // early store pinning all defs (see k_loopdepth)
    o[i] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8 + e9;

    float x = a[i + 80];
    if (x < 0.f) {           // ends in trap -> statically "cold" by
        float u = t0 + t1; u = fmaf(u, t2, t3); u = fmaf(u, t4, t5); u = fmaf(u, t6, t7); u = fmaf(u, t8, t9);
        o[i] = u;
        __trap();
    } else {
        float v = e0 + e1; v = fmaf(v, e2, e3); v = fmaf(v, e4, e5); v = fmaf(v, e6, e7); v = fmaf(v, e8, e9);
        o[i] = v * 2.f;
    }
}
