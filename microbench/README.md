# icache microbenchmark

`icache_gen.py` generates `icache_bench.cu`: one warp loops over a
straight-line body of N independent SASS FADDs (inline PTX asm volatile,
1:1 to 16B SASS instructions), timed with clock64().

Results on RTX 5090 (GB202, sm_120), driver 580.97, CUDA 13.0, reps=64:

| footprint | cyc/inst |
|-----------|----------|
| 2 KB – 120 KB | 1.10 – 1.16 (flat) |
| 128 KB | 1.63 (transition onset) |
| 136 KB | 3.22 |
| 144 KB – 384 KB | ~4.27 (L2 streaming) |

- Effective i-cache capacity per SM: **128 KB** (8192 instructions).
- No separate L0 cliff is visible: sequential next-line prefetch hides the
  L0->L1i path for straight-line code.
- Beyond capacity: ~4.27 cyc/inst, a ~3.8x slowdown; NCU WarpStateStats
  attributes 77.6% of CPI to the No-Inst (instruction fetch) stall.

## Disambiguation experiments (icache_gen2.py)

- **SMEM carveout**: 96KB body with 0 vs 99KB dynamic shared memory —
  cyc/inst unchanged (1.136). Instruction fetch does NOT share the unified
  L1/shared data array; the 128KB i-cache is a separate structure.
- **MULTI**: W warps on one SM, each running a disjoint 64KB region.
  Cliff at W=2 (128KB combined): 1.15 / 2.52 / 6.43 / 7.28 for W=1..4.
  Confirms the same 128KB capacity from a second access pattern and shows
  it is shared across all warps of the SM. Over-capacity contention with
  multiple warps (6.4-7.3) is worse than single-warp streaming (4.27).
- Unresolved: per-SM vs per-TPC sharing (needs SM-affinity control, e.g.
  green contexts). Either way the per-kernel hot-path budget is 128KB.
