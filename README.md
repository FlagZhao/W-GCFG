# GCFG_Weight

Extract the static SASS CFG from a CUDA binary, collect per-instruction dynamic
metrics with Nsight Compute (ncu), align both by PC, and emit a **weighted CFG**.

Metrics merged onto every PC:

| Field | Source | Meaning |
|---|---|---|
| `inst_executed` | ncu SourceCounters | warp-level execution count (instruction amount) |
| `thread_inst_executed` | ncu | thread-level execution count |
| `avg_active_lanes` | ncu | `thread_inst / inst`, average active lanes (0–32) |
| `avg_pred_on_lanes` | ncu | average lanes actually enabled after predication |
| `divergent_branches` | ncu | divergent-branch count at this PC |
| `live_gpr` / `live_pred` / `live_ugpr` | `nvdisasm -plr` | live general/predicate/uniform registers at this PC (static) |
| `file` / `line` | `nvdisasm -g` | source location (requires `-lineinfo` at compile time) |

Each kernel additionally carries `registers_per_thread` (ncu LaunchStats).

Basic-block-level aggregates: `exec_count` ((sub)warp executions — under
independent thread scheduling each subwarp arrival counts once, so divergence
is captured by the thread/warp ratio), `avg_active_lanes`,
`avg_pred_on_lanes` (execution-weighted), `divergence` (= 1 − lanes/32),
and a source-line histogram (`src_lines`: per-line static instruction count
and dynamic execution share, plus `src_dominant`).

Register flow per basic block (from the `-lrm narrow` per-register life
ranges, so upstream/downstream effects are explicit): `live_in_gpr`
(inherited from predecessors), `live_out_gpr` (handed to successors),
`live_through_gpr` (held across the whole block with no local def/use —
pressure imposed purely by surrounding code), `max_live_gpr` (peak inside),
`sum_live_gpr` (static pressure integral), plus the concrete register id
sets (`gpr_live_in` / `gpr_live_out` / `gpr_live_through`).
Edges carry topology only — true per-edge traversal counts would need
LBR-like hardware or binary instrumentation and are out of scope.

## Usage

```bash
# All in one: extract CFG + profile with ncu + merge
python3 gcfg_weight.py <cuda_binary> [app args...] -o out/

# Reuse an existing ncu report (skip profiling)
python3 gcfg_weight.py <cuda_binary> --ncu-rep profile.ncu-rep -o out/

# Only process kernels matching a regex; pass extra args to ncu
# (e.g. limit the number of profiled launches)
python3 gcfg_weight.py app --kernel 'vecop' --ncu-args '--launch-count 1' -o out/
```

A `.cubin` can be passed directly (still requires `--ncu-rep` from profiling the
corresponding host program).

## Outputs (three files per kernel)

- `<kernel>.json` — full structure: `bbs` (per-basic-block instruction list with
  all per-PC metrics, plus BB-level aggregates `exec_count` /
  `avg_active_lanes` / `max_live_gpr`) and `edges` (CFG edges)
- `<kernel>.dot` — Graphviz CFG, nodes heat-colored by `inst_executed`,
  labeled with warp execs / avg lanes / max live GPR
  (render with `dot -Tsvg x.dot -o x.svg`)
- `<kernel>.csv` — flat per-instruction table, ready for pandas

## Pipeline details

1. `cuobjdump -xelf all` extracts cubin(s) from the executable
2. `nvdisasm -bbcfg -poff` emits a basic-block-level DOT graph with
   per-instruction PC offsets
3. `nvdisasm -c -poff -plr -lrm count` yields live register counts per PC
4. `ncu --section SourceCounters --section LaunchStats --section Occupancy`
   profiles the app; `ncu --import --page source --csv` exports the
   per-address table
5. NCU reports absolute addresses; subtracting the kernel's minimum address
   gives the offset, which aligns with nvdisasm offsets. Kernel names are
   matched by demangling with `cu++filt` and normalizing signatures
6. Multiple launches of the same kernel are accumulated; avg lanes are
   weighted by `inst_executed`

## Notes

- The NCU CLI does not export the GUI's "Live Registers" column; register
  liveness comes from `nvdisasm -plr` instead (same static analysis as the GUI)
- Trailing padding NOPs are not part of nvdisasm's CFG; NCU reports them as
  zero-executed rows. The tool classifies these as tail padding without
  warning — a WARNING is raised only for unmatched PCs that were **actually
  executed**
- Profiling requires GPU performance-counter permissions; ncu replays each
  kernel several times
