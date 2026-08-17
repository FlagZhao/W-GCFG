#!/usr/bin/env python3
"""GCFG_Weight: annotate a CUDA binary's static CFG with per-PC dynamic metrics.

Pipeline:
  1. cuobjdump -xelf all   -> extract cubin(s) from the CUDA binary
  2. nvdisasm -bbcfg -poff -> basic-block CFG (DOT) with per-instruction PC offsets
  3. nvdisasm -plr -lrm count -> per-PC live register counts (GPR/PRED/UGPR)
  4. ncu --section SourceCounters ... -> profile; --page source --csv export gives
     per-address: Instructions Executed, Thread Instructions Executed,
     Avg. Threads Executed (= avg active lanes), Divergent Branches
  5. merge by PC offset -> weighted CFG (JSON + annotated DOT + flat CSV)

Usage:
  gcfg_weight.py <cuda_binary> [-- app args...]
  gcfg_weight.py <cuda_binary> --ncu-rep existing.ncu-rep
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

CUDA_BIN = os.path.dirname(shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm")


def run(cmd, **kw):
    kw.setdefault("check", True)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


# ---------------------------------------------------------------- CFG (DOT)

def unescape_dot(s):
    return re.sub(r"\\(.)", r"\1", s)


INST_RE = re.compile(r"^(?:\|?<[^>]+>)*([0-9a-fA-F]+):\s+(.*?)\s*;?\s*$")


def parse_bbcfg(dot_text):
    """Parse `nvdisasm -bbcfg -poff` DOT output.

    Returns {func_name: {"bbs": {node_id: {...}}, "edges": [(src, dst)]}}.
    With -bbcfg every DOT node is a single basic block.
    """
    funcs = {}
    cur = None
    node_re = re.compile(r'^"([^"]+)"$')
    label_re = re.compile(r'\[label="\{(.*?)\}"\]', re.DOTALL)
    edge_re = re.compile(r'^"([^"]+)":[\w]+:[\w] -> "([^"]+)":[\w]+:[\w]')

    lines = dot_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^subgraph "cluster_(.+)" \{', line)
        if m:
            cur = m.group(1)
            funcs[cur] = {"bbs": {}, "edges": []}
            i += 1
            continue
        m = edge_re.match(line)
        if m and cur:
            funcs[cur]["edges"].append((m.group(1), m.group(2)))
            i += 1
            continue
        m = node_re.match(line.strip())
        if m and cur:
            node_id = m.group(1)
            # label may span many lines; collect until we close [label="..."]
            j = i + 1
            buf = []
            while j < len(lines):
                buf.append(lines[j])
                if lines[j].rstrip().endswith('"]'):
                    break
                j += 1
            lm = label_re.search("\n".join(buf))
            if lm:
                insts = []
                for raw in lm.group(1).split(r"\l"):
                    raw = unescape_dot(raw.strip())
                    im = INST_RE.match(raw)
                    if im:
                        insts.append({"pc": int(im.group(1), 16),
                                      "sass": im.group(2)})
                if insts:
                    funcs[cur]["bbs"][node_id] = {"insts": insts}
            i = j + 1
            continue
        i += 1

    # drop nodes that had no instructions, and edges touching them
    for f in funcs.values():
        f["edges"] = [(a, b) for a, b in f["edges"]
                      if a in f["bbs"] and b in f["bbs"]]
    return funcs


# ------------------------------------------------------- live registers

def parse_life_ranges(text):
    """Parse `nvdisasm -c -poff -plr -lrm narrow` output.

    Returns {func_name: {pc: {"gpr": {reg: sym}, "pred": {...}, "ugpr": {...}}}}
    where sym is one of `^` (range starts: def), `:` (live through),
    `v` (range ends: last use), `x` (ends and restarts at this instruction).
    Register ids come from the vertical digit header; the `#` count columns
    carry no header digits and are skipped automatically.
    """
    out = defaultdict(dict)
    cur = None
    group_spans, digit_rows, colmap = None, [], None
    sec_re = re.compile(r"\.section\s+\.text\.([^\s,]+)")
    pc_re = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/")

    for raw in text.splitlines():
        m = sec_re.search(raw)
        if m:
            cur = m.group(1)
            group_spans, digit_rows, colmap = None, [], None
            continue
        if cur is None:
            continue
        ci = raw.find("//")
        if ci < 0:
            continue
        seg = raw[ci + 2:]

        if colmap is None:
            words = set(seg.replace("|", " ").split())
            if words and "GPR" in words and \
                    all(re.fullmatch(r"[A-Z]+", w) for w in words):
                pipes = [i for i, ch in enumerate(seg) if ch == "|"]
                group_spans = [(a + 1, b, seg[a + 1:b].strip())
                               for a, b in zip(pipes, pipes[1:])
                               if seg[a + 1:b].strip()]
                continue
            if group_spans and re.fullmatch(r"[0-9 #|]*", seg.rstrip()) \
                    and any(c.isdigit() for c in seg):
                digit_rows.append(seg)
                continue

        m = pc_re.match(raw)
        if not m:
            continue
        if colmap is None:
            if not digit_rows or not group_spans:
                continue
            width = max(len(r) for r in digit_rows)
            colmap = []
            for c in range(width):
                digs = "".join(r[c] for r in digit_rows
                               if c < len(r) and r[c].isdigit())
                if digs:
                    grp = next((n for a, b, n in group_spans if a <= c < b), None)
                    if grp:
                        colmap.append((c, grp.lower(), int(digs)))
        pc = int(m.group(1), 16)
        rec = {}
        for c, grp, rid in colmap:
            ch = seg[c] if c < len(seg) else " "
            if ch in "^:vx":
                rec.setdefault(grp, {})[rid] = ch
        out[cur][pc] = rec
    return dict(out)


# ----------------------------------------------------------- line info

def parse_line_info(text):
    """Parse `nvdisasm -c -poff -g` output.

    Returns {func_name: {pc: (file, line)}}. Requires the binary to be
    compiled with -lineinfo; returns empty maps otherwise.
    """
    out = defaultdict(dict)
    cur, cur_src = None, None
    sec_re = re.compile(r"\.section\s+\.text\.([^\s,]+)")
    file_re = re.compile(r'//##\s+File\s+"([^"]+)",\s+line\s+(\d+)')
    inst_re = re.compile(r"/\*([0-9a-fA-F]+)\*/")

    for line in text.splitlines():
        m = sec_re.search(line)
        if m:
            cur, cur_src = m.group(1), None
            continue
        m = file_re.search(line)
        if m:
            cur_src = (m.group(1), int(m.group(2)))
            continue
        m = inst_re.search(line)
        if m and cur and cur_src:
            out[cur][int(m.group(1), 16)] = cur_src
    return dict(out)


# ------------------------------------------------------------------ NCU

def profile(binary, app_args, rep_path, extra_ncu_args):
    cmd = [os.path.join(CUDA_BIN, "ncu"),
           "--section", "SourceCounters",
           "--section", "LaunchStats",
           "--section", "Occupancy",
           "-f", "-o", rep_path.removesuffix(".ncu-rep")]
    cmd += extra_ncu_args
    cmd += [binary] + app_args
    r = run(cmd, check=False)
    if not os.path.exists(rep_path):
        sys.exit(f"ncu profiling failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def _num(s):
    s = s.replace(",", "").strip()
    if s in ("", "-", "n/a"):
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def parse_source_page(rep_path, launch_counts):
    """Export + parse per-instruction metrics from the NCU source page.

    Returns {demangled_kernel_name: {offset: {metrics...}}} aggregated over
    all launches of the same kernel. The source-page CSV may emit the same
    table more than once per launch (duplicate views); the true launch count
    from the details page is used to rescale the sums.
    """
    r = run([os.path.join(CUDA_BIN, "ncu"), "--import", rep_path,
             "--page", "source", "--csv"])
    kernels = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    blocks_seen = defaultdict(int)

    blocks = re.split(r'(?m)^"Kernel Name",', r.stdout)
    for block in blocks[1:]:
        lines = block.splitlines()
        kname = next(csv.reader(io.StringIO(lines[0])))[0]
        blocks_seen[kname] += 1
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
        addrs = [int(row["Address"], 16) for row in rows if row.get("Address")]
        if not addrs:
            continue
        base = min(addrs)
        for row in rows:
            if not row.get("Address"):
                continue
            off = int(row["Address"], 16) - base
            tgt = kernels[kname][off]
            tgt["sass"] = row.get("Source", "").strip()
            for col, key in [("Instructions Executed", "inst_executed"),
                             ("Thread Instructions Executed", "thread_inst_executed"),
                             ("Divergent Branches", "divergent_branches")]:
                v = _num(row.get(col, ""))
                if v is not None:
                    tgt[key] += v
            v = _num(row.get("Avg. Predicated-On Threads Executed", ""))
            if v is not None:
                # weight by this launch's inst_executed later; keep sum of products
                ie = _num(row.get("Instructions Executed", "")) or 0
                tgt["_pred_on_x_inst"] += v * ie

    out = {}
    for kname, per_off in kernels.items():
        n_launch = norm_lookup(launch_counts, kname, 1) or 1
        dup = blocks_seen[kname] // n_launch if blocks_seen[kname] % n_launch == 0 else 1
        if dup > 1:
            print(f"[ncu] {kname}: {blocks_seen[kname]} source blocks for "
                  f"{n_launch} launch(es) — rescaling counts by 1/{dup}")
        entry = {}
        for off, m in per_off.items():
            ie = m.get("inst_executed", 0) / dup
            tie = m.get("thread_inst_executed", 0) / dup
            entry[off] = {
                "sass_ncu": m.get("sass", ""),
                "inst_executed": int(ie),
                "thread_inst_executed": int(tie),
                "avg_active_lanes": round(tie / ie, 3) if ie else None,
                "avg_pred_on_lanes": round(m["_pred_on_x_inst"] / dup / ie, 3) if ie else None,
                "divergent_branches": int(m.get("divergent_branches", 0) / dup),
            }
        out[kname] = {"per_pc": entry, "launches": n_launch}
    return out


OCC_METRICS = {
    "Theoretical Occupancy": "theoretical_pct",
    "Achieved Occupancy": "achieved_pct",
    "Block Limit Registers": "block_limit_registers",
    "Block Limit Shared Mem": "block_limit_shared_mem",
    "Block Limit Warps": "block_limit_warps",
    "Block Limit Barriers": "block_limit_barriers",
    "Block Limit SM": "block_limit_sm",
}


def parse_launch_stats(rep_path):
    """Return ({kernel: regs_per_thread}, {kernel: launch_count},
    {kernel: occupancy dict}).

    The details page emits one "Registers Per Thread" row per profiled
    launch, which makes it the authoritative launch count. Occupancy rows
    include the block-limit breakdown; the minimum limit is the binding
    occupancy limiter.
    """
    r = run([os.path.join(CUDA_BIN, "ncu"), "--import", rep_path,
             "--page", "details", "--csv"])
    regs, launches, occ = {}, defaultdict(int), defaultdict(dict)
    for row in csv.reader(io.StringIO(r.stdout)):
        kname = row[4] if len(row) > 4 else None
        if "Registers Per Thread" in row:
            launches[kname] += 1
            try:
                regs[kname] = int(float(row[row.index("Registers Per Thread") + 2]))
            except (ValueError, IndexError):
                pass
        for metric, key in OCC_METRICS.items():
            if metric in row:
                try:
                    occ[kname][key] = float(row[row.index(metric) + 2])
                except (ValueError, IndexError):
                    pass
    for kname, o in occ.items():
        limits = {k.removeprefix("block_limit_"): v for k, v in o.items()
                  if k.startswith("block_limit_")}
        if limits:
            o["limiter"] = min(limits, key=limits.get)
            o["registers_are_limiter"] = o["limiter"] == "registers"
    return regs, dict(launches), dict(occ)


# ------------------------------------------------- kernel name matching

def demangle(name):
    for tool in ("cu++filt", "c++filt"):
        path = shutil.which(tool) or os.path.join(CUDA_BIN, tool)
        if os.path.exists(path):
            r = run([path], input=name, check=False)
            d = r.stdout.strip()
            if d and d != name:
                return d
    return name


def norm_sig(sig):
    """Normalize a demangled signature across demangler dialects:
    'const float *' == 'float const*', '(bool)1' == '1', 'std::' dropped
    (NCU's details and source pages demangle differently)."""
    s = re.sub(r"\bconst\s+(\w+)", r"\1 const", sig)
    s = s.replace("std::", "")
    s = re.sub(r"\((?:bool|char|short|int|long|unsigned \w+)\)", "", s)
    return re.sub(r"\s+", "", s)


def norm_lookup(d, name, default=None):
    """Fetch d[name] tolerating demangler-dialect differences in keys."""
    if name in d:
        return d[name]
    target = norm_sig(name)
    for k, v in d.items():
        if norm_sig(k) == target:
            return v
    base = name.split("(")[0]
    hits = [v for k, v in d.items() if k.split("(")[0] == base]
    return hits[0] if len(hits) == 1 else default


def match_kernel(mangled, ncu_names):
    dem = demangle(mangled)
    base = dem.split("(")[0]
    exact = [n for n in ncu_names if norm_sig(n) == norm_sig(dem)]
    if exact:
        return exact[0]
    by_base = [n for n in ncu_names if n.split("(")[0] == base]
    if len(by_base) == 1:
        return by_base[0]
    return None


# ---------------------------------------------------------------- merge

WARP_SIZE = 32


def merge(funcs, live, lineinfo, ncu_kernels, regs_per_thread, occupancy):
    result = {}
    for fname, cfg in funcs.items():
        ncu_name = match_kernel(fname, list(ncu_kernels.keys()))
        per_pc = ncu_kernels[ncu_name]["per_pc"] if ncu_name else {}
        flive = live.get(fname, {})
        flines = lineinfo.get(fname, {})
        unmatched = set(per_pc.keys())

        bbs = {}
        for node_id, bb in cfg["bbs"].items():
            insts = []
            for inst in bb["insts"]:
                pc = inst["pc"]
                rec = {"pc": pc, "pc_hex": f"{pc:04x}", "sass": inst["sass"]}
                lv = flive.get(pc)
                if lv is not None:
                    for grp in ("gpr", "pred", "ugpr", "upred"):
                        rec[f"live_{grp}"] = len(lv.get(grp, {}))
                if pc in flines:
                    rec["file"], rec["line"] = flines[pc]
                if pc in per_pc:
                    m = per_pc[pc]
                    rec.update({k: v for k, v in m.items() if k != "sass_ncu"})
                    unmatched.discard(pc)
                insts.append(rec)

            # register flow across this BB, from the per-PC liveness symbol
            # maps: live-in inherited from predecessors, live-out handed to
            # successors, live-through = held across the whole block without
            # any local def/use (pressure imposed purely by up/downstream).
            first_lv = flive.get(insts[0]["pc"], {})
            last_lv = flive.get(insts[-1]["pc"], {})
            gpr_in = sorted(r for r, s in first_lv.get("gpr", {}).items() if s in ":vx")
            gpr_out = sorted(r for r, s in last_lv.get("gpr", {}).items() if s in ":^x")
            thru = set(gpr_in)
            for r in bb["insts"]:
                lv = flive.get(r["pc"])
                if lv is None:
                    thru = set()
                    break
                thru &= {reg for reg, s in lv.get("gpr", {}).items() if s == ":"}
            gpr_thru = sorted(thru)

            ie = [r.get("inst_executed") for r in insts if r.get("inst_executed") is not None]
            tie = [r.get("thread_inst_executed") for r in insts if r.get("thread_inst_executed") is not None]
            avg_lanes = round(sum(tie) / sum(ie), 3) if ie and sum(ie) else None

            # predicated-on lane density, weighted by per-inst execution count
            pw = [(r["avg_pred_on_lanes"], r["inst_executed"]) for r in insts
                  if r.get("avg_pred_on_lanes") is not None and r.get("inst_executed")]
            avg_pred_on = round(sum(p * w for p, w in pw) / sum(w for _, w in pw), 3) if pw else None

            # source-line histogram: "file:line" -> static/dynamic share
            line_hist = defaultdict(lambda: {"n_insts": 0, "inst_executed": 0})
            for r in insts:
                if "line" in r:
                    key = f"{os.path.basename(r['file'])}:{r['line']}"
                    line_hist[key]["n_insts"] += 1
                    line_hist[key]["inst_executed"] += r.get("inst_executed") or 0
            dominant = max(line_hist.items(),
                           key=lambda kv: (kv[1]["inst_executed"], kv[1]["n_insts"]),
                           default=(None, None))[0]

            bbs[node_id] = {
                "start_pc": insts[0]["pc"], "end_pc": insts[-1]["pc"],
                "n_insts": len(insts),
                "exec_count": max(ie) if ie else None,   # (sub)warp-level executions
                "inst_executed_total": sum(ie) if ie else None,
                "thread_inst_total": sum(tie) if tie else None,
                "avg_active_lanes": avg_lanes,
                "avg_pred_on_lanes": avg_pred_on,
                "divergence": round(1 - avg_lanes / WARP_SIZE, 3) if avg_lanes else None,
                "max_live_gpr": max((r.get("live_gpr", 0) for r in insts), default=0),
                "sum_live_gpr": sum(r.get("live_gpr", 0) for r in insts),
                "live_in_gpr": len(gpr_in), "live_out_gpr": len(gpr_out),
                "live_through_gpr": len(gpr_thru),
                "gpr_live_in": gpr_in, "gpr_live_out": gpr_out,
                "gpr_live_through": gpr_thru,
                "src_dominant": dominant,
                "src_lines": dict(sorted(line_hist.items())),
                "insts": insts,
            }

        # padding NOPs / unreachable tail bytes report as zero-executed rows
        unmatched_hot = sorted(p for p in unmatched
                               if per_pc[p]["inst_executed"] > 0)
        result[fname] = {
            "kernel": ncu_name or demangle(fname),
            "mangled": fname,
            "profiled": ncu_name is not None,
            "launches": ncu_kernels[ncu_name]["launches"] if ncu_name else 0,
            "registers_per_thread": norm_lookup(regs_per_thread, ncu_name) if ncu_name else None,
            "occupancy": norm_lookup(occupancy, ncu_name) if ncu_name else None,
            "bbs": bbs,
            "edges": cfg["edges"],
            "unmatched_ncu_pcs": sorted(unmatched),
            "unmatched_executed_pcs": unmatched_hot,
        }
    return result


# --------------------------------------------------------------- output

def heat_color(frac):
    """white -> yellow -> red ramp for relative hotness."""
    if frac is None:
        return "#eeeeee"
    r = 255
    g = int(255 - 120 * frac)
    b = int(235 * (1 - frac))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_dot(fname, data, path):
    bbs, edges = data["bbs"], data["edges"]
    totals = [b["inst_executed_total"] for b in bbs.values()
              if b["inst_executed_total"]]
    peak = max(totals) if totals else None
    with open(path, "w") as f:
        f.write(f'digraph "{data["kernel"]}" {{\n')
        f.write('  node [shape=box, fontname="Courier", fontsize=10, style=filled];\n')
        f.write(f'  label="{data["kernel"]}\\nregisters/thread: '
                f'{data["registers_per_thread"]}  launches: {data["launches"]}";\n')
        for node_id, bb in bbs.items():
            frac = (bb["inst_executed_total"] / peak) if (peak and bb["inst_executed_total"]) else None
            lines = [f"{node_id}  [{bb['start_pc']:04x}-{bb['end_pc']:04x}]",
                     f"insts: {bb['n_insts']}"]
            if bb.get("src_dominant"):
                extra = len(bb["src_lines"]) - 1
                src = bb["src_dominant"] + (f" (+{extra})" if extra else "")
                lines.append(f"src: {src}")
            if bb["exec_count"] is not None:
                lines.append(f"warp execs: {bb['exec_count']:,}")
                lines.append(f"avg lanes: {bb['avg_active_lanes']}"
                             f"  div: {bb['divergence']}")
            lines.append(f"GPR in/thru/peak/out: {bb['live_in_gpr']}/"
                         f"{bb['live_through_gpr']}/{bb['max_live_gpr']}/"
                         f"{bb['live_out_gpr']}")
            # header centered via \n, then one left-justified line per instruction
            label = "\\n".join(lines) + "\\n"
            for inst in bb["insts"]:
                sass = inst["sass"].replace("\\", "\\\\").replace('"', '\\"')
                label += f"{inst['pc']:04x}:  {sass}\\l"
            f.write(f'  "{node_id}" [fillcolor="{heat_color(frac)}", '
                    f'label="{label}"];\n')
        for a, b in edges:
            f.write(f'  "{a}" -> "{b}";\n')
        f.write("}\n")


def write_csv(fname, data, path):
    cols = ["bb", "pc_hex", "sass", "file", "line",
            "inst_executed", "thread_inst_executed",
            "avg_active_lanes", "avg_pred_on_lanes", "divergent_branches",
            "live_gpr", "live_pred", "live_ugpr"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for node_id, bb in data["bbs"].items():
            for inst in bb["insts"]:
                w.writerow([node_id] + [inst.get(c, "") for c in cols[1:]])


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("binary", help="CUDA executable (or a .cubin)")
    ap.add_argument("app_args", nargs="*", help="arguments passed to the app when profiling")
    ap.add_argument("--ncu-rep", help="reuse an existing .ncu-rep instead of profiling")
    ap.add_argument("--ncu-args", default="", help="extra args for ncu (quoted string)")
    ap.add_argument("--kernel", help="only process functions whose name matches this regex")
    ap.add_argument("-o", "--outdir", default="gcfg_out")
    args = ap.parse_args()

    binary = os.path.abspath(args.binary)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # 1) cubins
    cubins = []
    if binary.endswith(".cubin"):
        cubins = [binary]
    else:
        xdir = tempfile.mkdtemp(prefix="gcfg_cubin_", dir=outdir)
        run([os.path.join(CUDA_BIN, "cuobjdump"), "-xelf", "all", binary], cwd=xdir)
        cubins = sorted(os.path.join(xdir, f) for f in os.listdir(xdir)
                        if f.endswith(".cubin"))
        if not cubins:
            sys.exit("no cubin extracted — is this a CUDA binary with embedded ELF?")

    # 2) static CFG + live ranges per cubin
    funcs, live, lineinfo = {}, {}, {}
    for cb in cubins:
        try:
            dot = run([os.path.join(CUDA_BIN, "nvdisasm"), "-bbcfg", "-poff", cb]).stdout
            funcs.update(parse_bbcfg(dot))
        except subprocess.CalledProcessError:
            print(f"[static] WARNING: nvdisasm -bbcfg failed on {os.path.basename(cb)}, skipping")
            continue
        try:
            plr = run([os.path.join(CUDA_BIN, "nvdisasm"), "-c", "-poff",
                       "-plr", "-lrm", "narrow", cb]).stdout
            live.update(parse_life_ranges(plr))
        except subprocess.CalledProcessError:
            print(f"[static] WARNING: -plr failed on {os.path.basename(cb)} "
                  f"(no liveness for its kernels)")
        li = run([os.path.join(CUDA_BIN, "nvdisasm"), "-c", "-poff", "-g", cb],
                 check=False).stdout
        lineinfo.update(parse_line_info(li))
    if not any(lineinfo.values()):
        print("[static] no line info found (compile with -lineinfo for "
              "source mapping)")
    if args.kernel:
        funcs = {k: v for k, v in funcs.items()
                 if re.search(args.kernel, k) or re.search(args.kernel, demangle(k))}
    if not funcs:
        sys.exit("no matching functions found in cubin(s)")
    print(f"[static] {len(funcs)} function(s): {', '.join(funcs)}")

    # 3) profile / import
    rep = args.ncu_rep or os.path.join(outdir, "profile.ncu-rep")
    if not args.ncu_rep:
        print("[ncu] profiling (this replays kernels several times)...")
        profile(binary, args.app_args, rep, args.ncu_args.split())
    regs, launch_counts, occ = parse_launch_stats(rep)
    ncu_kernels = parse_source_page(rep, launch_counts)
    print(f"[ncu] {len(ncu_kernels)} profiled kernel(s): {', '.join(ncu_kernels)}")

    # 4) merge + emit
    merged = merge(funcs, live, lineinfo, ncu_kernels, regs, occ)
    for fname, data in merged.items():
        safe = re.sub(r"[^\w.]+", "_", fname)[:120]
        with open(os.path.join(outdir, f"{safe}.json"), "w") as f:
            json.dump(data, f, indent=1)
        write_dot(fname, data, os.path.join(outdir, f"{safe}.dot"))
        write_csv(fname, data, os.path.join(outdir, f"{safe}.csv"))
        n_pc = sum(bb["n_insts"] for bb in data["bbs"].values())
        n_hit = sum(1 for bb in data["bbs"].values()
                    for i in bb["insts"] if i.get("inst_executed") is not None)
        status = f"{n_hit}/{n_pc} PCs matched to NCU data" if data["profiled"] \
            else "not profiled (no NCU match)"
        if data["unmatched_executed_pcs"]:
            status += (f"  WARNING: {len(data['unmatched_executed_pcs'])} executed "
                       f"NCU PCs not in CFG")
        elif data["unmatched_ncu_pcs"]:
            status += (f"  ({len(data['unmatched_ncu_pcs'])} zero-executed NCU PCs "
                       f"outside CFG: tail padding)")
        print(f"[merge] {data['kernel']}: {len(data['bbs'])} BBs, {status}")
    print(f"[done] outputs in {outdir}/  (*.json, *.dot, *.csv)")


if __name__ == "__main__":
    main()
