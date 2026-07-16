# Real NCCL logs & dumps (use these live in the lecture)

These are **real files captured from LANTA A100 nodes** on 2026-07-15/16. Open them on the
projector and point at the lines while you teach. They match the Field Guide section by section.

| file | what it is | how it was made |
|------|-----------|-----------------|
| `topo_4gpu.xml` | the machine map NCCL drew (CPUs → PCI → GPUs/NICs, NVLinks) | `NCCL_TOPO_DUMP_FILE` |
| `graph_4gpu.xml` | the 12 rings NCCL planned (each `<channel>` = one ring order) | `NCCL_GRAPH_DUMP_FILE` |
| `nccl_verbose_1node_keylines.txt` | the important lines of a 1-node run, with line numbers | `NCCL_DEBUG=INFO SUBSYS=ALL` |
| `nccl_verbose_1node_excerpt.txt` | the first 60 lines (bootstrap + version) | same |
| `nccl_verbose_2node_keylines.txt` | 2-node run: bootstrap over hsn0, `cxi`, GDR-by-distance | same, 2 nodes |

The **full** 1-node log (~7500 lines) and 2-node log live on LANTA under
`/project/tn999992-rdma/day3-nccl-lab/results/cap1_*.out` and `cap2_*.out` — grep them live:

```bash
grep -n "=== System"    results/cap1_*.out     # the topology summary
grep -n "Ring 0"        results/cap1_*.out     # the ring orders
grep -n "AllReduce |"   results/cap1_*.out     # the tuning cost table
grep -n "Selected provider" results/cap2_*.out # cxi chosen on 2 nodes
```

Regenerate any of these yourself: `sbatch scripts/xml.sbatch` (dumps XML) or
`sbatch lab/partF_deepdive.sbatch` (full verbose log).

## Quick tour for the lecturer
1. Open `topo_4gpu.xml` → show one `<gpu>` with three `<nvlink count="4">` = all-to-all, 80 GB/s.
2. Open `graph_4gpu.xml` → count the `<channel>` blocks = 12 rings.
3. Open `nccl_verbose_1node_keylines.txt` → walk the 5 boot steps (Field Guide §2–§11).
4. Open `nccl_verbose_2node_keylines.txt` → show `Bootstrap : Using hsn0`, `Selected provider is
   cxi`, and the honest `GPU Direct RDMA Disabled ... (distance 7 > 6)` line.
