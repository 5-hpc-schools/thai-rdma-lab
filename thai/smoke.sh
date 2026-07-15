#!/bin/bash
# smoke.sh -- machine-checkable assertions for the Day-3 NCCL lab.
# Run ON LANTA after Part A/B/C jobs have produced results/*.out.
#   bash smoke.sh [results_dir]
# One essential, exit-code assertion per part. Exit 0 = all pass.
set -u
R=${1:-/project/tn999992-rdma/day3-nccl-lab/results}
fail=0
say(){ echo "[smoke] $*"; }

# newest output per part
A=$(ls -t "$R"/partA_*.out 2>/dev/null | head -1)
B=$(ls -t "$R"/partB_*.out 2>/dev/null | head -1)
C=$(ls -t "$R"/partC_*.out 2>/dev/null | head -1)

# helper: max busbw (col 11, in-place) across data rows of a file
maxbw(){ awk '/float +sum/{v=$11; if(v>m)m=v} END{printf "%.2f", m+0}' "$1" 2>/dev/null; }

# A: intra-node NVLink -> peak busbw must be large (>50 GB/s) and log shows NVL
abw=$(maxbw "$A")
if [ -n "$A" ] && awk "BEGIN{exit !($abw>50)}" && grep -q "NVL" "$A"; then
  say "PASS A intra-node NVLink: peak busbw=${abw} GB/s"
else say "FAIL A ($A) peak=${abw} (need >50 and 'NVL' in log)"; fail=1; fi

# B: inter-node Slingshot RDMA -> log picks cxi AND a real busbw row (>3 GB/s)
bbw=$(maxbw "$B")
if [ -n "$B" ] && grep -q "Selected provider is cxi" "$B" && awk "BEGIN{exit !($bbw>3)}"; then
  say "PASS B inter-node cxi RDMA: peak busbw=${bbw} GB/s"
else say "FAIL B ($B) peak=${bbw} (need cxi + busbw>3)"; fail=1; fi

# C: inter-node TCP completes (has a busbw row)
cbw=$(maxbw "$C")
if [ -n "$C" ] && awk "BEGIN{exit !($cbw>0)}" && grep -q "NCCL_NET.*Socket\|Using network Socket" "$C"; then
  say "PASS C inter-node TCP: peak busbw=${cbw} GB/s"
else say "FAIL C ($C) peak=${cbw} (need Socket + busbw>0)"; fail=1; fi

# Cross-check: RDMA must beat TCP (the lab's whole point)
if awk "BEGIN{exit !($bbw>$cbw)}"; then
  say "PASS contrast: RDMA ${bbw} > TCP ${cbw} GB/s ($(awk "BEGIN{printf \"%.1f\", $bbw/$cbw}")x)"
else say "FAIL contrast: RDMA ${bbw} !> TCP ${cbw}"; fail=1; fi

[ $fail -eq 0 ] && say "ALL PASS" || say "SOME FAILED"
exit $fail
