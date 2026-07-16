Day 3 — NCCL / RDMA Hands-on Lab  (LANTA / ThaiSC, A100 + Slingshot)
====================================================================

Open the .html files in ANY web browser (just double-click). Nothing to install.
All student text is simple English. Verified on LANTA A100 nodes, 2026-07-16.


FOR THE LECTURER
----------------
  SLIDES.html ............. the talk, 42 slides.
                            • arrow keys / click left-right to move
                            • top-right button = dark / light theme
                            • Ctrl-P = save as PDF
  INSTRUCTOR.md ........... run-of-show (half day), answer key, expected numbers,
                            and the multi-node queue plan. READ THIS FIRST.


FOR STUDENTS  (hand these out — laptop friendly / printable)
------------------------------------------------------------
  worksheet/LAB.html .......... the main lab, Parts A–F. Do it in order.
  worksheet/CHEATSHEET.html ... the only 4 commands they need.
  worksheet/FIELDGUIDE.html ... read a real NCCL log like a pro (deep dive, optional).
  worksheet/PYTORCH_RDMA.html . how Python (PyTorch DDP) ends up using RDMA.
  (.md files next to each .html are the editable sources.)


CODE  (also on LANTA at  /project/tn999992-rdma/day3-nccl-lab/lab/ )
--------------------------------------------------------------------
  lab/partA_1node.sbatch ......... Part A  intra-node NVLink
  lab/partB_multinode_rdma.sbatch  Part B  inter-node Slingshot RDMA
  lab/partC_multinode_tcp.sbatch . Part C  inter-node TCP (the slow contrast)
  lab/partD_explore.sbatch ....... Part D  protocols / ring vs tree / topology
  lab/partF_deepdive.sbatch ...... Part F  full NCCL_DEBUG boot log
  lab/nccl_hello.cu (+ solution) . Part E  fill-in-the-blank NCCL program
  lab/build_and_run_hello.sh ..... Part E  build+run helper
  lab/ddp_demo.py ................ the PyTorch DDP demo (Python -> NCCL -> RDMA)

  Students first copy the lab into their own space on LANTA:
      cp -r /project/tn999992-rdma/day3-nccl-lab/lab  ~/nccl-lab


REAL LOGS  (show these live while you teach)
--------------------------------------------
  logs/topo_4gpu.xml ....... the machine map NCCL drew
  logs/graph_4gpu.xml ...... the 12 rings NCCL planned
  logs/*keylines*.txt ...... key lines of real verbose runs (1-node & 2-node)
  logs/README.md ........... a lecturer tour of the logs


REBUILD / VERIFY  (for admins only)
-----------------------------------
  build_nccl_tests.sh ...... how the working NCCL benchmark was built on LANTA
  env_slingshot.sh ......... the Slingshot (cxi) environment for multi-node runs
  smoke.sh ................. re-checks the whole lab (parses real job output)


EDIT & REGENERATE
-----------------
  Slides:  python3 slides.py SLIDES.md "title" > SLIDES.html
  Docs:    python3 render.py worksheet/LAB.md "title" > worksheet/LAB.html
  (slides.py / render.py are tiny, zero-dependency, offline.)
