#!/bin/bash
# Slingshot (cxi) environment for MULTI-NODE NCCL on LANTA.  Source before srun.
#
# TWO things make multi-node NCCL work on LANTA (both non-obvious):
#  1. The batch job MUST request a network VNI:  #SBATCH --network=job_vni,single_node_vni
#     Without it, cxi cannot open a data domain -> "Device or resource busy" / hang.
#     (That line lives in the .sbatch file, not here.)
#  2. The fabric env vars below (HPE + NERSC/CSCS recommended for Slingshot+A100).
#
# Load AFTER: module load cuda/12.6 ; module load aws-ofi-nccl/1.6.0+cuda12.6
# and after putting the CUDA-12 NCCL 2.18.5 on LD_LIBRARY_PATH.

SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl
export LD_LIBRARY_PATH=$SDKNCCL/lib:$LD_LIBRARY_PATH

# --- libfabric / cxi tuning ---
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
# Use ONE cxi NIC (cxi0). The 2nd NIC (cxi1) returns EBUSY on this system.
export FI_CXI_DEVICE_NAME=cxi0

# --- NCCL ---
export NCCL_NET=OFI                # use the aws-ofi-nccl (Slingshot) transport
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PHB       # GPUDirect RDMA
export NCCL_SOCKET_IFNAME=hsn0      # bootstrap over the Slingshot host NIC
export NCCL_PROTO=^LL128            # LL128 performs poorly on Slingshot
