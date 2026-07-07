#!/bin/bash
set -euo pipefail

echo "== Reproducibility context =="
date -Is
echo "pwd $(pwd)"
echo "host $(hostname)"
echo "user ${USER:-unknown}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit $(git log --oneline -1)"
    echo "git_status"
    git status --short
fi

echo "slurm_job_id ${SLURM_JOB_ID:-unset}"
echo "slurm_job_name ${SLURM_JOB_NAME:-unset}"
echo "slurm_partition ${SLURM_JOB_PARTITION:-unset}"
echo "slurm_nodes ${SLURM_JOB_NUM_NODES:-unset}"
echo "slurm_ntasks ${SLURM_NTASKS:-unset}"
echo "slurm_tasks_per_node ${SLURM_TASKS_PER_NODE:-unset}"
echo "slurm_cpus_per_task ${SLURM_CPUS_PER_TASK:-unset}"
echo "slurm_submit_dir ${SLURM_SUBMIT_DIR:-unset}"

if command -v module >/dev/null 2>&1; then
    echo "module_list"
    module list 2>&1 || true
fi

if command -v cc >/dev/null 2>&1; then
    echo "cc_version"
    cc --version 2>&1 | head -5 || true
    echo "cc_cray_options"
    cc --cray-print-opts=all 2>&1 || true
fi

if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    echo "scontrol_show_job"
    scontrol show job "${SLURM_JOB_ID}" || true
fi

echo "== End reproducibility context =="
