#!/bin/bash
set -euo pipefail

toolchain="${1:-cpeCray/25.03}"
safe_toolchain="${toolchain//\//-}"

if ! type module >/dev/null 2>&1; then
    echo "The 'module' command is not available. Run this script on LANTA login or compute nodes." >&2
    exit 1
fi

module purge
module load "${toolchain}"
module list

./scripts/record_context.sh
make print-env CXX=CC BUILD_DIR="build/${safe_toolchain}"
make CXX=CC BUILD_DIR="build/${safe_toolchain}"

echo "Built examples in build/${safe_toolchain}"
