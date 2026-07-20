#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <meta> <rank-count> <alternate-root>" >&2
    exit 2
fi

META=$1
RANK_COUNT=$2
ALT_ROOT=$3

run_case() {
    local root=$1
    local bytes=$2
    local iterations=${3:-1}
    bash "${SCRIPT_DIR}/run_hvm_case.sh" \
        "${META}" "${RANK_COUNT}" "${root}" "${bytes}" checker "${iterations}"
}

# hccl-vm does not reliably recycle all simulated communication resources
# between independent mpirun commands. Give every matrix entry a fresh HVM
# session so a failed communicator setup cannot contaminate later cases.
for root in 0 "${ALT_ROOT}"; do
    run_case "${root}" 524288
    run_case "${root}" 536870912
    run_case "${root}" 419430404
done
for bytes in 268435452 268435456 268435460 419430400; do
    run_case 0 "${bytes}"
done
run_case 0 524288 20

echo "PASS full checker matrix meta=${META} ranks=${RANK_COUNT} alternate-root=${ALT_ROOT}"
