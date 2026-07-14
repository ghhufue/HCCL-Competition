#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_LOG="${1:-${PROJECT_DIR}/artifacts/hccl_vm_perf/broadcast_root0_4.log}"
OUTPUT_LOG="${2:-${INPUT_LOG%.log}.errors.log}"

if [[ ! -f "${INPUT_LOG}" ]]; then
    echo "input log not found: ${INPUT_LOG}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_LOG}")"

{
    printf 'source: %s\n' "${INPUT_LOG}"
    printf 'generated_at: %s\n' "$(date -Iseconds)"
    printf '\n'
    awk '
        BEGIN { IGNORECASE = 0 }
        /total err/ ||
        /check result failed/ ||
        /HcclBroadcast call trace/ ||
        /Failed to execute op/ ||
        /call trace: hcclRet/ ||
        /HCCL_E_/ ||
        /\[ERROR\]/ ||
        /\[error\]/ ||
        /FATAL/ ||
        /fatal/ ||
        /ret\[[1-9][0-9]*\]/ ||
        /ret -> [1-9][0-9]*/ ||
        /Hcomm[A-Za-z0-9_]* failed/ ||
        / local rank_id .* failed/ {
            printf "%d:%s\n", NR, $0
        }
    ' "${INPUT_LOG}"
} > "${OUTPUT_LOG}"

echo "wrote: ${OUTPUT_LOG}"
