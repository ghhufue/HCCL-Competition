#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/hvm_session_lib.sh"

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <meta> <rank-count> <alternate-root>" >&2
    exit 2
fi

META=$1
RANK_COUNT=$2
ALT_ROOT=$3
ASCEND_HOME_PATH=/home/workspace/hvm/Ascend/cann-9.1.0
HCCL_VM_HOME=/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install
CUSTOM_SO=/mnt/d/code/HCCL-Competition/Hccl_Broadcast_Final/build/lib64/libhccl.so
ROOT_SHIM=/mnt/d/code/HCCL-Competition/artifacts/libdynamic_root_shim.so
TEST_BIN=${ASCEND_HOME_PATH}/tools/hccl_test/bin/broadcast_test
LOG_FILE=/mnt/d/code/HCCL-Competition/artifacts/hvm_matrix/${META}_n${RANK_COUNT}_dynamic_root.log

set +u
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${HCCL_VM_HOME}/script/hccl_config.sh"
set -u
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${META}" == 141 ]]; then
    NPUS_PER_SERVER=1
else
    NPUS_PER_SERVER=8
fi

bash "${SCRIPT_DIR}/build_dynamic_root_shim.sh"
hvm_session_start "${LOG_FILE}" "${HCCL_VM_HOME}/bin/hccl-vm" start \
    ascend950_cluster_4_server_competition
printf 'hccl-vm plugin install @runner\n' >&3
printf 'hccl-vm mock-comm %s\n' "${META}" >&3
if [[ "${RANK_COUNT}" -gt 8 ]]; then
    printf 'python3 /mnt/d/code/HCCL-Competition/artifacts/normalize_hvm_ranktable.py\n' >&3
fi
printf 'env ASCEND_GLOBAL_LOG_LEVEL=3 HCCL_TEST_ALTERNATE_ROOT=%s LD_PRELOAD=%s:${LD_PRELOAD:+${LD_PRELOAD}:}%s mpirun --allow-run-as-root --oversubscribe -np %s %s -b 524288 -e 524288 -d fp32 -r 0 -w 0 -n 3 -c 1 -p %s </dev/null\n' \
    "${ALT_ROOT}" "${ROOT_SHIM}" "${CUSTOM_SO}" "${RANK_COUNT}" "${TEST_BIN}" "${NPUS_PER_SERVER}" >&3
RUNNER_MARKER="HCCL_DYNAMIC_ROOT_RUNNER_TERMINAL_${META}_${RANK_COUNT}_${ALT_ROOT}"
printf 'echo %s\n' "${RUNNER_MARKER}" >&3
hvm_session_wait_for_runner "${LOG_FILE}" "${RUNNER_MARKER}"
printf 'hccl-vm plugin run @checker\n' >&3
CHECKER_MARKER="HCCL_CHECKER_TERMINAL_DYNAMIC_ROOT_${META}_${RANK_COUNT}_${ALT_ROOT}"
printf 'echo %s\n' "${CHECKER_MARKER}" >&3
hvm_session_wait_for_checker_drain "${LOG_FILE}" "${CHECKER_MARKER}"
hvm_session_stop "${LOG_FILE}"
hvm_session_validate_log "${LOG_FILE}"
if ! grep -q '| success' "${LOG_FILE}"; then
    echo "dynamic-root runner did not report success: ${LOG_FILE}" >&2
    exit 1
fi

echo "PASS dynamic roots 0 -> ${ALT_ROOT} -> 0, ranks=${RANK_COUNT}"
