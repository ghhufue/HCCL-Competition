#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/hvm_session_lib.sh"

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 <meta> <rank-count> <root> <bytes> <runner|checker> [iterations]" >&2
    exit 2
fi

META=$1
RANK_COUNT=$2
ROOT=$3
BYTES=$4
MODE=$5
ITERS=${6:-1}

ASCEND_HOME_PATH=/home/workspace/hvm/Ascend/cann-9.1.0
HCCL_VM_HOME=/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install
CUSTOM_SO=/mnt/d/code/HCCL-Competition/Hccl_Broadcast_Final/build/lib64/libhccl.so
TEST_BIN=${ASCEND_HOME_PATH}/tools/hccl_test/bin/broadcast_test
LOG_DIR=/mnt/d/code/HCCL-Competition/artifacts/hvm_matrix
LOG_FILE=${LOG_DIR}/${META}_n${RANK_COUNT}_r${ROOT}_b${BYTES}_${MODE}_i${ITERS}.log
CLUSTER_MODEL=${HCCL_VM_CLUSTER_MODEL:-ascend950_cluster_4_server_competition}

mkdir -p "${LOG_DIR}"
set +u
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${HCCL_VM_HOME}/script/hccl_config.sh"
set -u
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${MODE}" == runner ]]; then
    START_SUFFIX=
    CHECK=1
    INSTALL_RUNNER='hccl-vm plugin install @runner'
elif [[ "${MODE}" == checker ]]; then
    START_SUFFIX='--check-only'
    CHECK=0
    INSTALL_RUNNER=
else
    echo "mode must be runner or checker" >&2
    exit 2
fi

if [[ "${META}" == 141 ]]; then
    NPUS_PER_SERVER=1
else
    NPUS_PER_SERVER=8
fi

HVM_ARGS=("${HCCL_VM_HOME}/bin/hccl-vm" start "${CLUSTER_MODEL}")
if [[ -n "${START_SUFFIX}" ]]; then
    HVM_ARGS+=("${START_SUFFIX}")
fi
hvm_session_start "${LOG_FILE}" "${HVM_ARGS[@]}"
if [[ -n "${INSTALL_RUNNER}" ]]; then
    printf '%s\n' "${INSTALL_RUNNER}" >&3
fi
printf 'hccl-vm mock-comm %s\n' "${META}" >&3
if [[ "${CLUSTER_MODEL}" == ascend950_cluster_4_server_competition && "${RANK_COUNT}" -gt 8 ]]; then
    printf 'python3 /mnt/d/code/HCCL-Competition/artifacts/normalize_hvm_ranktable.py\n' >&3
fi
printf 'env ASCEND_GLOBAL_LOG_LEVEL=3 LD_PRELOAD=${LD_PRELOAD:+${LD_PRELOAD}:}%s mpirun --allow-run-as-root --oversubscribe -np %s %s -b %s -e %s -d fp32 -r %s -w 0 -n %s -c %s -p %s </dev/null\n' \
    "${CUSTOM_SO}" "${RANK_COUNT}" "${TEST_BIN}" "${BYTES}" "${BYTES}" "${ROOT}" "${ITERS}" "${CHECK}" "${NPUS_PER_SERVER}" >&3
RUNNER_MARKER="HCCL_RUNNER_TERMINAL_${META}_${RANK_COUNT}_${ROOT}_${BYTES}_${MODE}_${ITERS}"
printf 'echo %s\n' "${RUNNER_MARKER}" >&3
hvm_session_wait_for_runner "${LOG_FILE}" "${RUNNER_MARKER}"
printf 'hccl-vm plugin run @checker\n' >&3
CHECKER_MARKER="HCCL_CHECKER_TERMINAL_${META}_${RANK_COUNT}_${ROOT}_${BYTES}_${MODE}_${ITERS}"
printf 'echo %s\n' "${CHECKER_MARKER}" >&3
hvm_session_wait_for_checker_drain "${LOG_FILE}" "${CHECKER_MARKER}"
hvm_session_stop "${LOG_FILE}"
hvm_session_validate_log "${LOG_FILE}"

if [[ "${MODE}" == runner ]]; then
    if ! grep -q '| success' "${LOG_FILE}"; then
        echo "data check did not report success: ${LOG_FILE}" >&2
        exit 1
    fi
fi

echo "PASS meta=${META} ranks=${RANK_COUNT} root=${ROOT} bytes=${BYTES} mode=${MODE} iters=${ITERS}"
