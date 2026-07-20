#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/hvm_session_lib.sh"

ASCEND_HOME_PATH=/home/workspace/hvm/Ascend/cann-9.1.0
HCCL_VM_HOME=/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install
CUSTOM_SO=/mnt/d/code/HCCL-Competition/Hccl_Broadcast_Final/build/lib64/libhccl.so
TEST_BIN=${ASCEND_HOME_PATH}/tools/hccl_test/bin/broadcast_test
LOG_FILE=/mnt/d/code/HCCL-Competition/artifacts/final_smoke_4x1.log

set +u
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${HCCL_VM_HOME}/script/hccl_config.sh"
set -u

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
hvm_session_start "${LOG_FILE}" "${HCCL_VM_HOME}/bin/hccl-vm" start \
    ascend950_cluster_4_server_competition
printf 'hccl-vm plugin install @runner\n' >&3
printf 'hccl-vm mock-comm 141\n' >&3
printf 'env LD_PRELOAD=${LD_PRELOAD:+${LD_PRELOAD}:}%s mpirun --allow-run-as-root --oversubscribe -np 4 %s -b 524288 -e 524288 -d fp32 -r 0 -w 0 -n 1 -c 1 </dev/null\n' \
    "${CUSTOM_SO}" "${TEST_BIN}" >&3
RUNNER_MARKER=HCCL_FINAL_SMOKE_RUNNER_TERMINAL
printf 'echo %s\n' "${RUNNER_MARKER}" >&3
hvm_session_wait_for_runner "${LOG_FILE}" "${RUNNER_MARKER}"
printf 'hccl-vm plugin run @checker\n' >&3
CHECKER_MARKER=HCCL_CHECKER_TERMINAL_FINAL_SMOKE
printf 'echo %s\n' "${CHECKER_MARKER}" >&3
hvm_session_wait_for_checker_drain "${LOG_FILE}" "${CHECKER_MARKER}"
hvm_session_stop "${LOG_FILE}"
hvm_session_validate_log "${LOG_FILE}"
grep -q '| success' "${LOG_FILE}"
echo "PASS final smoke"
