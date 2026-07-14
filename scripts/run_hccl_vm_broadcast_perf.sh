#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${PROJECT_DIR}/hccl_broadcast_problem_template"
WORK_DIR="${WORK_DIR:-/tmp/hccl_broadcast_problem_template_verify}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/workspace/hvm/Ascend/cann-9.1.0}"
HCCL_VM_HOME="${HCCL_VM_HOME:-/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install}"
HCCL_TEST_BIN="${HCCL_TEST_BIN:-${ASCEND_HOME_PATH}/tools/hccl_test/bin/broadcast_test}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-ascend950_cluster_4_server_competition.yaml}"
CLUSTER_CONFIG_NAME="${CLUSTER_CONFIG%.yaml}"
TOPO_META="${TOPO_META:-128}"
NP="${NP:-16}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
ROOTS="${ROOTS:-0 7}"
SIZES="${SIZES:-4 524288 536870912 419430404}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/artifacts/hccl_vm_perf}"

CUSTOM_LIB_DIR="${WORK_DIR}/build/lib64"
CUSTOM_HOST_SO="${CUSTOM_LIB_DIR}/libhccl.so"
CUSTOM_DEVICE_SO="${CUSTOM_LIB_DIR}/libhccl_device.so"
CUSTOM_AICPU_CONFIG="${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config"
CUSTOM_AICPU_KERNEL="${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel"

log()
{
    printf '[hccl-vm-test] %s\n' "$*"
}

require_file()
{
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        log "missing required file: ${file}"
        exit 1
    fi
}

run_as_root()
{
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
        return
    fi
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo "$@"
        return
    fi
    log "need root privileges for CANN custom-op install."
    log "Run this script through Windows as:"
    log "  wsl -u root -- bash /mnt/d/code/HCCL-Competition/scripts/run_hccl_vm_broadcast_perf.sh"
    exit 1
}

source_cann_env()
{
    set +u
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/set_env.sh"
    set -u
}

source_hccl_vm_env()
{
    set +u
    # shellcheck disable=SC1091
    source "${HCCL_VM_HOME}/script/hccl_config.sh"
    set -u
}

ensure_hccl_vm_compat_paths()
{
    local compat_dir="/home/teamserver/workspace/Ascend"
    local compat_path="${compat_dir}/cann"
    if [[ -e "${compat_path}/set_env.sh" ]]; then
        return
    fi
    log "creating HCCL-VM compatibility symlink ${compat_path} -> ${ASCEND_HOME_PATH}"
    run_as_root mkdir -p "${compat_dir}"
    run_as_root ln -sfn "${ASCEND_HOME_PATH}" "${compat_path}"
}

build_custom_op()
{
    log "building custom Broadcast op in ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
    cp -a "${TEMPLATE_DIR}" "${WORK_DIR}"
    rm -rf "${WORK_DIR}/build"
    source_cann_env
    (cd "${WORK_DIR}" && bash build.sh)
    require_file "${CUSTOM_HOST_SO}"
    require_file "${CUSTOM_DEVICE_SO}"
}

install_custom_op()
{
    log "installing custom host/device libraries into CANN custom vendor path"
    run_as_root mkdir -p "${ASCEND_HOME_PATH}/opp/vendors/cust/lib64" "${CUSTOM_AICPU_CONFIG}" "${CUSTOM_AICPU_KERNEL}"
    run_as_root cp -f "${CUSTOM_HOST_SO}" "${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_broadcast_custom.so"
    run_as_root cp -f "${CUSTOM_DEVICE_SO}" "${CUSTOM_AICPU_KERNEL}/libhccl_device.so"
    run_as_root cp -f "${CUSTOM_DEVICE_SO}" "${HCCL_VM_HOME}/lib/aarch64/libhccl_device.so"

    local json_tmp
    json_tmp="$(mktemp)"
    cat >"${json_tmp}" <<'JSON'
{
  "HcclAICPUKernel": {
    "opInfo": {
      "opKernelLib": "AICPUKernel",
      "kernelSo": "libhccl_device.so",
      "functionName": "HcclAICPUKernel"
    }
  }
}
JSON
    run_as_root cp -f "${json_tmp}" "${CUSTOM_AICPU_CONFIG}/aicpu_kernel.json"
    rm -f "${json_tmp}"
}

prepare_hccl_vm()
{
    log "preparing HCCL-VM environment for ${CLUSTER_CONFIG_NAME}"
    ensure_hccl_vm_compat_paths
    source_cann_env
    source_hccl_vm_env
    "${HCCL_VM_HOME}/bin/hccl-vm" reset >/dev/null 2>&1 || true
}

run_one_case()
{
    local root="$1"
    local bytes="$2"
    local log_file="${RESULT_DIR}/broadcast_root${root}_${bytes}.log"
    local cmd

    log "running broadcast root=${root} bytes=${bytes}"
    cmd="env LD_PRELOAD=${CUSTOM_HOST_SO}\${LD_PRELOAD:+:\${LD_PRELOAD}} mpirun --allow-run-as-root --oversubscribe -np ${NP} ${HCCL_TEST_BIN} -b ${bytes} -e ${bytes} -d fp32 -r ${root} -w ${WARMUP} -n ${ITERS} -c 1"
    {
        printf 'hccl-vm mock-comm %s\n' "${TOPO_META}"
        printf '%s\n' "${cmd}"
        printf 'exit\n'
    } | "${HCCL_VM_HOME}/bin/hccl-vm" start "${CLUSTER_CONFIG_NAME}" 2>&1 | tee "${log_file}"

    if grep -E 'check result failed|total err|HcclBroadcast call trace|Failed to execute op|call trace: hcclRet' \
        "${log_file}" >/dev/null; then
        log "case failed, see ${log_file}"
        return 1
    fi
}

main()
{
    require_file "${TEMPLATE_DIR}/build.sh"
    require_file "${ASCEND_HOME_PATH}/set_env.sh"
    require_file "${HCCL_VM_HOME}/script/hccl_config.sh"
    require_file "${HCCL_VM_HOME}/bin/hccl-vm"
    require_file "${HCCL_TEST_BIN}"
    mkdir -p "${RESULT_DIR}"

    build_custom_op
    install_custom_op
    prepare_hccl_vm

    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"
    for root in ${ROOTS}; do
        for bytes in ${SIZES}; do
            run_one_case "${root}" "${bytes}"
        done
    done

    log "logs written to ${RESULT_DIR}"
}

main "$@"
