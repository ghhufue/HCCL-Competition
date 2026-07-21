#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}/Hccl_Broadcast_Final}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/workspace/hvm/Ascend/cann-9.1.0}"
HCCL_VM_HOME="${HCCL_VM_HOME:-/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install}"
HCCL_TEST_BIN="${HCCL_TEST_BIN:-${ASCEND_HOME_PATH}/tools/hccl_test/bin/broadcast_test}"
CLUSTER_MODEL="${CLUSTER_MODEL:-ascend950_cluster_4_server_competition}"
CUSTOM_HOST_SO="${CUSTOM_HOST_SO:-${SOURCE_DIR}/build/lib64/libhccl.so}"
RANKTABLE_DEDUP="${PROJECT_DIR}/scripts/dedup_hccl_vm_ranktable.py"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/artifacts/hccl_vm_final_checker}"

# Finals checker matrix:
#   128      = 2 servers x 8 ranks, roots 0/7
#   141      = 4 servers x 1 rank,  roots 0/1
#   12_8_4   = 8 + 4 ranks,          roots 0/7
TOPOLOGIES="${TOPOLOGIES:-128:16:7,141:4:1,12_8_4:12:7}"
ROOTS="${ROOTS:-}"
SIZES="${SIZES:-524288 536870912 419430404}"
MODE="${MODE:-checker}"
ITERS="${ITERS:-1}"
WARMUP="${WARMUP:-0}"
REBUILD="${REBUILD:-1}"

HCCL_BROADCAST_ALGO="${HCCL_BROADCAST_ALGO:-auto}"
HCCL_BROADCAST_ENABLE_PUSH_BATCH_MERGE="${HCCL_BROADCAST_ENABLE_PUSH_BATCH_MERGE:-0}"
HCCL_BROADCAST_PUSH_WINDOW_DEPTH="${HCCL_BROADCAST_PUSH_WINDOW_DEPTH:-2}"
HCCL_BROADCAST_TILE_SIZE_BYTES="${HCCL_BROADCAST_TILE_SIZE_BYTES:-4194304}"
HCCL_BROADCAST_MAX_PUSH_BATCH_BYTES="${HCCL_BROADCAST_MAX_PUSH_BATCH_BYTES:-8388608}"

HVM_RUNNER_TIMEOUT_SECONDS="${HVM_RUNNER_TIMEOUT_SECONDS:-1800}"
HVM_CHECKER_TIMEOUT_SECONDS="${HVM_CHECKER_TIMEOUT_SECONDS:-1800}"
HVM_STABLE_CHECKS="${HVM_STABLE_CHECKS:-3}"

HVM_SESSION_DIR=
HVM_SESSION_FIFO=
HVM_SESSION_PID=
HVM_SESSION_FD_OPEN=0
HVM_SESSION_STOPPED=0
HVM_CHECKER_DRAIN_LINE=

log()
{
    printf '[hccl-vm-final] %s\n' "$*"
}

die()
{
    log "ERROR: $*" >&2
    exit 1
}

require_file()
{
    [[ -f "$1" ]] || die "missing required file: $1"
}

require_uint()
{
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer; got ${value}"
}

source_environments()
{
    set +u
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/set_env.sh"
    # hccl_config.sh selects CCU_SCHED and configures the HCCL-VM ranktable.
    # shellcheck disable=SC1091
    source "${HCCL_VM_HOME}/script/hccl_config.sh"
    set -u
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
}

build_custom_op()
{
    if [[ "${REBUILD}" == 0 ]]; then
        log "skipping build (REBUILD=0)"
        require_file "${CUSTOM_HOST_SO}"
        return
    fi
    log "building Finals CCU Broadcast from ${SOURCE_DIR}"
    (cd "${SOURCE_DIR}" && ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" bash build.sh)
    require_file "${CUSTOM_HOST_SO}"
}

hvm_session_cleanup()
{
    local exit_status=$?
    trap - EXIT INT TERM
    if [[ "${HVM_SESSION_STOPPED}" -eq 0 && -n "${HVM_SESSION_PID}" ]] &&
        kill -0 "${HVM_SESSION_PID}" 2>/dev/null; then
        if [[ "${HVM_SESSION_FD_OPEN}" -eq 1 ]]; then
            printf '%s\n' exit >&3 2>/dev/null || true
        fi
    fi
    if [[ "${HVM_SESSION_FD_OPEN}" -eq 1 ]]; then
        exec 3>&-
        HVM_SESSION_FD_OPEN=0
    fi
    if [[ -n "${HVM_SESSION_PID}" ]]; then
        wait "${HVM_SESSION_PID}" 2>/dev/null || true
    fi
    [[ -z "${HVM_SESSION_FIFO}" ]] || rm -f -- "${HVM_SESSION_FIFO}"
    [[ -z "${HVM_SESSION_DIR}" ]] || rmdir -- "${HVM_SESSION_DIR}" 2>/dev/null || true
    exit "${exit_status}"
}

hvm_session_start()
{
    local log_file="$1"
    shift
    if [[ "$(id -u)" -ne 0 ]] && ! sudo -n -v >/dev/null 2>&1; then
        die "hccl-vm needs root access; run 'sudo -v' first or launch this script with wsl -u root"
    fi
    HVM_SESSION_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hccl-broadcast-final.XXXXXX")"
    HVM_SESSION_FIFO="${HVM_SESSION_DIR}/stdin"
    mkfifo "${HVM_SESSION_FIFO}"
    : >"${log_file}"
    "$@" <"${HVM_SESSION_FIFO}" >"${log_file}" 2>&1 &
    HVM_SESSION_PID=$!
    exec 3>"${HVM_SESSION_FIFO}"
    HVM_SESSION_FD_OPEN=1
    HVM_SESSION_STOPPED=0
    trap hvm_session_cleanup EXIT INT TERM
}

hvm_session_wait_for_pattern()
{
    local log_file="$1"
    local pattern="$2"
    local description="$3"
    local timeout_seconds="$4"
    local deadline=$((SECONDS + timeout_seconds))
    while ! grep -qE "${pattern}" "${log_file}" 2>/dev/null; do
        if ! kill -0 "${HVM_SESSION_PID}" 2>/dev/null; then
            tail -n 80 "${log_file}" >&2 || true
            die "hccl-vm exited while waiting for ${description}; log=${log_file}"
        fi
        if ((SECONDS >= deadline)); then
            tail -n 80 "${log_file}" >&2 || true
            die "timed out after ${timeout_seconds}s waiting for ${description}; log=${log_file}"
        fi
        sleep 1
    done
}

hvm_session_wait_for_checker_drain()
{
    local log_file="$1"
    local marker="$2"
    hvm_session_wait_for_pattern "${log_file}" "${marker}" \
        "checker terminal marker" "${HVM_CHECKER_TIMEOUT_SECONDS}"

    local submitted completed failed snapshot previous_snapshot=
    local stable_checks=0
    local deadline=$((SECONDS + HVM_CHECKER_TIMEOUT_SECONDS))
    while true; do
        submitted="$(grep -c 'Op summary, opIndex=' "${log_file}" 2>/dev/null || true)"
        completed="$(grep -c 'Checker Success' "${log_file}" 2>/dev/null || true)"
        failed="$(grep -ci 'Checker Failed' "${log_file}" 2>/dev/null || true)"
        snapshot="${submitted}:${completed}:${failed}"
        if [[ "${submitted}" -gt 0 && "$((completed + failed))" -eq "${submitted}" ]]; then
            if [[ "${snapshot}" == "${previous_snapshot}" ]]; then
                stable_checks=$((stable_checks + 1))
            else
                stable_checks=0
            fi
            if [[ "${stable_checks}" -ge "${HVM_STABLE_CHECKS}" ]]; then
                break
            fi
        else
            stable_checks=0
        fi
        previous_snapshot="${snapshot}"
        if ! kill -0 "${HVM_SESSION_PID}" 2>/dev/null; then
            tail -n 80 "${log_file}" >&2 || true
            die "hccl-vm exited while waiting for checker drain; log=${log_file}"
        fi
        if ((SECONDS >= deadline)); then
            tail -n 80 "${log_file}" >&2 || true
            die "timed out waiting for checker drain; submitted=${submitted} completed=${completed} failed=${failed}"
        fi
        sleep 1
    done
    HVM_CHECKER_DRAIN_LINE="CHECKER_DRAINED submitted=${submitted} completed=${completed} failed=${failed}"
}

hvm_session_stop()
{
    local log_file="$1"
    printf '%s\n' exit >&3
    exec 3>&-
    HVM_SESSION_FD_OPEN=0
    if ! wait "${HVM_SESSION_PID}"; then
        die "hccl-vm exited with failure; log=${log_file}"
    fi
    grep -q 'Plugin \[checker\] exited successfully' "${log_file}" ||
        die "checker plugin did not exit successfully; log=${log_file}"
    printf '%s\n' "${HVM_CHECKER_DRAIN_LINE}" >>"${log_file}"
    HVM_SESSION_STOPPED=1
    rm -f -- "${HVM_SESSION_FIFO}"
    HVM_SESSION_FIFO=
    rmdir -- "${HVM_SESSION_DIR}"
    HVM_SESSION_DIR=
    HVM_SESSION_PID=
    trap - EXIT INT TERM
}

validate_log()
{
    local log_file="$1"
    local drain_line submitted completed failed
    drain_line="$(grep 'CHECKER_DRAINED submitted=' "${log_file}" | tail -n 1 || true)"
    if [[ ! "${drain_line}" =~ submitted=([0-9]+)[[:space:]]+completed=([0-9]+)[[:space:]]+failed=([0-9]+) ]]; then
        die "checker drain marker missing or malformed; log=${log_file}"
    fi
    submitted="${BASH_REMATCH[1]}"
    completed="${BASH_REMATCH[2]}"
    failed="${BASH_REMATCH[3]}"
    if [[ "${submitted}" -eq 0 || "${submitted}" -ne "${completed}" || "${failed}" -ne 0 ]]; then
        die "checker did not drain successfully (${drain_line}); log=${log_file}"
    fi
    if log_has_failure "${log_file}"; then
        die "validation failure found; log=${log_file}"
    fi
}

log_has_failure()
{
    grep -qiE 'Checker Failed|Runner Failed|Runner Error|Result is: Failed|sync timeout|ETIMEDOUT|hccl_op_base execute failed|Failed to execute op|check result failed|HcclBroadcast.*call trace|hccl interface return err' \
        "$1"
}

run_one_case()
{
    local meta="$1"
    local rank_count="$2"
    local root="$3"
    local bytes="$4"
    local npus_per_server=8
    local check=0
    local start_suffix=
    local install_runner=
    local log_file="${RESULT_DIR}/${meta}_n${rank_count}_r${root}_b${bytes}_${MODE}_i${ITERS}.log"
    local runner_marker="HCCL_RUNNER_TERMINAL_${meta}_${rank_count}_${root}_${bytes}_${MODE}_${ITERS}"
    local checker_marker="HCCL_CHECKER_TERMINAL_${meta}_${rank_count}_${root}_${bytes}_${MODE}_${ITERS}"
    local hvm_args=("${HCCL_VM_HOME}/bin/hccl-vm" start "${CLUSTER_MODEL}")

    if [[ "${meta}" == 141 ]]; then
        npus_per_server=1
    fi
    if [[ "${MODE}" == checker ]]; then
        start_suffix=--check-only
        hvm_args+=("${start_suffix}")
    else
        check=1
        install_runner=1
    fi

    log "running meta=${meta} ranks=${rank_count} root=${root} bytes=${bytes} mode=${MODE}"
    hvm_session_start "${log_file}" "${hvm_args[@]}"
    if [[ -n "${install_runner}" ]]; then
        printf '%s\n' 'hccl-vm plugin install @runner' >&3
    fi
    printf 'hccl-vm mock-comm %s\n' "${meta}" >&3
    if [[ "${CLUSTER_MODEL}" == ascend950_cluster_4_server_competition && "${rank_count}" -gt 8 ]]; then
        printf 'python3 %q %q --no-backup\n' "${RANKTABLE_DEDUP}" \
            "${HCCL_VM_HOME}/data/ranktable.json" >&3
    fi
    printf 'env HCCL_BROADCAST_ALGO=%q HCCL_BROADCAST_ENABLE_PUSH_BATCH_MERGE=%q HCCL_BROADCAST_PUSH_WINDOW_DEPTH=%q HCCL_BROADCAST_TILE_SIZE_BYTES=%q HCCL_BROADCAST_MAX_PUSH_BATCH_BYTES=%q LD_PRELOAD=${LD_PRELOAD:+${LD_PRELOAD}:}%q mpirun --allow-run-as-root --oversubscribe -np %q %q -b %q -e %q -d fp32 -r %q -w %q -n %q -c %q -p %q </dev/null\n' \
        "${HCCL_BROADCAST_ALGO}" "${HCCL_BROADCAST_ENABLE_PUSH_BATCH_MERGE}" \
        "${HCCL_BROADCAST_PUSH_WINDOW_DEPTH}" "${HCCL_BROADCAST_TILE_SIZE_BYTES}" \
        "${HCCL_BROADCAST_MAX_PUSH_BATCH_BYTES}" "${CUSTOM_HOST_SO}" "${rank_count}" \
        "${HCCL_TEST_BIN}" "${bytes}" "${bytes}" "${root}" "${WARMUP}" "${ITERS}" \
        "${check}" "${npus_per_server}" >&3
    printf 'echo %s\n' "${runner_marker}" >&3
    hvm_session_wait_for_pattern "${log_file}" "${runner_marker}" \
        "runner terminal marker" "${HVM_RUNNER_TIMEOUT_SECONDS}"
    if log_has_failure "${log_file}"; then
        tail -n 120 "${log_file}" >&2 || true
        die "runner failed before checker execution; log=${log_file}"
    fi
    printf '%s\n' 'hccl-vm plugin run @checker' >&3
    printf 'echo %s\n' "${checker_marker}" >&3
    hvm_session_wait_for_checker_drain "${log_file}" "${checker_marker}"
    hvm_session_stop "${log_file}"
    validate_log "${log_file}"
    if [[ "${MODE}" == runner ]]; then
        grep -q '| success' "${log_file}" || die "runner data check did not report success; log=${log_file}"
    fi
    log "PASS meta=${meta} ranks=${rank_count} root=${root} bytes=${bytes} ${HVM_CHECKER_DRAIN_LINE}"
}

validate_configuration()
{
    [[ "${MODE}" == checker || "${MODE}" == runner ]] || die "MODE must be checker or runner"
    [[ "${REBUILD}" == 0 || "${REBUILD}" == 1 ]] || die "REBUILD must be 0 or 1"
    [[ "${HCCL_BROADCAST_ALGO}" =~ ^(auto|small_pull|owner_write|contiguous_owner_write|pull)$ ]] ||
        die "unsupported HCCL_BROADCAST_ALGO=${HCCL_BROADCAST_ALGO}"
    require_uint ITERS "${ITERS}"
    require_uint WARMUP "${WARMUP}"
    require_uint HVM_RUNNER_TIMEOUT_SECONDS "${HVM_RUNNER_TIMEOUT_SECONDS}"
    require_uint HVM_CHECKER_TIMEOUT_SECONDS "${HVM_CHECKER_TIMEOUT_SECONDS}"
    require_uint HVM_STABLE_CHECKS "${HVM_STABLE_CHECKS}"
    require_file "${SOURCE_DIR}/build.sh"
    require_file "${ASCEND_HOME_PATH}/set_env.sh"
    require_file "${HCCL_VM_HOME}/script/hccl_config.sh"
    require_file "${HCCL_VM_HOME}/bin/hccl-vm"
    require_file "${HCCL_TEST_BIN}"
    require_file "${RANKTABLE_DEDUP}"
}

main()
{
    local topology_list spec meta rest rank_count alternate_root roots_for_spec root bytes
    validate_configuration
    mkdir -p "${RESULT_DIR}"
    source_environments
    build_custom_op

    topology_list="${TOPOLOGIES//,/ }"
    for spec in ${topology_list}; do
        meta="${spec%%:*}"
        rest="${spec#*:}"
        rank_count="${rest%%:*}"
        alternate_root="${rest#*:}"
        [[ -n "${meta}" && "${rest}" != "${spec}" && "${alternate_root}" != "${rest}" ]] ||
            die "invalid topology entry '${spec}'; expected meta:rank-count:alternate-root"
        require_uint rank-count "${rank_count}"
        require_uint alternate-root "${alternate_root}"
        ((alternate_root < rank_count)) || die "root ${alternate_root} is outside rank count ${rank_count}"
        roots_for_spec="${ROOTS:-0 ${alternate_root}}"
        for root in ${roots_for_spec}; do
            require_uint root "${root}"
            ((root < rank_count)) || die "root ${root} is outside rank count ${rank_count}"
            for bytes in ${SIZES}; do
                require_uint bytes "${bytes}"
                run_one_case "${meta}" "${rank_count}" "${root}" "${bytes}"
            done
        done
    done
    log "all Finals checker cases passed; logs=${RESULT_DIR}"
}

main "$@"
