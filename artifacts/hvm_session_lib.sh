#!/usr/bin/env bash

# Shared lifecycle helpers for hccl-vm validation scripts. Callers must use
# `set -euo pipefail` and invoke hvm_session_stop after the checker drains.

HVM_SESSION_DIR=
HVM_SESSION_FIFO=
HVM_SESSION_PID=
HVM_SESSION_FD_OPEN=0
HVM_SESSION_STOPPED=0
HVM_CHECKER_DRAIN_LINE=

hvm_session_cleanup() {
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
    if [[ -n "${HVM_SESSION_FIFO}" ]]; then
        rm -f -- "${HVM_SESSION_FIFO}"
    fi
    if [[ -n "${HVM_SESSION_DIR}" ]]; then
        rmdir -- "${HVM_SESSION_DIR}" 2>/dev/null || true
    fi
    exit "${exit_status}"
}

hvm_session_start() {
    local log_file=$1
    shift
    if ! sudo -n -v >/dev/null 2>&1; then
        echo "hccl-vm requires a valid non-interactive sudo credential; run sudo -v first" >&2
        return 1
    fi
    HVM_SESSION_DIR=$(mktemp -d "${TMPDIR:-/tmp}/hccl-broadcast-hvm.XXXXXX")
    HVM_SESSION_FIFO="${HVM_SESSION_DIR}/stdin"
    mkfifo "${HVM_SESSION_FIFO}"
    : >"${log_file}"
    "$@" <"${HVM_SESSION_FIFO}" >"${log_file}" 2>&1 &
    HVM_SESSION_PID=$!
    exec 3>"${HVM_SESSION_FIFO}"
    HVM_SESSION_FD_OPEN=1
    trap hvm_session_cleanup EXIT INT TERM
}

hvm_session_wait_for_pattern() {
    local log_file=$1
    local pattern=$2
    local description=$3
    local timeout_seconds=$4
    local deadline=$((SECONDS + timeout_seconds))
    while ! grep -qE "${pattern}" "${log_file}" 2>/dev/null; do
        if ! kill -0 "${HVM_SESSION_PID}" 2>/dev/null; then
            echo "hccl-vm exited while waiting for ${description}: ${log_file}" >&2
            tail -n 80 "${log_file}" >&2 || true
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "timed out after ${timeout_seconds}s waiting for ${description}: ${log_file}" >&2
            tail -n 80 "${log_file}" >&2 || true
            return 1
        fi
        sleep 1
    done
}

hvm_session_wait_for_runner() {
    local log_file=$1
    local marker=$2
    local timeout_seconds=${HVM_RUNNER_TIMEOUT_SECONDS:-1800}
    hvm_session_wait_for_pattern "${log_file}" "${marker}" "runner terminal marker" "${timeout_seconds}"
}

hvm_session_wait_for_checker_drain() {
    local log_file=$1
    local marker=$2
    local timeout_seconds=${HVM_CHECKER_TIMEOUT_SECONDS:-1800}
    hvm_session_wait_for_pattern "${log_file}" "${marker}" \
        "checker terminal marker" "${timeout_seconds}"

    local submitted completed failed snapshot previous_snapshot=
    local stable_checks=0
    local deadline=$((SECONDS + timeout_seconds))
    while true; do
        submitted=$(grep -c 'Op summary, opIndex=' "${log_file}" 2>/dev/null || true)
        completed=$(grep -c 'Checker Success' "${log_file}" 2>/dev/null || true)
        failed=$(grep -ci 'Checker Failed' "${log_file}" 2>/dev/null || true)
        snapshot="${submitted}:${completed}:${failed}"
        if [[ "${submitted}" -gt 0 && "$((completed + failed))" -eq "${submitted}" ]]; then
            if [[ "${snapshot}" == "${previous_snapshot}" ]]; then
                stable_checks=$((stable_checks + 1))
            else
                stable_checks=0
            fi
            if [[ "${stable_checks}" -ge 3 ]]; then
                break
            fi
        else
            stable_checks=0
        fi
        previous_snapshot=${snapshot}
        if ! kill -0 "${HVM_SESSION_PID}" 2>/dev/null; then
            echo "hccl-vm exited while waiting for checker drain: ${log_file}" >&2
            tail -n 80 "${log_file}" >&2 || true
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "timed out after ${timeout_seconds}s waiting for checker drain: ${log_file}" >&2
            tail -n 80 "${log_file}" >&2 || true
            return 1
        fi
        sleep 1
    done
    HVM_CHECKER_DRAIN_LINE=$(printf 'CHECKER_DRAINED submitted=%s completed=%s failed=%s' \
        "${submitted}" "${completed}" "${failed}")
}

hvm_session_stop() {
    local log_file=$1
    printf '%s\n' exit >&3
    exec 3>&-
    HVM_SESSION_FD_OPEN=0
    if ! wait "${HVM_SESSION_PID}"; then
        echo "hccl-vm exited with failure: ${log_file}" >&2
        return 1
    fi
    if ! grep -q 'Plugin \[checker\] exited successfully' "${log_file}"; then
        echo "checker plugin did not exit successfully: ${log_file}" >&2
        return 1
    fi
    printf '%s\n' "${HVM_CHECKER_DRAIN_LINE}" >>"${log_file}"
    HVM_SESSION_STOPPED=1
    rm -f -- "${HVM_SESSION_FIFO}"
    HVM_SESSION_FIFO=
    rmdir -- "${HVM_SESSION_DIR}"
    HVM_SESSION_DIR=
    trap - EXIT INT TERM
}

hvm_session_validate_log() {
    local log_file=$1
    local drain_line submitted completed failed
    drain_line=$(grep 'CHECKER_DRAINED submitted=' "${log_file}" | tail -n 1 || true)
    if [[ ! "${drain_line}" =~ submitted=([0-9]+)[[:space:]]+completed=([0-9]+)[[:space:]]+failed=([0-9]+) ]]; then
        echo "checker drain marker missing or malformed: ${log_file}" >&2
        return 1
    fi
    submitted=${BASH_REMATCH[1]}
    completed=${BASH_REMATCH[2]}
    failed=${BASH_REMATCH[3]}
    if [[ "${submitted}" -eq 0 || "${submitted}" -ne "${completed}" || "${failed}" -ne 0 ]]; then
        echo "checker did not drain successfully (${drain_line}): ${log_file}" >&2
        return 1
    fi
    if grep -qiE 'Checker Failed|Runner Failed|Runner Error|Result is: Failed|sync timeout|hccl_op_base execute failed' \
        "${log_file}"; then
        echo "validation failure found: ${log_file}" >&2
        return 1
    fi
}
