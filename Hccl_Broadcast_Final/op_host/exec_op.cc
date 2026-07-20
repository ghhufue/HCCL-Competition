/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t FP32_ALIGNMENT = sizeof(float);
constexpr uint64_t AUTO_DIRECT_THRESHOLD_BYTES = 512ULL * 1024ULL;
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;

HcclResult LaunchChunk(ThreadHandle thread, const OpParam &param, CcuKernelHandle kernel, uint64_t baseAddr,
    uint64_t token, const ChunkDesc &chunk, const uint64_t *phase = nullptr)
{
    std::vector<uint64_t> taskArgs;
    taskArgs.reserve(phase == nullptr ? 8 : 9);
    taskArgs.push_back(baseAddr);
    taskArgs.push_back(token);
    taskArgs.push_back(param.root);
    taskArgs.push_back(chunk.offset);
    taskArgs.push_back(chunk.bytes);
    taskArgs.push_back(chunk.sliceStride);
    taskArgs.push_back(chunk.activeSlices);
    taskArgs.push_back(chunk.tailBytes);
    if (phase != nullptr) {
        taskArgs.push_back(*phase);
    }

    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchChunk] CCU kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult BuildDieLaunchOrder(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t (&dieOrder)[BROADCAST_CCU_DIE_NUM], uint32_t &dieCount)
{
    dieCount = 0;
    if (param.myRank != param.root) {
        const uint32_t rootDie = resCtx.peerDieByRank[param.root];
        CHK_PRT_RET(rootDie >= BROADCAST_CCU_DIE_NUM ||
                (resCtx.activeDieMask & (1U << rootDie)) == 0,
            HCCL_ERROR("[BuildDieLaunchOrder] invalid root channel die=%u", rootDie), HCCL_E_INTERNAL);
        dieOrder[dieCount++] = rootDie;
    }
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((resCtx.activeDieMask & (1U << dieId)) == 0 ||
            (dieCount != 0 && dieOrder[0] == dieId)) {
            continue;
        }
        dieOrder[dieCount++] = dieId;
    }
    CHK_PRT_RET(dieCount == 0 || dieCount > BROADCAST_CCU_DIE_NUM,
        HCCL_ERROR("[BuildDieLaunchOrder] invalid die count=%u", dieCount), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult LaunchPhaseOnDies(const OpParam &param, const AlgResourceCtx &resCtx,
    const CcuKernelHandle (&kernels)[BROADCAST_CCU_DIE_NUM], uint64_t baseAddr, uint64_t token,
    const ChunkDesc &chunk, uint64_t phase)
{
    uint32_t dieOrder[BROADCAST_CCU_DIE_NUM]{};
    uint32_t dieCount = 0;
    CHK_RET(BuildDieLaunchOrder(param, resCtx, dieOrder, dieCount));
    if (dieCount == 1) {
        return LaunchChunk(param.cpuThread, param, kernels[dieOrder[0]], baseAddr, token, chunk, &phase);
    }

    CHK_PRT_RET(resCtx.slaveThreadCount != 1 || resCtx.slaveThread == 0,
        HCCL_ERROR("[LaunchPhaseOnDies] invalid slave resource, count=%u handle=%llu",
            resCtx.slaveThreadCount, static_cast<unsigned long long>(resCtx.slaveThread)),
        HCCL_E_INTERNAL);

    const uint32_t mainDie = dieOrder[0];
    const uint32_t slaveDie = dieOrder[1];
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, resCtx.slaveThread, THREAD_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.slaveThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET(LaunchChunk(param.cpuThread, param, kernels[mainDie], baseAddr, token, chunk, &phase));
    CHK_RET(LaunchChunk(resCtx.slaveThread, param, kernels[slaveDie], baseAddr, token, chunk, &phase));
    CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.slaveThread, param.cpuThread, THREAD_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    HCCL_DEBUG("[LaunchPhaseOnDies] rank=%u root=%u mainDie=%u slaveDie=%u phase=%llu offset=%llu bytes=%llu",
        param.myRank, param.root, mainDie, slaveDie, static_cast<unsigned long long>(phase),
        static_cast<unsigned long long>(chunk.offset), static_cast<unsigned long long>(chunk.bytes));
    return HCCL_SUCCESS;
}

HcclResult SelectAlgorithm(uint64_t totalBytes, KernelKind &algorithm)
{
    const char *requested = std::getenv("HCCL_BROADCAST_ALGO");
    if (requested == nullptr || requested[0] == '\0' || std::strcmp(requested, "auto") == 0) {
        algorithm = totalBytes <= AUTO_DIRECT_THRESHOLD_BYTES ? KernelKind::DIRECT :
                                                                 KernelKind::PULL_SCATTER_ALLGATHER;
        return HCCL_SUCCESS;
    }
    if (std::strcmp(requested, "direct") == 0) {
        algorithm = KernelKind::DIRECT;
        return HCCL_SUCCESS;
    }
    if (std::strcmp(requested, "pull") == 0) {
        algorithm = KernelKind::PULL_SCATTER_ALLGATHER;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[SelectAlgorithm] unsupported HCCL_BROADCAST_ALGO=%s", requested);
    return HCCL_E_PARA;
}

bool SyncEachChunkForDebug()
{
    const char *enabled = std::getenv("HCCL_BROADCAST_SYNC_EACH_CHUNK");
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

} // namespace

HcclResult BuildExecutionPlan(uint64_t totalBytes, uint32_t rankSize, ExecutionPlan &plan)
{
    CHK_PRT_RET(rankSize == 0 || rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[BuildExecutionPlan] invalid rankSize=%u", rankSize), HCCL_E_PARA);
    CHK_PRT_RET((totalBytes % FP32_ALIGNMENT) != 0,
        HCCL_ERROR("[BuildExecutionPlan] totalBytes=%llu is not FP32 aligned",
            static_cast<unsigned long long>(totalBytes)),
        HCCL_E_PARA);

    CHK_RET(SelectAlgorithm(totalBytes, plan.algorithm));
    plan.chunks.clear();

    uint64_t offset = 0;
    while (offset < totalBytes) {
        ChunkDesc chunk;
        chunk.offset = offset;
        chunk.bytes = std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
        const uint64_t unalignedStride = (chunk.bytes + rankSize - 1) / rankSize;
        chunk.sliceStride = (unalignedStride + FP32_ALIGNMENT - 1) & ~(FP32_ALIGNMENT - 1);

        chunk.activeSlices = (chunk.bytes + chunk.sliceStride - 1) / chunk.sliceStride;
        CHK_PRT_RET(chunk.activeSlices == 0 || chunk.activeSlices > rankSize,
            HCCL_ERROR("[BuildExecutionPlan] invalid active slice count=%llu",
                static_cast<unsigned long long>(chunk.activeSlices)), HCCL_E_INTERNAL);
        chunk.tailBytes = chunk.bytes - (chunk.activeSlices - 1) * chunk.sliceStride;
        plan.chunks.push_back(chunk);
        offset += chunk.bytes;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDirectChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk)
{
    ChunkDesc directChunk = chunk;
    directChunk.sliceStride = chunk.bytes;
    directChunk.activeSlices = 1;
    directChunk.tailBytes = chunk.bytes;
    for (uint32_t phase = 0; phase < DIRECT_PHASE_COUNT; ++phase) {
        const uint64_t directPhase = phase;
        CHK_RET(LaunchPhaseOnDies(
            param, resCtx, resCtx.directKernels, baseAddr, token, directChunk, directPhase));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchPhaseAcrossDies(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr,
    uint64_t token, const ChunkDesc &chunk, PullPhase phase)
{
    return LaunchPhaseOnDies(param, resCtx, resCtx.pullKernels, baseAddr, token, chunk,
        static_cast<uint64_t>(phase));
}

HcclResult LaunchPullScatterAllGatherChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk)
{
    for (uint32_t phase = 0; phase < PULL_PHASE_COUNT; ++phase) {
        const PullPhase pullPhase = static_cast<PullPhase>(phase);
        CHK_RET(LaunchPhaseAcrossDies(param, resCtx, baseAddr, token, chunk, pullPhase));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, aclrtStream stream)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != AlgResourceCtx::SerializedSize(),
        HCCL_ERROR("[ExecOp] invalid engine context size=%llu", static_cast<unsigned long long>(param.ctxSize)),
        HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_PRT_RET(resCtx.version != RESOURCE_LAYOUT_VERSION,
        HCCL_ERROR("[ExecOp] resource layout version mismatch, got=%u expected=%u", resCtx.version,
            RESOURCE_LAYOUT_VERSION),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.rankSize != param.rankSize,
        HCCL_ERROR("[ExecOp] context rankSize=%u does not match call rankSize=%u", resCtx.rankSize, param.rankSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.activeDieMask == 0 ||
            (resCtx.activeDieMask & ~((1U << BROADCAST_CCU_DIE_NUM) - 1U)) != 0,
        HCCL_ERROR("[ExecOp] invalid active die mask=%u", resCtx.activeDieMask), HCCL_E_INTERNAL);
    const bool multiDie = (resCtx.activeDieMask & (resCtx.activeDieMask - 1U)) != 0;
    CHK_PRT_RET((multiDie && (resCtx.slaveThreadCount != 1 || resCtx.slaveThread == 0)) ||
            (!multiDie && resCtx.slaveThreadCount != 0),
        HCCL_ERROR("[ExecOp] inconsistent slave thread resource, activeDieMask=%u count=%u handle=%llu",
            resCtx.activeDieMask, resCtx.slaveThreadCount, static_cast<unsigned long long>(resCtx.slaveThread)),
        HCCL_E_INTERNAL);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("[ExecOp] unsupported data type"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeIt->second,
        HCCL_ERROR("[ExecOp] byte size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * sizeIt->second;
    if (totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ExecutionPlan plan;
    CHK_RET(BuildExecutionPlan(totalBytes, param.rankSize, plan));
    HCCL_DEBUG("[ExecOp] rank=%u root=%u bytes=%llu activeDieMask=%u algorithm=%u chunks=%zu", param.myRank,
        param.root, static_cast<unsigned long long>(totalBytes), resCtx.activeDieMask,
        static_cast<uint32_t>(plan.algorithm), plan.chunks.size());

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, totalBytes, &token));

    for (const auto &chunk : plan.chunks) {
        CHK_PRT_RET(chunk.bytes == 0 || chunk.bytes > MAX_DATA_SIZE || chunk.offset > totalBytes - chunk.bytes,
            HCCL_ERROR("[ExecOp] invalid chunk boundary"), HCCL_E_INTERNAL);
        if (plan.algorithm == KernelKind::DIRECT) {
            CHK_RET(LaunchDirectChunk(param, resCtx, baseAddr, token, chunk));
        } else {
            CHK_RET(LaunchPullScatterAllGatherChunk(param, resCtx, baseAddr, token, chunk));
        }
        if (SyncEachChunkForDebug()) {
            const aclError syncRet = aclrtSynchronizeStream(stream);
            CHK_PRT_RET(syncRet != ACL_SUCCESS,
                HCCL_ERROR("[ExecOp] failed to synchronize chunk, aclRet=%d", syncRet), HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
