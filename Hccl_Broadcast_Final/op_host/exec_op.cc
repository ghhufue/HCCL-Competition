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
constexpr uint64_t SMALL_BROADCAST_BYTES = 512ULL * 1024ULL;
// Keep both kernels registered for A/B measurements. Pull is the correctness/performance baseline.
constexpr bool USE_DIRECT_FOR_512_KIB = false;

HcclResult LaunchChunk(const OpParam &param, CcuKernelHandle kernel, uint64_t baseAddr, uint64_t token,
    const ChunkDesc &chunk, const uint64_t *phase = nullptr)
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
        param.cpuThread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
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

} // namespace

HcclResult BuildExecutionPlan(uint64_t totalBytes, uint32_t rankSize, ExecutionPlan &plan)
{
    CHK_PRT_RET(rankSize == 0 || rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[BuildExecutionPlan] invalid rankSize=%u", rankSize), HCCL_E_PARA);
    CHK_PRT_RET((totalBytes % FP32_ALIGNMENT) != 0,
        HCCL_ERROR("[BuildExecutionPlan] totalBytes=%llu is not FP32 aligned",
            static_cast<unsigned long long>(totalBytes)),
        HCCL_E_PARA);

    plan.algorithm = (USE_DIRECT_FOR_512_KIB && totalBytes == SMALL_BROADCAST_BYTES)
        ? KernelKind::DIRECT
        : KernelKind::PULL_SCATTER_ALLGATHER;
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
    uint32_t dieOrder[BROADCAST_CCU_DIE_NUM]{};
    uint32_t dieCount = 0;
    CHK_RET(BuildDieLaunchOrder(param, resCtx, dieOrder, dieCount));
    for (uint32_t phase = 0; phase < DIRECT_PHASE_COUNT; ++phase) {
        const uint64_t directPhase = phase;
        for (uint32_t dieIdx = 0; dieIdx < dieCount; ++dieIdx) {
            const uint32_t dieId = dieOrder[dieIdx];
            CHK_RET(LaunchChunk(
                param, resCtx.directKernels[dieId], baseAddr, token, directChunk, &directPhase));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchPullScatterAllGatherChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk)
{
    uint32_t dieOrder[BROADCAST_CCU_DIE_NUM]{};
    uint32_t dieCount = 0;
    CHK_RET(BuildDieLaunchOrder(param, resCtx, dieOrder, dieCount));
    for (uint32_t phase = 0; phase < PULL_PHASE_COUNT; ++phase) {
        const PullPhase pullPhase = static_cast<PullPhase>(phase);
        const uint64_t phaseArg = static_cast<uint64_t>(pullPhase);
        for (uint32_t dieIdx = 0; dieIdx < dieCount; ++dieIdx) {
            const uint32_t dieId = dieOrder[dieIdx];
            CHK_RET(LaunchChunk(
                param, resCtx.pullKernels[dieId], baseAddr, token, chunk, &phaseArg));
        }
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
    if ((resCtx.activeDieMask & (resCtx.activeDieMask - 1U)) != 0) {
        // CCU kernels cannot bind channels from different I/O dies. A split Pull
        // kernel cannot implement its global seed/read barriers across those dies,
        // while Direct completes independently on each root-peer channel.
        plan.algorithm = KernelKind::DIRECT;
    }

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
        const aclError syncRet = aclrtSynchronizeStream(stream);
        CHK_PRT_RET(syncRet != ACL_SUCCESS,
            HCCL_ERROR("[ExecOp] failed to synchronize chunk, aclRet=%d", syncRet), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
