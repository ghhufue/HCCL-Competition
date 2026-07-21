/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <cerrno>
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
constexpr uint64_t SMALL_PULL_THRESHOLD_BYTES = 512ULL * 1024ULL;
constexpr uint64_t DEFAULT_TILE_SIZE_BYTES = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t DEFAULT_MAX_PUSH_BATCH_BYTES = 8ULL * 1024ULL * 1024ULL;
constexpr uint32_t DEFAULT_PUSH_WINDOW_DEPTH = 2;
constexpr uint64_t MAX_LOOP_ITERATIONS = 8191;
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;

HcclResult ParseBoolEnv(const char *name, bool defaultValue, bool &value)
{
    const char *text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        value = defaultValue;
        return HCCL_SUCCESS;
    }
    if (std::strcmp(text, "1") == 0 || std::strcmp(text, "true") == 0 || std::strcmp(text, "on") == 0) {
        value = true;
        return HCCL_SUCCESS;
    }
    if (std::strcmp(text, "0") == 0 || std::strcmp(text, "false") == 0 || std::strcmp(text, "off") == 0) {
        value = false;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[ParseBoolEnv] %s must be 0/1, false/true, or off/on; got %s", name, text);
    return HCCL_E_PARA;
}

HcclResult ParseSizeEnv(const char *name, uint64_t defaultValue, uint64_t &value)
{
    const char *text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        value = defaultValue;
        return HCCL_SUCCESS;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        HCCL_ERROR("[ParseSizeEnv] %s must be a positive byte count; got %s", name, text);
        return HCCL_E_PARA;
    }
    value = static_cast<uint64_t>(parsed);
    return HCCL_SUCCESS;
}

HcclResult LoadOwnerWriteConfigImpl(OwnerWriteConfig &config)
{
    CHK_RET(ParseSizeEnv("HCCL_BROADCAST_TILE_SIZE_BYTES", DEFAULT_TILE_SIZE_BYTES, config.tileSizeBytes));
    CHK_RET(ParseBoolEnv(
        "HCCL_BROADCAST_ENABLE_PUSH_BATCH_MERGE", false, config.enablePushBatchMerge));
    CHK_RET(ParseSizeEnv("HCCL_BROADCAST_MAX_PUSH_BATCH_BYTES",
        DEFAULT_MAX_PUSH_BATCH_BYTES, config.maxPushBatchBytes));
    uint64_t pushWindowDepth = DEFAULT_PUSH_WINDOW_DEPTH;
    CHK_RET(ParseSizeEnv("HCCL_BROADCAST_PUSH_WINDOW_DEPTH", DEFAULT_PUSH_WINDOW_DEPTH, pushWindowDepth));
    CHK_PRT_RET(pushWindowDepth != 1 && pushWindowDepth != 2 && pushWindowDepth != 4,
        HCCL_ERROR("[LoadOwnerWriteConfig] HCCL_BROADCAST_PUSH_WINDOW_DEPTH must be 1, 2, or 4; got %llu",
            static_cast<unsigned long long>(pushWindowDepth)), HCCL_E_PARA);
    config.pushWindowDepth = static_cast<uint32_t>(pushWindowDepth);
    CHK_PRT_RET(config.tileSizeBytes % FP32_ALIGNMENT != 0 || config.tileSizeBytes > MAX_DATA_SIZE,
        HCCL_ERROR("[LoadOwnerWriteConfig] invalid tileSizeBytes=%llu",
            static_cast<unsigned long long>(config.tileSizeBytes)), HCCL_E_PARA);
    CHK_PRT_RET(config.maxPushBatchBytes < config.tileSizeBytes ||
            config.maxPushBatchBytes > MAX_DATA_SIZE ||
            config.maxPushBatchBytes % config.tileSizeBytes != 0,
        HCCL_ERROR("[LoadOwnerWriteConfig] maxPushBatchBytes=%llu must be a multiple of tileSizeBytes=%llu "
                   "and no larger than %u",
            static_cast<unsigned long long>(config.maxPushBatchBytes),
            static_cast<unsigned long long>(config.tileSizeBytes), MAX_DATA_SIZE), HCCL_E_PARA);
    CHK_PRT_RET(config.maxPushBatchBytes / config.tileSizeBytes > 4,
        HCCL_ERROR("[LoadOwnerWriteConfig] merge factor greater than four is not supported"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult SelectAlgorithm(uint64_t totalBytes, KernelKind &algorithm)
{
    const char *requested = std::getenv("HCCL_BROADCAST_ALGO");
    if (requested == nullptr || requested[0] == '\0' || std::strcmp(requested, "auto") == 0) {
        algorithm = totalBytes <= SMALL_PULL_THRESHOLD_BYTES ? KernelKind::SMALL_RECEIVER_PULL :
                                                                KernelKind::CONTIGUOUS_OWNER_WRITE;
        return HCCL_SUCCESS;
    }
    if (std::strcmp(requested, "small_pull") == 0) {
        algorithm = KernelKind::SMALL_RECEIVER_PULL;
        return HCCL_SUCCESS;
    }
    if (std::strcmp(requested, "owner_write") == 0 ||
        std::strcmp(requested, "contiguous_owner_write") == 0 ||
        std::strcmp(requested, "pull") == 0) {
        algorithm = KernelKind::CONTIGUOUS_OWNER_WRITE;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[SelectAlgorithm] unsupported HCCL_BROADCAST_ALGO=%s", requested);
    return HCCL_E_PARA;
}

const char *KernelKindName(KernelKind algorithm)
{
    return algorithm == KernelKind::SMALL_RECEIVER_PULL ? "SMALL_RECEIVER_PULL" :
                                                          "CONTIGUOUS_OWNER_WRITE";
}

uint32_t ActiveDieCount(uint32_t dieMask)
{
    uint32_t count = 0;
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        count += (dieMask & (1U << dieId)) != 0 ? 1U : 0U;
    }
    return count;
}

HcclResult StartPushThreads(
    const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dieMask)
{
    const uint32_t expected = ActiveDieCount(dieMask);
    CHK_PRT_RET(expected == 0 || expected > resCtx.pushThreadCount ||
            (dieMask & ~resCtx.activeDieMask) != 0,
        HCCL_ERROR("[StartPushThreads] invalid dieMask=%u activeDieMask=%u pushThreadCount=%u",
            dieMask, resCtx.activeDieMask, resCtx.pushThreadCount), HCCL_E_INTERNAL);
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((dieMask & (1U << dieId)) == 0) {
            continue;
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.pushThreads[dieId], THREAD_NOTIFY_INDEX));
    }
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((dieMask & (1U << dieId)) == 0) {
            continue;
        }
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.pushThreads[dieId], THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(ThreadHandle thread, const OpParam &param, CcuKernelHandle kernel,
    uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk, uint64_t phase, bool ownerWriteArgs)
{
    std::vector<uint64_t> taskArgs;
    if (!ownerWriteArgs) {
        taskArgs = {baseAddr, token, param.root, chunk.offset, chunk.bytes, phase};
    } else {
        taskArgs = {baseAddr, token, param.root,
            chunk.owner.offset, chunk.owner.bytes, chunk.tileSizeBytes,
            chunk.seedFullTileCount / BROADCAST_READY_RING_SLOTS,
            chunk.seedFullTileCount % BROADCAST_READY_RING_SLOTS,
            chunk.seedTailBytes,
            chunk.enablePushBatchMerge ? 1ULL : 0ULL, chunk.maxPushBatchBytes,
            chunk.maxPushBatchBytes / chunk.tileSizeBytes,
            chunk.pushWindowDepth, chunk.push.firstBytes, chunk.push.loopCount,
            chunk.push.tailBytes, chunk.push.tailReadyTiles, phase};
    }
    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchKernel] CCU kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult JoinPushThreads(
    const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dieMask)
{
    const uint32_t expected = ActiveDieCount(dieMask);
    CHK_PRT_RET(expected == 0 || expected > resCtx.pushThreadCount,
        HCCL_ERROR("[JoinPushThreads] invalid dieMask=%u pushThreadCount=%u", dieMask, resCtx.pushThreadCount),
        HCCL_E_INTERNAL);
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((dieMask & (1U << dieId)) == 0) {
            continue;
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(
            resCtx.pushThreads[dieId], param.cpuThread, THREAD_NOTIFY_INDEX));
    }
    for (uint32_t i = 0; i < expected; ++i) {
        CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchPhaseOnPushThreads(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dieMask,
    const CcuKernelHandle (&kernels)[BROADCAST_CCU_DIE_NUM], uint64_t baseAddr, uint64_t token,
    const ChunkDesc &chunk, uint64_t phase, bool ownerWriteArgs)
{
    CHK_PRT_RET(dieMask == 0 || (dieMask & ~resCtx.activeDieMask) != 0,
        HCCL_ERROR("[LaunchPhaseOnPushThreads] invalid dieMask=%u activeDieMask=%u",
            dieMask, resCtx.activeDieMask), HCCL_E_INTERNAL);
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((dieMask & (1U << dieId)) == 0) {
            continue;
        }
        CHK_PRT_RET(resCtx.pushThreads[dieId] == 0 || kernels[dieId] == 0,
            HCCL_ERROR("[LaunchPhaseOnPushThreads] missing resource for die=%u", dieId), HCCL_E_INTERNAL);
        CHK_RET(LaunchKernel(resCtx.pushThreads[dieId], param, kernels[dieId],
            baseAddr, token, chunk, phase, ownerWriteArgs));
    }
    return JoinPushThreads(param, resCtx, dieMask);
}

HcclResult GetSmallPullDieMask(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t &dieMask)
{
    if (param.myRank == param.root) {
        dieMask = resCtx.activeDieMask;
        return HCCL_SUCCESS;
    }
    const uint32_t rootDie = resCtx.peerDieByRank[param.root];
    CHK_PRT_RET(rootDie >= BROADCAST_CCU_DIE_NUM ||
            (resCtx.activeDieMask & (1U << rootDie)) == 0,
        HCCL_ERROR("[GetSmallPullDieMask] invalid root channel die=%u", rootDie), HCCL_E_INTERNAL);
    dieMask = 1U << rootDie;
    return HCCL_SUCCESS;
}

uint32_t GetSeedDie(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.myRank != param.root) {
        return resCtx.peerDieByRank[param.root];
    }
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((resCtx.activeDieMask & (1U << dieId)) != 0) {
            return dieId;
        }
    }
    return BROADCAST_CCU_DIE_NUM;
}

} // namespace

HcclResult LoadOwnerWriteConfig(OwnerWriteConfig &config)
{
    return LoadOwnerWriteConfigImpl(config);
}

HcclResult BuildExecutionPlan(
    uint64_t totalBytes, uint32_t rankSize, uint32_t rankId, ExecutionPlan &plan)
{
    CHK_PRT_RET(rankSize == 0 || rankSize > MAX_RANK_SIZE || rankId >= rankSize,
        HCCL_ERROR("[BuildExecutionPlan] invalid rank=%u rankSize=%u", rankId, rankSize), HCCL_E_PARA);
    CHK_PRT_RET(totalBytes % FP32_ALIGNMENT != 0,
        HCCL_ERROR("[BuildExecutionPlan] totalBytes=%llu is not FP32 aligned",
            static_cast<unsigned long long>(totalBytes)), HCCL_E_PARA);

    CHK_RET(SelectAlgorithm(totalBytes, plan.algorithm));
    plan.chunks.clear();
    ChunkDesc chunk;
    chunk.bytes = totalBytes;
    if (plan.algorithm == KernelKind::CONTIGUOUS_OWNER_WRITE) {
        OwnerWriteConfig config;
        CHK_RET(LoadOwnerWriteConfig(config));
        chunk.tileSizeBytes = config.tileSizeBytes;
        chunk.enablePushBatchMerge = config.enablePushBatchMerge;
        chunk.maxPushBatchBytes = config.maxPushBatchBytes;
        chunk.pushWindowDepth = config.pushWindowDepth;
        chunk.owner = GetOwnerBlock(totalBytes, rankSize, rankId, config.tileSizeBytes);
        const uint64_t fullTileCount = chunk.owner.bytes / config.tileSizeBytes;
        CHK_PRT_RET(fullTileCount > MAX_LOOP_ITERATIONS,
            HCCL_ERROR("[BuildExecutionPlan] owner block needs too many Tile iterations=%llu",
                static_cast<unsigned long long>(fullTileCount)), HCCL_E_PARA);
        chunk.seedTailBytes = chunk.owner.bytes - fullTileCount * config.tileSizeBytes;
        chunk.seedFullTileCount = fullTileCount;
        chunk.push = BuildPushBatchPlan(chunk.owner.bytes, config.tileSizeBytes,
            config.enablePushBatchMerge, config.maxPushBatchBytes);
        CHK_PRT_RET(chunk.push.loopCount > MAX_LOOP_ITERATIONS,
            HCCL_ERROR("[BuildExecutionPlan] owner push needs too many loop iterations=%llu",
                static_cast<unsigned long long>(chunk.push.loopCount)), HCCL_E_PARA);
    }
    plan.chunks.push_back(chunk);
    return HCCL_SUCCESS;
}

HcclResult LaunchSmallReceiverPullChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk)
{
    uint32_t dieMask = 0;
    CHK_RET(GetSmallPullDieMask(param, resCtx, dieMask));
    CHK_RET(StartPushThreads(param, resCtx, dieMask));
    for (uint32_t phase = 0; phase < SMALL_PULL_PHASE_COUNT; ++phase) {
        CHK_RET(LaunchPhaseOnPushThreads(param, resCtx, dieMask, resCtx.smallPullKernels,
            baseAddr, token, chunk, phase, false));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchContiguousOwnerWriteChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk)
{
    CHK_RET(StartPushThreads(param, resCtx, resCtx.activeDieMask));
    CHK_RET(LaunchPhaseOnPushThreads(param, resCtx, resCtx.activeDieMask, resCtx.ownerWriteKernels,
        baseAddr, token, chunk, static_cast<uint64_t>(OwnerWritePhase::PRESYNC_PUBLISH), true));
    CHK_RET(LaunchPhaseOnPushThreads(param, resCtx, resCtx.activeDieMask, resCtx.ownerWriteKernels,
        baseAddr, token, chunk, static_cast<uint64_t>(OwnerWritePhase::PRESYNC_WAIT), true));

    const uint32_t seedDie = GetSeedDie(param, resCtx);
    CHK_PRT_RET(seedDie >= BROADCAST_CCU_DIE_NUM || resCtx.seedKernels[seedDie] == 0,
        HCCL_ERROR("[LaunchContiguousOwnerWriteChunk] invalid seed die=%u", seedDie), HCCL_E_INTERNAL);
    CHK_RET(LaunchKernel(param.cpuThread, param, resCtx.seedKernels[seedDie], baseAddr, token,
        chunk, static_cast<uint64_t>(OwnerSeedPhase::RUN), true));
    CHK_RET(LaunchKernel(resCtx.pushThreads[seedDie], param, resCtx.ownerWriteKernels[seedDie],
        baseAddr, token, chunk, OWNER_WRITE_PHASE_COUNT, true));
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((resCtx.activeDieMask & (1U << dieId)) == 0 || dieId == seedDie) {
            continue;
        }
        // CcuLocalNotify is Die-local on the checker/runtime. Keep the Seed
        // Die on the Tile-ready pipeline and release the other Die once the
        // complete owner block has been pulled into local memory.
        CHK_RET(HcommThreadNotifyRecordOnThread(
            param.cpuThread, resCtx.pushThreads[dieId], THREAD_NOTIFY_INDEX));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            resCtx.pushThreads[dieId], THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));
        CHK_RET(LaunchKernel(resCtx.pushThreads[dieId], param, resCtx.ownerWriteKernels[dieId],
            baseAddr, token, chunk, OWNER_WRITE_PHASE_COUNT, true));
    }
    CHK_RET(JoinPushThreads(param, resCtx, resCtx.activeDieMask));
    CHK_RET(LaunchKernel(param.cpuThread, param, resCtx.seedKernels[seedDie], baseAddr, token,
        chunk, static_cast<uint64_t>(OwnerSeedPhase::FINAL_CREDIT_DRAIN), true));

    CHK_RET(LaunchPhaseOnPushThreads(param, resCtx, resCtx.activeDieMask, resCtx.ownerWriteKernels,
        baseAddr, token, chunk, static_cast<uint64_t>(OwnerWritePhase::OWNER_DONE), true));
    CHK_RET(LaunchPhaseOnPushThreads(param, resCtx, resCtx.activeDieMask, resCtx.ownerWriteKernels,
        baseAddr, token, chunk, static_cast<uint64_t>(OwnerWritePhase::GLOBAL_DONE), true));
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, aclrtStream stream)
{
    (void)stream;
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != AlgResourceCtx::SerializedSize(),
        HCCL_ERROR("[ExecOp] invalid engine context size=%llu", static_cast<unsigned long long>(param.ctxSize)),
        HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    CHK_PRT_RET(resCtx.version != RESOURCE_LAYOUT_VERSION,
        HCCL_ERROR("[ExecOp] resource layout version mismatch, got=%u expected=%u",
            resCtx.version, RESOURCE_LAYOUT_VERSION), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.rankSize != param.rankSize,
        HCCL_ERROR("[ExecOp] context rankSize=%u does not match call rankSize=%u",
            resCtx.rankSize, param.rankSize), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.rootRank != param.root,
        HCCL_ERROR("[ExecOp] context root=%u does not match call root=%u",
            resCtx.rootRank, param.root), HCCL_E_INTERNAL);
    const uint32_t activeCount = ActiveDieCount(resCtx.activeDieMask);
    CHK_PRT_RET(activeCount == 0 || activeCount != resCtx.pushThreadCount,
        HCCL_ERROR("[ExecOp] activeDieMask=%u does not match pushThreadCount=%u",
            resCtx.activeDieMask, resCtx.pushThreadCount), HCCL_E_INTERNAL);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("[ExecOp] unsupported data type"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeIt->second,
        HCCL_ERROR("[ExecOp] byte size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * sizeIt->second;
    if (totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ExecutionPlan plan;
    CHK_RET(BuildExecutionPlan(totalBytes, param.rankSize, param.myRank, plan));
    const ChunkDesc &chunk = plan.chunks.front();
    if (plan.algorithm == KernelKind::CONTIGUOUS_OWNER_WRITE) {
        CHK_PRT_RET(chunk.pushWindowDepth != resCtx.pushWindowDepth ||
                static_cast<uint32_t>(chunk.enablePushBatchMerge) != resCtx.enablePushBatchMerge,
            HCCL_ERROR("[ExecOp] owner-write config changed after kernel registration: "
                       "depth=%u/%u merge=%u/%u",
                chunk.pushWindowDepth, resCtx.pushWindowDepth,
                chunk.enablePushBatchMerge ? 1U : 0U, resCtx.enablePushBatchMerge), HCCL_E_PARA);
    }
    const uint64_t pushFullWindows = chunk.pushWindowDepth == 0 ? 0 :
        chunk.push.loopCount / chunk.pushWindowDepth;
    const uint64_t pushWindowRemainder = chunk.pushWindowDepth == 0 ? 0 :
        chunk.push.loopCount % chunk.pushWindowDepth;
    HCCL_DEBUG("[ExecOp] rank=%u root=%u bytes=%llu activeDieMask=%u algorithm=%s "
               "ownerOffset=%llu ownerBytes=%llu tileSize=%llu merge=%u maxBatch=%llu "
               "pushWindowDepth=%u pushLoopBatches=%llu pushFullWindows=%llu pushWindowRemainder=%llu",
        param.myRank, param.root, static_cast<unsigned long long>(totalBytes), resCtx.activeDieMask,
        KernelKindName(plan.algorithm), static_cast<unsigned long long>(chunk.owner.offset),
        static_cast<unsigned long long>(chunk.owner.bytes),
        static_cast<unsigned long long>(chunk.tileSizeBytes), chunk.enablePushBatchMerge ? 1U : 0U,
        static_cast<unsigned long long>(chunk.maxPushBatchBytes), chunk.pushWindowDepth,
        static_cast<unsigned long long>(chunk.push.loopCount),
        static_cast<unsigned long long>(pushFullWindows),
        static_cast<unsigned long long>(pushWindowRemainder));

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, totalBytes, &token));
    if (plan.algorithm == KernelKind::SMALL_RECEIVER_PULL) {
        return LaunchSmallReceiverPullChunk(param, resCtx, baseAddr, token, chunk);
    }
    return LaunchContiguousOwnerWriteChunk(param, resCtx, baseAddr, token, chunk);
}

} // namespace ops_hccl
