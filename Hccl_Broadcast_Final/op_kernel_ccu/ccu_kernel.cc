/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <cstdint>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace {

constexpr char SEED_READY_DIE0[] = "broadcast_owner_seed_ready_die0";
constexpr char SEED_READY_DIE1[] = "broadcast_owner_seed_ready_die1";

constexpr uint64_t SetBits(uint16_t bitCount)
{
    return (uint64_t(1) << bitCount) - 1;
}

constexpr uint64_t PackParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & SetBits(7)) << 55) |
        ((repeatLoopIndex & SetBits(7)) << 48) |
        ((totalLoopNum & SetBits(7)) << 41);
}

constexpr uint64_t PackOffsetParam(uint64_t gsaOffset, uint64_t eventOffset)
{
    return ((gsaOffset & SetBits(32)) << 21) | (eventOffset & SetBits(10));
}

ccu::LocalAddr LocalAt(BroadcastContext &ctx, const ccu::Variable &relativeOffset)
{
    ccu::LocalAddr local;
    local.addr = ctx.buffer[ctx.arg->rankId];
    local.addr += ctx.ownerOffset;
    local.addr += relativeOffset;
    local.token = ctx.token[ctx.arg->rankId];
    return local;
}

ccu::RemoteAddr RemoteAt(BroadcastContext &ctx, uint32_t peerRank, const ccu::Variable &relativeOffset)
{
    ccu::RemoteAddr remote;
    remote.addr = ctx.buffer[peerRank];
    remote.addr += ctx.ownerOffset;
    remote.addr += relativeOffset;
    remote.token = ctx.token[peerRank];
    return remote;
}

CcuResult NotifyPhase(ChannelHandle channel, uint32_t mask, bool wait)
{
    if (wait) {
        CCU_CHK_RET(ccu::NotifyWait(channel, CKE_PHASE, mask));
    } else {
        CCU_CHK_RET(ccu::NotifyRecord(channel, CKE_PHASE, mask));
    }
    return CCU_SUCCESS;
}

CcuResult NotifyRoot(BroadcastContext &ctx, uint32_t mask, bool wait)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
            CCU_CHK_RET(NotifyPhase(ctx.arg->channels[i], mask, wait));
        }
    }
    return CCU_SUCCESS;
}

void RecordSeedReady(const BroadcastContext &ctx)
{
    if ((ctx.arg->activeDieMask & 1U) != 0) {
        (void)ccu::EventRecord(SEED_READY_DIE0, 1U);
    }
    if ((ctx.arg->activeDieMask & 2U) != 0) {
        (void)ccu::EventRecord(SEED_READY_DIE1, 1U);
    }
}

void WaitOneSeedReady(const BroadcastContext &ctx)
{
    (void)ccu::EventWait(ctx.arg->dieId == 0 ? SEED_READY_DIE0 : SEED_READY_DIE1, 1U);
}

void WaitSeedReadyCount(BroadcastContext &ctx, ccu::Variable &readyTiles)
{
    CCU_IF(readyTiles == 1) {
        WaitOneSeedReady(ctx);
    }
    CCU_IF(readyTiles == 2) {
        WaitOneSeedReady(ctx);
        WaitOneSeedReady(ctx);
    }
    CCU_IF(readyTiles == 3) {
        WaitOneSeedReady(ctx);
        WaitOneSeedReady(ctx);
        WaitOneSeedReady(ctx);
    }
    CCU_IF(readyTiles == 4) {
        WaitOneSeedReady(ctx);
        WaitOneSeedReady(ctx);
        WaitOneSeedReady(ctx);
        WaitOneSeedReady(ctx);
    }
}

CcuResult ExecuteLoop(ccu::Variable &loopParam, ccu::Func &body)
{
    ccu::Loop loop(loopParam, body);
    ccu::Variable parallelConfig;
    parallelConfig = PackParallelParam(0, 0, 1);
    ccu::Variable offsetConfig;
    offsetConfig = PackOffsetParam(0, 0);
    std::vector<ccu::Loop> loops{loop};
    ccu::LoopGroup group(parallelConfig, offsetConfig, 1, loops);
    (void)group;
    return CCU_SUCCESS;
}

void SubmitOwnerWritesNoWait(BroadcastContext &ctx, ccu::LocalAddr &source,
    std::vector<ccu::RemoteAddr> &destinations, ccu::Variable &bytes, ccu::Event &event)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->remoteRanks[i];
        const uint16_t peerMask = static_cast<uint16_t>(1U << peerRank);
        CCU_IF(ctx.root == ctx.arg->rankId) {
            (void)ccu::Write(ctx.arg->channels[i], destinations[i], source, bytes, event, peerMask);
        }
        CCU_IF(ctx.root != ctx.arg->rankId) {
            CCU_IF(ctx.root != peerRank) {
                (void)ccu::Write(ctx.arg->channels[i], destinations[i], source, bytes, event, peerMask);
            }
        }
    }
}

void WaitOwnerWriteBatch(BroadcastContext &ctx, ccu::Event &event)
{
    // Wait only peers that actually received a Write. In particular, a
    // non-root owner neither writes nor waits on its root Channel.
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->remoteRanks[i];
        const uint16_t peerMask = static_cast<uint16_t>(1U << peerRank);
        CCU_IF(ctx.root == ctx.arg->rankId) {
            (void)ccu::EventWait(event, peerMask);
        }
        CCU_IF(ctx.root != ctx.arg->rankId) {
            CCU_IF(ctx.root != peerRank) {
                (void)ccu::EventWait(event, peerMask);
            }
        }
    }
}

void SubmitAndDrainWindow(BroadcastContext &ctx, ccu::Variable &baseOffset,
    ccu::Variable &batchBytes, ccu::Variable &readyTiles, uint32_t slotCount)
{
    std::vector<ccu::Variable> offsets(slotCount);
    std::vector<ccu::LocalAddr> sources(slotCount);
    std::vector<std::vector<ccu::RemoteAddr>> destinations(slotCount);
    offsets[0] = baseOffset;
    for (uint32_t slot = 1; slot < slotCount; ++slot) {
        offsets[slot] = offsets[slot - 1] + batchBytes;
    }

    // Fill the Window first. Every Slot owns a distinct Event, so no Batch
    // relies on completion-bit accumulation in another Slot.
    for (uint32_t slot = 0; slot < slotCount; ++slot) {
        WaitSeedReadyCount(ctx, readyTiles);
        sources[slot] = LocalAt(ctx, offsets[slot]);
        destinations[slot].resize(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            destinations[slot][i] = RemoteAt(ctx, ctx.arg->remoteRanks[i], offsets[slot]);
        }
        SubmitOwnerWritesNoWait(
            ctx, sources[slot], destinations[slot], batchBytes, ctx.pushEvents[slot]);
    }

    // Reclaim every valid Slot before the next Loop iteration can reuse it.
    for (uint32_t slot = 0; slot < slotCount; ++slot) {
        WaitOwnerWriteBatch(ctx, ctx.pushEvents[slot]);
    }
}

CcuResult RunSeedFullTiles(BroadcastContext &ctx)
{
    ccu::Variable remainingTiles;
    remainingTiles = ctx.seedFullTileCount;
    ccu::Variable offset;
    offset = 0;
    ccu::Variable negativeOne;
    negativeOne = UINT64_MAX;
    CCU_WHILE(remainingTiles != 0) {
        ccu::LocalAddr destination = LocalAt(ctx, offset);
        std::vector<ccu::RemoteAddr> sources(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            sources[i] = RemoteAt(ctx, ctx.arg->remoteRanks[i], offset);
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
                (void)ccu::Read(ctx.arg->channels[i], destination, sources[i],
                    ctx.tileSizeBytes, ctx.pushEvents[0], 1U);
                (void)ccu::EventWait(ctx.pushEvents[0], 1U);
            }
        }
        RecordSeedReady(ctx);
        offset += ctx.tileSizeBytes;
        remainingTiles = remainingTiles + negativeOne;
    }
    return CCU_SUCCESS;
}

CcuResult RunSeedTail(BroadcastContext &ctx)
{
    CCU_IF(ctx.seedTailBytes != 0) {
        ccu::Variable offset;
        offset = ctx.seedFullBytes;
        ccu::LocalAddr destination = LocalAt(ctx, offset);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
                ccu::RemoteAddr source = RemoteAt(ctx, ctx.arg->remoteRanks[i], offset);
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[i], destination, source,
                    ctx.seedTailBytes, ctx.pushEvents[0], 1U));
                CCU_CHK_RET(ccu::EventWait(ctx.pushEvents[0], 1U));
            }
        }
        RecordSeedReady(ctx);
    }
    return CCU_SUCCESS;
}

CcuResult RunPushFirst(BroadcastContext &ctx)
{
    CCU_IF(ctx.pushFirstBytes != 0) {
        ccu::Variable offset;
        offset = 0;
        ccu::Variable readyTiles;
        readyTiles = 1;
        SubmitAndDrainWindow(ctx, offset, ctx.pushFirstBytes, readyTiles, 1);
    }
    return CCU_SUCCESS;
}

CcuResult RunPushLoop(BroadcastContext &ctx)
{
    CCU_IF(ctx.pushLoopBytes != 0) {
        ccu::Variable batchBytes;
        batchBytes = ctx.enablePushBatchMerge;
        CCU_IF(ctx.enablePushBatchMerge == 0) {
            batchBytes = ctx.tileSizeBytes;
        }
        CCU_IF(ctx.enablePushBatchMerge != 0) {
            batchBytes = ctx.maxPushBatchBytes;
        }
        ccu::Variable readyTiles;
        readyTiles = 1;
        CCU_IF(ctx.enablePushBatchMerge != 0) {
            readyTiles = ctx.pushMergeFactor;
        }

        ccu::Func body([&ctx, &batchBytes, &readyTiles]() {
            CCU_IF(ctx.pushWindowDepth == 1) {
                SubmitAndDrainWindow(ctx, ctx.pushLoopOffset, batchBytes, readyTiles, 1);
            }
            CCU_IF(ctx.pushWindowDepth == 2) {
                SubmitAndDrainWindow(ctx, ctx.pushLoopOffset, batchBytes, readyTiles, 2);
            }
            CCU_IF(ctx.pushWindowDepth == 4) {
                SubmitAndDrainWindow(ctx, ctx.pushLoopOffset, batchBytes, readyTiles, 4);
            }
        });
        CCU_CHK_RET(ExecuteLoop(ctx.pushLoopParam, body));
    }

    ccu::Variable remainderOffset;
    remainderOffset = ctx.pushLoopOffset + ctx.pushLoopBytes;
    ccu::Variable remainderBatchBytes;
    remainderBatchBytes = ctx.tileSizeBytes;
    CCU_IF(ctx.enablePushBatchMerge != 0) {
        remainderBatchBytes = ctx.maxPushBatchBytes;
    }
    ccu::Variable remainderReadyTiles;
    remainderReadyTiles = 1;
    CCU_IF(ctx.enablePushBatchMerge != 0) {
        remainderReadyTiles = ctx.pushMergeFactor;
    }
    CCU_IF(ctx.pushLoopRemainder == 1) {
        SubmitAndDrainWindow(ctx, remainderOffset, remainderBatchBytes, remainderReadyTiles, 1);
    }
    CCU_IF(ctx.pushLoopRemainder == 2) {
        SubmitAndDrainWindow(ctx, remainderOffset, remainderBatchBytes, remainderReadyTiles, 2);
    }
    CCU_IF(ctx.pushLoopRemainder == 3) {
        SubmitAndDrainWindow(ctx, remainderOffset, remainderBatchBytes, remainderReadyTiles, 3);
    }
    return CCU_SUCCESS;
}

CcuResult RunPushTail(BroadcastContext &ctx)
{
    CCU_IF(ctx.pushTailBytes != 0) {
        SubmitAndDrainWindow(ctx, ctx.pushTailOffset, ctx.pushTailBytes, ctx.pushTailReadyTiles, 1);
    }
    return CCU_SUCCESS;
}

CcuResult ReadFullChunkFromRoot(BroadcastContext &ctx)
{
    ccu::LocalAddr destination;
    destination.addr = ctx.buffer[ctx.arg->rankId];
    destination.addr += ctx.chunkOffset;
    destination.token = ctx.token[ctx.arg->rankId];
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
            ccu::RemoteAddr source;
            source.addr = ctx.buffer[ctx.arg->remoteRanks[i]];
            source.addr += ctx.chunkOffset;
            source.token = ctx.token[ctx.arg->remoteRanks[i]];
            CCU_CHK_RET(ccu::Read(ctx.arg->channels[i], destination, source,
                ctx.chunkBytes, ctx.pushEvents[0], 1U));
            CCU_CHK_RET(ccu::EventWait(ctx.pushEvents[0], 1U));
        }
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult InitBroadcastResource(BroadcastContext &ctx, const BroadcastKernelArg *arg)
{
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->dieId >= BROADCAST_CCU_DIE_NUM ||
        arg->channelCount > arg->rankSize - 1 ||
        (arg->activeDieMask & (1U << arg->dieId)) == 0) {
        return CCU_E_PARA;
    }
    bool seen[MAX_RANK_SIZE]{};
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        const uint32_t peerRank = arg->remoteRanks[i];
        if (peerRank >= arg->rankSize || peerRank == arg->rankId || seen[peerRank]) {
            return CCU_E_PARA;
        }
        seen[peerRank] = true;
        ctx.buffer[peerRank] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], BUFFER_XN_ID);
        ctx.token[peerRank] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID);
    }
    ctx.arg = arg;
    return CCU_SUCCESS;
}

CcuResult LoadSmallBroadcastArgs(BroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.buffer[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.root, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    argId += 3;
    CCU_CHK_RET(ccu::LoadArg(ctx.kernelPhase, argId));
    return CCU_SUCCESS;
}

CcuResult LoadOwnerWriteArgs(BroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.buffer[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.root, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownerOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ownerBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tileSizeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.seedFullTileCount, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.seedFullBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.seedTailBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.enablePushBatchMerge, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.maxPushBatchBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushMergeFactor, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushWindowDepth, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushFirstBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushLoopOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushLoopBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushLoopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushLoopRemainder, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushTailOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushTailBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushTailReadyTiles, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.kernelPhase, argId));
    return CCU_SUCCESS;
}

CcuResult PublishBufferInfo(BroadcastContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.buffer[ctx.arg->rankId],
            BUFFER_XN_ID, CKE_PRESYNC, MASK_BUFFER_READY));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.arg->rankId],
            TOKEN_XN_ID, CKE_PRESYNC, MASK_TOKEN_READY));
    }
    return CCU_SUCCESS;
}

CcuResult WaitBufferInfo(BroadcastContext &ctx)
{
    constexpr uint32_t readyMask = MASK_BUFFER_READY | MASK_TOKEN_READY;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_PRESYNC, readyMask));
    }
    return CCU_SUCCESS;
}

CcuResult PreSyncBufferInfo(BroadcastContext &ctx)
{
    CCU_CHK_RET(PublishBufferInfo(ctx));
    CCU_CHK_RET(WaitBufferInfo(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastSmallReceiverPullKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    BroadcastContext ctx;
    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadSmallBroadcastArgs(ctx));
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(SmallPullPhase::TRANSFER)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(PublishBufferInfo(ctx));
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                CCU_CHK_RET(NotifyPhase(ctx.arg->channels[i], NOTIFY_SMALL_READ_DONE, true));
            }
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            constexpr uint32_t readyMask = MASK_BUFFER_READY | MASK_TOKEN_READY;
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
                    CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_PRESYNC, readyMask));
                }
            }
            CCU_CHK_RET(ReadFullChunkFromRoot(ctx));
            CCU_CHK_RET(NotifyRoot(ctx, NOTIFY_SMALL_READ_DONE, false));
        }
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(SmallPullPhase::GLOBAL_DONE)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                CCU_CHK_RET(NotifyPhase(ctx.arg->channels[i], NOTIFY_SMALL_GLOBAL_DONE, false));
            }
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, NOTIFY_SMALL_GLOBAL_DONE, true));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastOwnerSeedKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    BroadcastContext ctx;
    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadOwnerWriteArgs(ctx));
    CCU_CHK_RET(RunSeedFullTiles(ctx));
    CCU_CHK_RET(RunSeedTail(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastContiguousOwnerWriteKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    BroadcastContext ctx;
    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadOwnerWriteArgs(ctx));
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(OwnerWritePhase::PRESYNC_PUBLISH)) {
        CCU_CHK_RET(PublishBufferInfo(ctx));
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(OwnerWritePhase::PRESYNC_WAIT)) {
        CCU_CHK_RET(WaitBufferInfo(ctx));
    }
    CCU_IF(ctx.kernelPhase == OWNER_WRITE_PHASE_COUNT) {
        CCU_CHK_RET(RunPushFirst(ctx));
        CCU_CHK_RET(RunPushLoop(ctx));
        CCU_CHK_RET(RunPushTail(ctx));
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(OwnerWritePhase::OWNER_DONE)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                CCU_CHK_RET(NotifyPhase(ctx.arg->channels[i], NOTIFY_OWNER_DONE, true));
            }
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, NOTIFY_OWNER_DONE, false));
        }
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(OwnerWritePhase::GLOBAL_DONE)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                CCU_CHK_RET(NotifyPhase(ctx.arg->channels[i], NOTIFY_OWNER_GLOBAL_DONE, false));
            }
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, NOTIFY_OWNER_GLOBAL_DONE, true));
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
