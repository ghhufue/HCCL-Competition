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
constexpr char SEED_CREDIT_DIE0[] = "broadcast_owner_seed_credit_die0";
constexpr char SEED_CREDIT_DIE1[] = "broadcast_owner_seed_credit_die1";
constexpr uint16_t READY_RING_MASK = static_cast<uint16_t>((1U << READY_RING_SLOTS) - 1U);

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

const char *ReadyTag(const BroadcastContext &ctx)
{
    return ctx.arg->dieId == 0 ? SEED_READY_DIE0 : SEED_READY_DIE1;
}

const char *CreditTag(const BroadcastContext &ctx)
{
    return ctx.arg->dieId == 0 ? SEED_CREDIT_DIE0 : SEED_CREDIT_DIE1;
}

CcuResult RecordSeedReady(const BroadcastContext &ctx, uint16_t mask)
{
    return ccu::EventRecord(ReadyTag(ctx), mask);
}

CcuResult WaitSeedCredits(const BroadcastContext &ctx, uint16_t mask)
{
    return ccu::EventWait(CreditTag(ctx), mask);
}

CcuResult WaitSeedReady(const BroadcastContext &ctx, uint16_t mask)
{
    return ccu::EventWait(ReadyTag(ctx), mask);
}

CcuResult RecordSeedCredit(const BroadcastContext &ctx, uint16_t mask)
{
    return ccu::EventRecord(CreditTag(ctx), mask);
}

CcuResult RecordSeedReadySlot(const BroadcastContext &ctx, ccu::Variable &slot)
{
    for (uint32_t i = 0; i < READY_RING_SLOTS; ++i) {
        CCU_IF(slot == i) {
            CCU_CHK_RET(RecordSeedReady(ctx, static_cast<uint16_t>(1U << i)));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WaitSeedReadySlot(const BroadcastContext &ctx, ccu::Variable &slot)
{
    for (uint32_t i = 0; i < READY_RING_SLOTS; ++i) {
        CCU_IF(slot == i) {
            CCU_CHK_RET(WaitSeedReady(ctx, static_cast<uint16_t>(1U << i)));
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReadSeedTile(BroadcastContext &ctx, ccu::Variable &offset, ccu::Variable &bytes, ccu::Event &event)
{
    ccu::LocalAddr destination = LocalAt(ctx, offset);
    if (ctx.arg->rankId != ctx.arg->rootRank) {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            if (ctx.arg->remoteRanks[i] == ctx.arg->rootRank) {
                ccu::RemoteAddr source = RemoteAt(ctx, ctx.arg->rootRank, offset);
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[i], destination, source, bytes, event, 1U));
                CCU_CHK_RET(ccu::EventWait(event, 1U));
            }
        }
    }
    offset += bytes;
    return CCU_SUCCESS;
}

CcuResult ProduceFullReadyGroup(BroadcastContext &ctx, ccu::Variable &offset, ccu::Event &event)
{
    // Each Tile owns one bit until every active Push Die consumes it. The
    // group credit below prevents bit reuse from collapsing notifications.
    for (uint32_t slot = 0; slot < READY_RING_SLOTS; ++slot) {
        CCU_CHK_RET(ReadSeedTile(ctx, offset, ctx.tileSizeBytes, event));
        CCU_CHK_RET(RecordSeedReady(ctx, static_cast<uint16_t>(1U << slot)));
    }
    return CCU_SUCCESS;
}

CcuResult RunSeed(BroadcastContext &ctx, ccu::Event &event)
{
    ccu::Variable offset;
    offset = 0;
    ccu::Variable negativeOne;
    negativeOne = UINT64_MAX;
    ccu::Variable one;
    one = 1;
    ccu::Variable groupCount;
    groupCount = ctx.seedFullGroupCount;
    ccu::Variable finalReadyCount;
    finalReadyCount = ctx.seedFullRemainder;
    CCU_IF(ctx.seedTailBytes != 0) {
        finalReadyCount += one;
    }
    CCU_WHILE(groupCount != 0) {
        CCU_CHK_RET(ProduceFullReadyGroup(ctx, offset, event));
        groupCount = groupCount + negativeOne;
        ccu::Variable moreData;
        moreData = groupCount;
        moreData += finalReadyCount;
        CCU_IF(moreData != 0) {
            // Credits mean that both Push Dies consumed this Ready group. Wait
            // only before reusing the ring; the final group is drained after
            // Push completion so the Seed and Push graphs do not form a
            // terminal dependency cycle.
            CCU_CHK_RET(WaitSeedCredits(ctx, READY_RING_MASK));
        }
    }

    ccu::Variable remaining;
    remaining = ctx.seedFullRemainder;
    ccu::Variable slot;
    slot = 0;
    CCU_WHILE(remaining != 0) {
        CCU_CHK_RET(ReadSeedTile(ctx, offset, ctx.tileSizeBytes, event));
        CCU_CHK_RET(RecordSeedReadySlot(ctx, slot));
        slot += one;
        remaining = remaining + negativeOne;
    }
    CCU_IF(ctx.seedTailBytes != 0) {
        CCU_CHK_RET(ReadSeedTile(ctx, offset, ctx.seedTailBytes, event));
        CCU_CHK_RET(RecordSeedReadySlot(ctx, slot));
        slot += one;
    }

    return CCU_SUCCESS;
}

bool IsPushPeer(const BroadcastContext &ctx, uint32_t peerRank)
{
    return ctx.arg->rankId == ctx.arg->rootRank || peerRank != ctx.arg->rootRank;
}

uint16_t PushCompletionMask(const BroadcastContext &ctx)
{
    uint16_t mask = 0;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->remoteRanks[i];
        if (IsPushPeer(ctx, peerRank)) {
            mask = static_cast<uint16_t>(mask | (1U << peerRank));
        }
    }
    return mask;
}

CcuResult SubmitOwnerWritesNoWait(BroadcastContext &ctx, ccu::Variable &offset,
    ccu::Variable &bytes, ccu::Event &event)
{
    ccu::LocalAddr source = LocalAt(ctx, offset);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->remoteRanks[i];
        if (!IsPushPeer(ctx, peerRank)) {
            continue;
        }
        ccu::RemoteAddr destination = RemoteAt(ctx, peerRank, offset);
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[i], destination, source, bytes,
            event, static_cast<uint16_t>(1U << peerRank)));
    }
    return CCU_SUCCESS;
}

CcuResult WaitOwnerWriteBatch(BroadcastContext &ctx, ccu::Event &event)
{
    const uint16_t mask = PushCompletionMask(ctx);
    if (mask != 0) {
        CCU_CHK_RET(ccu::EventWait(event, mask));
    }
    return CCU_SUCCESS;
}

template <uint32_t Depth, bool UseReady>
CcuResult PushFullReadyGroup(BroadcastContext &ctx, ccu::Variable &offset, ccu::Event *events)
{
    for (uint32_t windowStart = 0; windowStart < READY_RING_SLOTS; windowStart += Depth) {
        for (uint32_t eventSlot = 0; eventSlot < Depth; ++eventSlot) {
            const uint32_t readySlot = windowStart + eventSlot;
            if (UseReady) {
                CCU_CHK_RET(WaitSeedReady(ctx, static_cast<uint16_t>(1U << readySlot)));
            }
            CCU_CHK_RET(SubmitOwnerWritesNoWait(ctx, offset, ctx.tileSizeBytes, events[eventSlot]));
            offset += ctx.tileSizeBytes;
        }
        if (UseReady && windowStart + Depth == READY_RING_SLOTS) {
            // Release the Ready ring before the final Write window drains.
            CCU_CHK_RET(RecordSeedCredit(ctx, READY_RING_MASK));
        }
        for (uint32_t eventSlot = 0; eventSlot < Depth; ++eventSlot) {
            CCU_CHK_RET(WaitOwnerWriteBatch(ctx, events[eventSlot]));
        }
    }
    return CCU_SUCCESS;
}

template <uint32_t Depth>
CcuResult SubmitDynamicEventSlot(BroadcastContext &ctx, ccu::Variable &eventSlot,
    ccu::Variable &offset, ccu::Variable &bytes, ccu::Event *events)
{
    for (uint32_t i = 0; i < Depth; ++i) {
        CCU_IF(eventSlot == i) {
            CCU_CHK_RET(SubmitOwnerWritesNoWait(ctx, offset, bytes, events[i]));
        }
    }
    return CCU_SUCCESS;
}

template <uint32_t Depth>
CcuResult DrainFullDynamicWindow(BroadcastContext &ctx, ccu::Variable &eventSlot, ccu::Event *events)
{
    CCU_IF(eventSlot == Depth) {
        for (uint32_t i = 0; i < Depth; ++i) {
            CCU_CHK_RET(WaitOwnerWriteBatch(ctx, events[i]));
        }
        eventSlot = 0;
    }
    return CCU_SUCCESS;
}

template <uint32_t Depth>
CcuResult DrainPartialDynamicWindow(BroadcastContext &ctx, ccu::Variable &eventSlot, ccu::Event *events)
{
    for (uint32_t used = 1; used < Depth; ++used) {
        CCU_IF(eventSlot == used) {
            for (uint32_t i = 0; i < used; ++i) {
                CCU_CHK_RET(WaitOwnerWriteBatch(ctx, events[i]));
            }
        }
    }
    return CCU_SUCCESS;
}

template <uint32_t Depth, bool UseReady>
CcuResult RunUnmergedPush(BroadcastContext &ctx, ccu::Event *events)
{
    ccu::Variable offset;
    offset = 0;
    ccu::Variable negativeOne;
    negativeOne = UINT64_MAX;
    ccu::Variable one;
    one = 1;
    ccu::Variable groupCount;
    groupCount = ctx.seedFullGroupCount;
    CCU_WHILE(groupCount != 0) {
        CCU_CHK_RET((PushFullReadyGroup<Depth, UseReady>(ctx, offset, events)));
        groupCount = groupCount + negativeOne;
    }

    ccu::Variable remaining;
    remaining = ctx.seedFullRemainder;
    ccu::Variable finalReadyCount;
    finalReadyCount = remaining;
    ccu::Variable readySlot;
    readySlot = 0;
    ccu::Variable eventSlot;
    eventSlot = 0;
    CCU_WHILE(remaining != 0) {
        if (UseReady) {
            CCU_CHK_RET(WaitSeedReadySlot(ctx, readySlot));
        }
        CCU_CHK_RET(SubmitDynamicEventSlot<Depth>(ctx, eventSlot, offset, ctx.tileSizeBytes, events));
        offset += ctx.tileSizeBytes;
        readySlot += one;
        eventSlot += one;
        CCU_CHK_RET(DrainFullDynamicWindow<Depth>(ctx, eventSlot, events));
        remaining = remaining + negativeOne;
    }
    CCU_IF(ctx.seedTailBytes != 0) {
        finalReadyCount += one;
        if (UseReady) {
            CCU_CHK_RET(WaitSeedReadySlot(ctx, readySlot));
        }
        CCU_CHK_RET(SubmitDynamicEventSlot<Depth>(ctx, eventSlot, offset, ctx.seedTailBytes, events));
        eventSlot += one;
        CCU_CHK_RET(DrainFullDynamicWindow<Depth>(ctx, eventSlot, events));
    }
    if (UseReady) {
        CCU_IF(finalReadyCount != 0) {
            CCU_CHK_RET(RecordSeedCredit(ctx, READY_RING_MASK));
        }
    }
    CCU_CHK_RET(DrainPartialDynamicWindow<Depth>(ctx, eventSlot, events));
    return CCU_SUCCESS;
}

CcuResult ConsumeReadyTile(BroadcastContext &ctx, ccu::Variable &readySlot, ccu::Variable &one)
{
    CCU_CHK_RET(WaitSeedReadySlot(ctx, readySlot));
    readySlot += one;
    CCU_IF(readySlot == READY_RING_SLOTS) {
        CCU_CHK_RET(RecordSeedCredit(ctx, READY_RING_MASK));
        readySlot = 0;
    }
    return CCU_SUCCESS;
}

CcuResult ConsumeReadyTiles(BroadcastContext &ctx, ccu::Variable &readySlot,
    ccu::Variable &readyTiles, ccu::Variable &one)
{
    for (uint32_t count = 1; count <= 4; ++count) {
        CCU_IF(readyTiles == count) {
            for (uint32_t i = 0; i < count; ++i) {
                CCU_CHK_RET(ConsumeReadyTile(ctx, readySlot, one));
            }
        }
    }
    return CCU_SUCCESS;
}

template <uint32_t Depth, bool UseReady>
CcuResult SubmitMergedBatch(BroadcastContext &ctx, ccu::Variable &readySlot, ccu::Variable &eventSlot,
    ccu::Variable &offset, ccu::Variable &bytes, ccu::Variable &readyTiles,
    ccu::Variable &one, ccu::Event *events)
{
    if (UseReady) {
        CCU_CHK_RET(ConsumeReadyTiles(ctx, readySlot, readyTiles, one));
    }
    CCU_CHK_RET(SubmitDynamicEventSlot<Depth>(ctx, eventSlot, offset, bytes, events));
    offset += bytes;
    eventSlot += one;
    CCU_CHK_RET(DrainFullDynamicWindow<Depth>(ctx, eventSlot, events));
    return CCU_SUCCESS;
}

template <uint32_t Depth, bool UseReady>
CcuResult RunMergedPush(BroadcastContext &ctx, ccu::Event *events)
{
    ccu::Variable offset;
    offset = 0;
    ccu::Variable readySlot;
    readySlot = 0;
    ccu::Variable eventSlot;
    eventSlot = 0;
    ccu::Variable one;
    one = 1;
    CCU_IF(ctx.pushFirstBytes != 0) {
        CCU_CHK_RET((SubmitMergedBatch<Depth, UseReady>(ctx, readySlot, eventSlot, offset,
            ctx.pushFirstBytes, one, one, events)));
    }

    ccu::Variable negativeOne;
    negativeOne = UINT64_MAX;
    ccu::Variable loopCount;
    loopCount = ctx.pushLoopCount;
    CCU_WHILE(loopCount != 0) {
        CCU_CHK_RET((SubmitMergedBatch<Depth, UseReady>(ctx, readySlot, eventSlot, offset,
            ctx.maxPushBatchBytes, ctx.pushMergeFactor, one, events)));
        loopCount = loopCount + negativeOne;
    }
    CCU_IF(ctx.pushTailBytes != 0) {
        CCU_CHK_RET((SubmitMergedBatch<Depth, UseReady>(ctx, readySlot, eventSlot, offset,
            ctx.pushTailBytes, ctx.pushTailReadyTiles, one, events)));
    }
    if (UseReady) {
        CCU_IF(readySlot != 0) {
            CCU_CHK_RET(RecordSeedCredit(ctx, READY_RING_MASK));
        }
    }
    CCU_CHK_RET(DrainPartialDynamicWindow<Depth>(ctx, eventSlot, events));
    return CCU_SUCCESS;
}

template <bool UseReady>
CcuResult RunPush(BroadcastContext &ctx, ccu::Event *events)
{
    if (ctx.arg->enablePushBatchMerge == 0) {
        if (ctx.arg->pushWindowDepth == 1) {
            return RunUnmergedPush<1, UseReady>(ctx, events);
        } else if (ctx.arg->pushWindowDepth == 2) {
            return RunUnmergedPush<2, UseReady>(ctx, events);
        } else {
            return RunUnmergedPush<4, UseReady>(ctx, events);
        }
    } else {
        if (ctx.arg->pushWindowDepth == 1) {
            return RunMergedPush<1, UseReady>(ctx, events);
        } else if (ctx.arg->pushWindowDepth == 2) {
            return RunMergedPush<2, UseReady>(ctx, events);
        } else {
            return RunMergedPush<4, UseReady>(ctx, events);
        }
    }
}

CcuResult ReadFullChunkFromRoot(BroadcastContext &ctx, ccu::Event &event)
{
    ccu::LocalAddr destination;
    destination.addr = ctx.buffer[ctx.arg->rankId];
    destination.addr += ctx.chunkOffset;
    destination.token = ctx.token[ctx.arg->rankId];
    if (ctx.arg->rankId != ctx.arg->rootRank) {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            if (ctx.arg->remoteRanks[i] == ctx.arg->rootRank) {
                ccu::RemoteAddr source;
                source.addr = ctx.buffer[ctx.arg->rootRank];
                source.addr += ctx.chunkOffset;
                source.token = ctx.token[ctx.arg->rootRank];
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[i], destination, source,
                    ctx.chunkBytes, event, 1U));
                CCU_CHK_RET(ccu::EventWait(event, 1U));
            }
        }
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult InitBroadcastResource(BroadcastContext &ctx, const BroadcastKernelArg *arg)
{
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->rootRank >= arg->rankSize ||
        arg->dieId >= BROADCAST_CCU_DIE_NUM ||
        arg->seedDie >= BROADCAST_CCU_DIE_NUM ||
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
    CCU_CHK_RET(ccu::LoadArg(ctx.kernelPhase, argId++));
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
    CCU_CHK_RET(ccu::LoadArg(ctx.seedFullGroupCount, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.seedFullRemainder, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.seedTailBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.enablePushBatchMerge, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.maxPushBatchBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushMergeFactor, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushWindowDepth, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushFirstBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.pushLoopCount, argId++));
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
    ccu::Event readEvent;
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
            CCU_CHK_RET(ReadFullChunkFromRoot(ctx, readEvent));
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
    ccu::Event readEvent;
    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadOwnerWriteArgs(ctx));
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(OwnerSeedPhase::RUN)) {
        CCU_CHK_RET(RunSeed(ctx, readEvent));
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(OwnerSeedPhase::FINAL_CREDIT_DRAIN)) {
        CCU_CHK_RET(WaitSeedCredits(ctx, READY_RING_MASK));
    }
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
    if (kernelArg->pushWindowDepth == 1) {
        ccu::Event pushEvents[1];
        CCU_IF(ctx.kernelPhase == OWNER_WRITE_PHASE_COUNT) {
            if (kernelArg->dieId == kernelArg->seedDie) {
                CCU_CHK_RET(RunPush<true>(ctx, pushEvents));
            } else {
                CCU_CHK_RET(RunPush<false>(ctx, pushEvents));
            }
        }
    } else if (kernelArg->pushWindowDepth == 2) {
        ccu::Event pushEvents[2];
        CCU_IF(ctx.kernelPhase == OWNER_WRITE_PHASE_COUNT) {
            if (kernelArg->dieId == kernelArg->seedDie) {
                CCU_CHK_RET(RunPush<true>(ctx, pushEvents));
            } else {
                CCU_CHK_RET(RunPush<false>(ctx, pushEvents));
            }
        }
    } else {
        ccu::Event pushEvents[4];
        CCU_IF(ctx.kernelPhase == OWNER_WRITE_PHASE_COUNT) {
            if (kernelArg->dieId == kernelArg->seedDie) {
                CCU_CHK_RET(RunPush<true>(ctx, pushEvents));
            } else {
                CCU_CHK_RET(RunPush<false>(ctx, pushEvents));
            }
        }
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
