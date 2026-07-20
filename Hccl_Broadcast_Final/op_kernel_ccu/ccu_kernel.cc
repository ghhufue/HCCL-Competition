/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

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

void AddSliceOffset(ccu::Address &address, const ccu::Variable &sliceStride, uint32_t ownerRank)
{
    for (uint32_t i = 0; i < ownerRank; ++i) {
        address += sliceStride;
    }
}

ccu::LocalAddr LocalSlice(BroadcastContext &ctx, uint32_t ownerRank)
{
    ccu::LocalAddr local;
    local.addr = ctx.buffer[ctx.arg->rankId];
    local.addr += ctx.chunkOffset;
    AddSliceOffset(local.addr, ctx.sliceStride, ownerRank);
    local.token = ctx.token[ctx.arg->rankId];
    return local;
}

ccu::RemoteAddr RemoteSlice(BroadcastContext &ctx, uint32_t sourceRank, uint32_t ownerRank)
{
    ccu::RemoteAddr remote;
    remote.addr = ctx.buffer[sourceRank];
    remote.addr += ctx.chunkOffset;
    AddSliceOffset(remote.addr, ctx.sliceStride, ownerRank);
    remote.token = ctx.token[sourceRank];
    return remote;
}

CcuResult NotifyPhase(BroadcastContext &ctx, ChannelHandle channel, bool wait)
{
    if (wait) {
        CCU_CHK_RET(ccu::NotifyWait(channel, CKE_PHASE, 1U));
    } else {
        CCU_CHK_RET(ccu::NotifyRecord(channel, CKE_PHASE, 1U));
    }
    return CCU_SUCCESS;
}

CcuResult NotifyRoot(BroadcastContext &ctx, bool wait)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
            CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], wait));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RunDirectRoot(BroadcastContext &ctx)
{
    ccu::LocalAddr source;
    source.addr = ctx.buffer[ctx.arg->rankId];
    source.addr += ctx.chunkOffset;
    source.token = ctx.token[ctx.arg->rankId];

    uint16_t completionMask = 0;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->remoteRanks[i];
        const uint16_t peerMask = static_cast<uint16_t>(1U << peerRank);
        ccu::RemoteAddr destination;
        destination.addr = ctx.buffer[peerRank];
        destination.addr += ctx.chunkOffset;
        destination.token = ctx.token[peerRank];
        CCU_CHK_RET(ccu::Write(
            ctx.arg->channels[i], destination, source, ctx.chunkBytes, ctx.event, peerMask));
        completionMask = static_cast<uint16_t>(completionMask | peerMask);
    }
    if (completionMask != 0) {
        CCU_CHK_RET(ccu::EventWait(ctx.event, completionMask));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], false));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], true));
    }
    return CCU_SUCCESS;
}

CcuResult RunDirectReceiver(BroadcastContext &ctx)
{
    CCU_CHK_RET(NotifyRoot(ctx, true));
    CCU_CHK_RET(NotifyRoot(ctx, false));
    return CCU_SUCCESS;
}

CcuResult RunPullSeedRoot(BroadcastContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], true));
    }
    return CCU_SUCCESS;
}

CcuResult RunPullPhase2StartRoot(BroadcastContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], false));
    }
    return CCU_SUCCESS;
}

CcuResult RunPullReadDoneRoot(BroadcastContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], true));
    }
    return CCU_SUCCESS;
}

CcuResult RunPullGlobalDoneRoot(BroadcastContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], false));
    }
    return CCU_SUCCESS;
}

CcuResult RunPullSeed(BroadcastContext &ctx)
{
    const uint32_t myRank = ctx.arg->rankId;
    const uint16_t myMask = static_cast<uint16_t>(1U << myRank);
    ccu::LocalAddr destination = LocalSlice(ctx, myRank);

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_IF(ctx.root == ctx.arg->remoteRanks[i]) {
            ccu::RemoteAddr source = RemoteSlice(ctx, ctx.arg->remoteRanks[i], myRank);
            CCU_IF(ctx.sliceBytes[myRank] != 0) {
                CCU_CHK_RET(ccu::Read(ctx.arg->channels[i], destination, source,
                    ctx.sliceBytes[myRank], ctx.event, myMask));
            }
            CCU_IF(ctx.sliceBytes[myRank] == 0) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, myMask));
            }
            CCU_CHK_RET(ccu::EventWait(ctx.event, myMask));
            CCU_CHK_RET(NotifyPhase(ctx, ctx.arg->channels[i], false));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RunPullAllGather(BroadcastContext &ctx)
{
    uint16_t completionMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t ownerRank = ctx.arg->remoteRanks[channelIdx];
        const uint16_t ownerMask = static_cast<uint16_t>(1U << ownerRank);
        ccu::LocalAddr destination = LocalSlice(ctx, ownerRank);
        ccu::RemoteAddr source = RemoteSlice(ctx, ownerRank, ownerRank);
        CCU_IF(ctx.sliceBytes[ownerRank] != 0) {
            CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], destination, source,
                ctx.sliceBytes[ownerRank], ctx.event, ownerMask));
        }
        CCU_IF(ctx.sliceBytes[ownerRank] == 0) {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, ownerMask));
        }
        completionMask = static_cast<uint16_t>(completionMask | ownerMask);
    }

    if (completionMask != 0) {
        CCU_CHK_RET(ccu::EventWait(ctx.event, completionMask));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult InitBroadcastResource(BroadcastContext &ctx, const BroadcastKernelArg *arg)
{
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE || arg->rankId >= arg->rankSize ||
        arg->channelCount > arg->rankSize - 1) {
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

CcuResult LoadBroadcastArgs(BroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.buffer[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.root, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.activeSlices, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tailBytes, argId++));

    for (uint32_t ownerRank = 0; ownerRank < ctx.arg->rankSize; ++ownerRank) {
        ctx.sliceBytes[ownerRank] = 0;
    }
    for (uint32_t activeCount = 1; activeCount <= ctx.arg->rankSize; ++activeCount) {
        CCU_IF(ctx.activeSlices == activeCount) {
            for (uint32_t ownerRank = 0; ownerRank + 1 < activeCount; ++ownerRank) {
                ctx.sliceBytes[ownerRank] = ctx.sliceStride;
            }
            ctx.sliceBytes[activeCount - 1] = ctx.tailBytes;
        }
    }
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

CcuResult CcuBroadcastDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    BroadcastContext ctx;
    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadBroadcastArgs(ctx));
    CCU_CHK_RET(ccu::LoadArg(ctx.kernelPhase, 8));

    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(DirectPhase::PRESYNC_PUBLISH)) {
        CCU_CHK_RET(PublishBufferInfo(ctx));
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(DirectPhase::PRESYNC_WAIT)) {
        CCU_CHK_RET(WaitBufferInfo(ctx));
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(DirectPhase::DATA)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(RunDirectRoot(ctx));
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(RunDirectReceiver(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastPullScatterAllGatherKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    BroadcastContext ctx;
    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadBroadcastArgs(ctx));
    CCU_CHK_RET(ccu::LoadArg(ctx.kernelPhase, 8));

    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::PRESYNC_PUBLISH)) {
        CCU_CHK_RET(PublishBufferInfo(ctx));
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::PRESYNC_WAIT)) {
        CCU_CHK_RET(WaitBufferInfo(ctx));
    }

    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::SEED)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(RunPullSeedRoot(ctx));
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(RunPullSeed(ctx));
        }
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::PHASE2_START)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(RunPullPhase2StartRoot(ctx));
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, true));
        }
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::ALLGATHER)) {
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(RunPullAllGather(ctx));
        }
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::READ_DONE)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(RunPullReadDoneRoot(ctx));
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, false));
        }
    }
    CCU_IF(ctx.kernelPhase == static_cast<uint64_t>(PullPhase::GLOBAL_DONE)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(RunPullGlobalDoneRoot(ctx));
        }
        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, true));
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
