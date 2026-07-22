/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t RESOURCE_LAYOUT_VERSION = 14;
constexpr uint32_t BROADCAST_CCU_DIE_NUM = 2;
constexpr uint32_t BROADCAST_READY_RING_SLOTS = 8;
constexpr uint32_t BROADCAST_SEGMENT_PIPELINE_DEPTH = 2;
constexpr uint32_t SMALL_PULL_PHASE_COUNT = 2;
constexpr uint32_t OWNER_WRITE_PHASE_COUNT = 4;
constexpr uint64_t OWNER_SEGMENT_PULL_PHASE = OWNER_WRITE_PHASE_COUNT + 1;

namespace ops_hccl {

struct OwnerBlock {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint32_t tileCount = 0;
};

struct PushBatchPlan {
    uint64_t firstBytes = 0;
    uint64_t loopOffset = 0;
    uint64_t loopBytes = 0;
    uint64_t loopBatchBytes = 0;
    uint64_t loopCount = 0;
    uint64_t tailOffset = 0;
    uint64_t tailBytes = 0;
    uint32_t tailReadyTiles = 0;
};

constexpr uint64_t DivideRoundUp(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + (value % divisor != 0 ? 1 : 0);
}

// Broadcast accepts FP32 only. Partition in FP32 elements so boundaries stay
// legal and the first remainder ranks absorb the complete, non-divisible tail.
constexpr OwnerBlock GetOwnerBlock(
    uint64_t totalBytes, uint32_t rankSize, uint32_t ownerRank, uint64_t tileSizeBytes)
{
    constexpr uint64_t alignment = sizeof(float);
    if (rankSize == 0 || ownerRank >= rankSize || tileSizeBytes == 0 || totalBytes % alignment != 0) {
        return {};
    }
    const uint64_t totalElements = totalBytes / alignment;
    const uint64_t baseElements = totalElements / rankSize;
    const uint64_t remainderElements = totalElements % rankSize;
    const uint64_t ownerElements = baseElements + (ownerRank < remainderElements ? 1 : 0);
    const uint64_t precedingExtra = ownerRank < remainderElements ? ownerRank : remainderElements;
    const uint64_t offsetElements = static_cast<uint64_t>(ownerRank) * baseElements + precedingExtra;
    const uint64_t ownerBytes = ownerElements * alignment;
    return {offsetElements * alignment, ownerBytes,
        static_cast<uint32_t>(DivideRoundUp(ownerBytes, tileSizeBytes))};
}

constexpr uint64_t BytesUntilCurrentTileEnd(
    uint64_t submittedBytes, uint64_t ownerBytes, uint64_t tileSizeBytes)
{
    if (submittedBytes >= ownerBytes || tileSizeBytes == 0) {
        return 0;
    }
    const uint64_t tileRemaining = tileSizeBytes - submittedBytes % tileSizeBytes;
    return std::min(tileRemaining, ownerBytes - submittedBytes);
}

constexpr uint64_t SelectPushBatchBytes(uint64_t availableBytes, uint64_t submittedBytes,
    uint64_t ownerBytes, uint64_t tileSizeBytes, bool enablePushBatchMerge, uint64_t maxPushBatchBytes)
{
    if (availableBytes == 0 || submittedBytes >= ownerBytes) {
        return 0;
    }
    if (!enablePushBatchMerge) {
        return std::min(availableBytes,
            BytesUntilCurrentTileEnd(submittedBytes, ownerBytes, tileSizeBytes));
    }
    return std::min(availableBytes, maxPushBatchBytes);
}

constexpr PushBatchPlan BuildPushBatchPlan(uint64_t ownerBytes, uint64_t tileSizeBytes,
    bool enablePushBatchMerge, uint64_t maxPushBatchBytes)
{
    PushBatchPlan plan;
    if (ownerBytes == 0 || tileSizeBytes == 0) {
        return plan;
    }
    plan.firstBytes = enablePushBatchMerge ? std::min(ownerBytes, tileSizeBytes) : 0;
    plan.loopOffset = plan.firstBytes;
    const uint64_t remaining = ownerBytes - plan.firstBytes;
    plan.loopBatchBytes = enablePushBatchMerge ? maxPushBatchBytes : tileSizeBytes;
    if (plan.loopBatchBytes == 0) {
        return {};
    }
    plan.loopCount = remaining / plan.loopBatchBytes;
    plan.loopBytes = plan.loopCount * plan.loopBatchBytes;
    plan.tailOffset = plan.loopOffset + plan.loopBytes;
    plan.tailBytes = remaining - plan.loopBytes;
    plan.tailReadyTiles = static_cast<uint32_t>(DivideRoundUp(plan.tailBytes, tileSizeBytes));
    return plan;
}

} // namespace ops_hccl

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

struct BroadcastKernelArg : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t rootRank;
    uint32_t dieId;
    uint32_t seedDie;
    uint32_t activeDieMask;
    uint32_t pushGroup;
    uint32_t pushWindowDepth;
    uint32_t enablePushBatchMerge;
    uint32_t remoteRanks[MAX_RANK_SIZE];
    uint32_t peerGroups[MAX_RANK_SIZE];
};

enum class KernelKind : uint32_t {
    SMALL_RECEIVER_PULL = 0,
    CONTIGUOUS_OWNER_WRITE = 1,
};

enum class SmallPullPhase : uint64_t {
    TRANSFER = 0,
    GLOBAL_DONE = 1,
};

enum class OwnerWritePhase : uint64_t {
    PRESYNC_PUBLISH = 0,
    PRESYNC_WAIT = 1,
    OWNER_DONE = 2,
    GLOBAL_DONE = 3,
};

enum class OwnerSeedPhase : uint64_t {
    RUN = 0,
    FINAL_CREDIT_DRAIN = 1,
};

struct CcuKernelInfo {
    char kernelFuncName[64];
    void *kernelFunc;
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    uint32_t version = RESOURCE_LAYOUT_VERSION;
    uint32_t rankSize = 0;
    uint32_t rootRank = 0;
    uint32_t activeDieMask = 0;
    uint32_t group1DieMask = 0;
    uint32_t pushWindowDepth = 2;
    uint32_t enablePushBatchMerge = 0;
    uint32_t peerDieByRank[MAX_RANK_SIZE]{};
    CommBuffer localBuffer{nullptr, 0};
    CcuKernelHandle smallPullKernels[BROADCAST_CCU_DIE_NUM]{};
    CcuKernelHandle ownerWriteKernels[BROADCAST_CCU_DIE_NUM]{};
    CcuKernelHandle seedKernels[BROADCAST_CCU_DIE_NUM]{};
    CcuKernelHandle segmentPushKernels[BROADCAST_CCU_DIE_NUM]{};
    ThreadHandle pushThreads[BROADCAST_CCU_DIE_NUM]{};
    ThreadHandle pushGroup1Threads[BROADCAST_CCU_DIE_NUM]{};
    uint32_t pushThreadCount = 0;

    static constexpr uint64_t SerializedSize()
    {
        return sizeof(version) + sizeof(rankSize) + sizeof(rootRank) + sizeof(activeDieMask) +
            sizeof(group1DieMask) +
            sizeof(pushWindowDepth) + sizeof(enablePushBatchMerge) + sizeof(peerDieByRank) +
            sizeof(localBuffer) +
            sizeof(smallPullKernels) + sizeof(ownerWriteKernels) + sizeof(seedKernels) +
            sizeof(segmentPushKernels) +
            sizeof(pushThreads) + sizeof(pushGroup1Threads) + sizeof(pushThreadCount);
    }

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << version;
        binaryStream << rankSize;
        binaryStream << rootRank;
        binaryStream << activeDieMask;
        binaryStream << group1DieMask;
        binaryStream << pushWindowDepth;
        binaryStream << enablePushBatchMerge;
        for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
            binaryStream << peerDieByRank[rank];
        }
        binaryStream << localBuffer;
        for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
            binaryStream << smallPullKernels[dieId];
            binaryStream << ownerWriteKernels[dieId];
            binaryStream << seedKernels[dieId];
            binaryStream << segmentPushKernels[dieId];
            binaryStream << pushThreads[dieId];
            binaryStream << pushGroup1Threads[dieId];
        }
        binaryStream << pushThreadCount;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> version;
        binaryStream >> rankSize;
        binaryStream >> rootRank;
        binaryStream >> activeDieMask;
        binaryStream >> group1DieMask;
        binaryStream >> pushWindowDepth;
        binaryStream >> enablePushBatchMerge;
        for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
            binaryStream >> peerDieByRank[rank];
        }
        binaryStream >> localBuffer;
        for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
            binaryStream >> smallPullKernels[dieId];
            binaryStream >> ownerWriteKernels[dieId];
            binaryStream >> seedKernels[dieId];
            binaryStream >> segmentPushKernels[dieId];
            binaryStream >> pushThreads[dieId];
            binaryStream >> pushGroup1Threads[dieId];
        }
        binaryStream >> pushThreadCount;
    }
};

#endif // OPS_HCCL_CUSTOM_H
