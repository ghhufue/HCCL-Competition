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
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
// 极小包由 root 串行扇出；中包由 Channel worker 并行扇出；大包先分片
// 到 owner，再由 owner 并行扩散。
enum class AlgorithmKind : uint32_t {
    DIRECT_FANOUT = 0,
    PARALLEL_DIRECT_FANOUT = 1,
    DISTRIBUTED_SCATTER_FANOUT = 2,
};

struct AlgorithmConfig {
    // 中包只做一次本地拷贝，再由 worker 并行写所有 peer。下限避免 4B 等
    // 极小包支付 worker 启动开销；上限同时是 Distributed 的启用阈值。
    static constexpr uint64_t kParallelDirectMinBytes = 64ULL << 10;
    static constexpr uint64_t kParallelDirectMaxBytes = 1ULL << 20;
    // 这是期望值；如果 HCCL Buffer 不足，BuildExecutionPlan 会按槽位容量缩小。
    static constexpr uint64_t kPreferredTileBytes = 4ULL << 20;
    static constexpr uint64_t kMinimumTileBytes = 4ULL << 10;
    static constexpr uint32_t kPreferredPipelineDepth = 2;
};

// 根据本次调用动态生成的执行计划，不会写回 Engine Context。
struct ExecutionPlan {
    AlgorithmKind algorithm = AlgorithmKind::DIRECT_FANOUT;
    uint64_t totalBytes = 0;
    uint64_t tileBytes = 0;
    // 一个 stripe 最多包含 ownerCount 个 Tile，每个 owner 各负责一个。
    uint64_t stripeCount = 0;
    uint32_t ownerCount = 0;
    uint32_t pipelineDepth = 1;
};

const char *AlgorithmName(AlgorithmKind algorithm)
{
    switch (algorithm) {
        case AlgorithmKind::DIRECT_FANOUT:
            return "DIRECT_FANOUT";
        case AlgorithmKind::PARALLEL_DIRECT_FANOUT:
            return "PARALLEL_DIRECT_FANOUT";
        case AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT:
            return "DISTRIBUTED_SCATTER_FANOUT";
        default:
            return "UNKNOWN";
    }
}

// 一个连续 Tile 的纯描述。所有 rank 用同一公式计算，通信双方无需额外交换元数据。
struct TileDesc {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint32_t ownerIndex = 0;
    uint32_t ownerRank = INVALID_VALUE_RANKID;
    // 双缓冲窗口编号，即 stripeIndex % pipelineDepth。
    uint32_t windowIndex = 0;
    bool valid = false;
};

HcclResult CheckHcommRet(int32_t ret, const char *apiName)
{
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("%s failed, ret=%d", apiName, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

uint8_t *OffsetPtr(void *ptr, uint64_t offset)
{
    return static_cast<uint8_t *>(ptr) + offset;
}

uint64_t DivideRoundUp(uint64_t value, uint64_t divisor)
{
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value - value % alignment;
}

uint32_t OwnerIndexToRank(uint32_t ownerIndex, uint32_t root)
{
    // owner 序号跳过 root。例如 root=7 时 owner 0..6 -> rank 0..6，
    // owner 7..14 -> rank 8..15。
    return ownerIndex < root ? ownerIndex : ownerIndex + 1;
}

bool RankToOwnerIndex(uint32_t rank, uint32_t root, uint32_t &ownerIndex)
{
    if (rank == root) {
        return false;
    }
    ownerIndex = rank < root ? rank : rank - 1;
    return true;
}

const ChannelInfo *FindChannelByRemoteRank(const AlgResourceCtx &resource, uint32_t remoteRank)
{
    for (const auto &channel : resource.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

uint64_t GetMinRemoteBufferSize(const AlgResourceCtx &resource)
{
    uint64_t minSize = UINT64_MAX;
    for (const auto &channel : resource.channels) {
        minSize = std::min(minSize, channel.remoteCclMem.size);
    }
    return minSize == UINT64_MAX ? 0 : minSize;
}

uint32_t DataReadyNotifyIndex(uint32_t windowIndex)
{
    // window 0/1 分别使用 Channel Notify 0/1。
    return windowIndex;
}

uint32_t SlotConsumedNotifyIndex(const ExecutionPlan &plan, uint32_t windowIndex)
{
    // DATA_READY 之后的另一半 Notify 用于接收方返回“槽位可安全复用”。
    // pipelineDepth=2 时即 Notify 2/3。
    return plan.pipelineDepth + windowIndex;
}

uint32_t WorkerStartNotifyIndex(uint32_t windowIndex)
{
    // 每个 pipeline window 使用独立的启动 Notify，避免多个 Record 合并成一个信号。
    return windowIndex;
}

uint32_t WorkerDoneNotifyIndex(const ExecutionPlan &plan, uint32_t workerIndex, uint32_t windowIndex)
{
    // 每个 worker 的每个 window 都使用独立的完成 Notify。
    return plan.pipelineDepth + windowIndex * plan.ownerCount + (workerIndex - 1);
}

uint32_t WorkerLaunchNotifyIndex(uint32_t workerCount, uint32_t pipelineDepth)
{
    // 一次性 main -> worker 启动 Notify 位于所有逐 window Notify 之后。
    return pipelineDepth * (workerCount + 1);
}

uint64_t RequiredThreadNotifyCount(uint32_t workerCount, uint32_t pipelineDepth)
{
    return static_cast<uint64_t>(pipelineDepth) * (workerCount + 1) + 1;
}

// 这里同时校验调用参数和反序列化后的静态资源，防止在错误 Context 上做地址计算。
HcclResult ValidateExecutionContext(const OpParam &param, const AlgResourceCtx &resource, uint64_t &totalBytes)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resource.localBuffer.addr);

    if (param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize) {
        HCCL_ERROR("invalid rank info, myRank=%u root=%u rankSize=%u", param.myRank, param.root, param.rankSize);
        return HCCL_E_PARA;
    }
    if (resource.layoutVersion != static_cast<uint32_t>(ResourceLayoutVersion::VERSION_3)) {
        HCCL_ERROR("unsupported resource layout version=%u", resource.layoutVersion);
        return HCCL_E_PARA;
    }
    if (resource.rankSize != param.rankSize) {
        HCCL_ERROR("resource rankSize mismatch, resource=%u param=%u", resource.rankSize, param.rankSize);
        return HCCL_E_PARA;
    }

    auto iter = SIZE_TABLE.find(param.dataType);
    if (iter == SIZE_TABLE.end() || iter->second == 0) {
        HCCL_ERROR("unsupported dataType=%d", static_cast<int32_t>(param.dataType));
        return HCCL_E_PARA;
    }
    if (param.count > UINT64_MAX / iter->second) {
        HCCL_ERROR("count overflow, count=%lu elementSize=%u", param.count, iter->second);
        return HCCL_E_PARA;
    }
    totalBytes = param.count * iter->second;
    if (param.rankSize > 1 && resource.channels.size() != param.rankSize - 1) {
        HCCL_ERROR("channel count mismatch, channels=%lu rankSize=%u", resource.channels.size(), param.rankSize);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

bool HasDistributedResources(const OpParam &param, const AlgResourceCtx &resource, uint32_t pipelineDepth)
{
    // Distributed 要求：每个 peer 有唯一 worker、所有 Channel 按 rank 升序、
    // Thread/Channel Notify 数足够支撑双缓冲。条件不满足时安全回退 Direct。
    uint32_t ownerCount = param.rankSize - 1;
    uint64_t requiredThreadNotify = RequiredThreadNotifyCount(ownerCount, pipelineDepth);
    if (ownerCount == 0 || resource.workerCount != ownerCount || resource.threads.size() != ownerCount + 1 ||
        resource.aicpuThread != resource.threads[0] || requiredThreadNotify > UINT32_MAX ||
        resource.notifyNumPerThread < requiredThreadNotify ||
        resource.channelNotifyNum < pipelineDepth * 2) {
        return false;
    }

    std::vector<bool> workerSeen(ownerCount + 1, false);
    uint32_t previousRank = INVALID_VALUE_RANKID;
    for (const auto &channel : resource.channels) {
        if (channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank ||
            (previousRank != INVALID_VALUE_RANKID && channel.remoteRank <= previousRank) ||
            channel.workerIndex == 0 || channel.workerIndex > ownerCount || workerSeen[channel.workerIndex] ||
            channel.notifyNum < pipelineDepth * 2 || channel.remoteCclMem.addr == nullptr) {
            return false;
        }
        previousRank = channel.remoteRank;
        workerSeen[channel.workerIndex] = true;
    }
    return true;
}

bool HasParallelDirectResources(const OpParam &param, const AlgResourceCtx &resource, uint32_t pipelineDepth)
{
    // Parallel Direct 允许多个 Channel 共享 worker，因此只要求 workerIndex
    // 落在已申请线程范围内。最终每个 worker（包括没有分到 Channel 的 worker）
    // 都会发送一次完成通知，主线程的 Record/Wait 数量始终严格一致。
    uint64_t requiredThreadNotify = RequiredThreadNotifyCount(resource.workerCount, pipelineDepth);
    if (resource.workerCount == 0 || resource.threads.size() != resource.workerCount + 1 ||
        resource.aicpuThread != resource.threads[0] || requiredThreadNotify > UINT32_MAX ||
        resource.notifyNumPerThread < requiredThreadNotify ||
        resource.channelNotifyNum <= NOTIFY_IDX_DATA_SIGNAL) {
        return false;
    }

    uint32_t previousRank = INVALID_VALUE_RANKID;
    for (const auto &channel : resource.channels) {
        if (channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank ||
            (previousRank != INVALID_VALUE_RANKID && channel.remoteRank <= previousRank) ||
            channel.workerIndex == 0 || channel.workerIndex > resource.workerCount ||
            channel.notifyNum <= NOTIFY_IDX_DATA_SIGNAL || channel.remoteCclMem.addr == nullptr) {
            return false;
        }
        previousRank = channel.remoteRank;
    }
    return true;
}

HcclResult BuildExecutionPlan(const OpParam &param, const AlgResourceCtx &resource, ExecutionPlan &plan)
{
    uint64_t totalBytes = 0;
    CHK_RET(ValidateExecutionContext(param, resource, totalBytes));
    plan.totalBytes = totalBytes;
    if (totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    // 同一个槽位公式会用于本地 Buffer 和所有远端 Buffer，因此容量必须取最小值。
    uint64_t minRemoteBuffer = GetMinRemoteBufferSize(resource);
    uint64_t commonBufferBytes = std::min(resource.localBuffer.size, minRemoteBuffer);
    if (commonBufferBytes == 0) {
        HCCL_ERROR("invalid HCCL buffer size, local=%lu remoteMin=%lu", resource.localBuffer.size, minRemoteBuffer);
        return HCCL_E_PARA;
    }

    uint64_t elementSize = SIZE_TABLE.find(param.dataType)->second;
    uint64_t directTileBytes = AlignDown(std::min(AlgorithmConfig::kPreferredTileBytes, commonBufferBytes), elementSize);
    if (directTileBytes == 0) {
        HCCL_ERROR("HCCL buffer too small, common=%lu elementSize=%lu", commonBufferBytes, elementSize);
        return HCCL_E_PARA;
    }

    plan.ownerCount = param.rankSize - 1;
    uint32_t pipelineDepth = std::max<uint32_t>(1,
        std::min(resource.pipelineDepth, AlgorithmConfig::kPreferredPipelineDepth));
    bool parallelDirectRequested = totalBytes >= AlgorithmConfig::kParallelDirectMinBytes &&
        totalBytes <= AlgorithmConfig::kParallelDirectMaxBytes;
    if (parallelDirectRequested) {
        if (totalBytes <= commonBufferBytes && HasParallelDirectResources(param, resource, pipelineDepth)) {
            plan.algorithm = AlgorithmKind::PARALLEL_DIRECT_FANOUT;
            // StartAllWorkers 必须继续使用按 pipelineDepth=2 资源布局末尾申请的
            // 独立 launch Notify，不能退回到可能与 window Notify 重叠的索引。
            plan.pipelineDepth = pipelineDepth;
            plan.tileBytes = totalBytes;
            plan.stripeCount = 1;
            return HCCL_SUCCESS;
        }

        HCCL_WARNING("parallel direct resources unavailable, fallback to serial direct, bytes=%lu "
                     "commonBuffer=%lu workers=%u threads=%lu channelNotify=%u threadNotify=%u",
            totalBytes, commonBufferBytes, resource.workerCount, resource.threads.size(),
            resource.channelNotifyNum, resource.notifyNumPerThread);
    }

    bool distributedRequested = totalBytes > AlgorithmConfig::kParallelDirectMaxBytes;
    if (distributedRequested) {
        // Buffer 布局是 [window][owner][tile]。Tile 大小不能超过单槽位容量。
        uint64_t slotCount = static_cast<uint64_t>(plan.ownerCount) * pipelineDepth;
        uint64_t slotCapacity = slotCount == 0 ? 0 : commonBufferBytes / slotCount;
        uint64_t distributedTileBytes = AlignDown(
            std::min(AlgorithmConfig::kPreferredTileBytes, slotCapacity), elementSize);

        if (HasDistributedResources(param, resource, pipelineDepth) &&
            distributedTileBytes >= AlgorithmConfig::kMinimumTileBytes) {
            plan.algorithm = AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT;
            plan.pipelineDepth = pipelineDepth;
            plan.tileBytes = distributedTileBytes;
            // Tile 按 owner 轮转分配；ownerCount 个 Tile 构成一个 stripe。
            uint64_t tileCount = DivideRoundUp(totalBytes, plan.tileBytes);
            plan.stripeCount = DivideRoundUp(tileCount, plan.ownerCount);
            return HCCL_SUCCESS;
        }

        HCCL_WARNING("distributed resources unavailable, fallback to direct fanout, workers=%u threads=%lu "
                     "channelNotify=%u threadNotify=%u localBuffer=%lu remoteMin=%lu",
            resource.workerCount, resource.threads.size(), resource.channelNotifyNum,
            resource.notifyNumPerThread, resource.localBuffer.size, minRemoteBuffer);
    }

    plan.algorithm = AlgorithmKind::DIRECT_FANOUT;
    plan.pipelineDepth = 1;
    plan.tileBytes = directTileBytes;
    plan.stripeCount = DivideRoundUp(totalBytes, plan.tileBytes);
    return HCCL_SUCCESS;
}

TileDesc MakeTileDesc(const OpParam &param, const ExecutionPlan &plan, uint64_t stripeIndex,
    uint32_t ownerIndex)
{
    TileDesc tile;
    tile.ownerIndex = ownerIndex;
    tile.ownerRank = OwnerIndexToRank(ownerIndex, param.root);
    tile.windowIndex = static_cast<uint32_t>(stripeIndex % plan.pipelineDepth);

    if (stripeIndex > (UINT64_MAX - ownerIndex) / plan.ownerCount) {
        return tile;
    }
    // Block-cyclic 映射：tile 0..14 属于 owner 0..14，tile 15 再回到 owner 0。
    uint64_t globalTileIndex = stripeIndex * plan.ownerCount + ownerIndex;
    if (globalTileIndex > UINT64_MAX / plan.tileBytes) {
        return tile;
    }
    tile.offset = globalTileIndex * plan.tileBytes;
    if (tile.offset >= plan.totalBytes) {
        return tile;
    }
    // 最后一个 Tile 可以只有 4 字节，不能按固定 tileBytes 访问越界。
    tile.bytes = std::min(plan.tileBytes, plan.totalBytes - tile.offset);
    tile.valid = true;
    return tile;
}

HcclResult GetSlotAddress(const AlgResourceCtx &resource, const ExecutionPlan &plan, const TileDesc &tile,
    void *bufferBase, uint64_t bufferSize, void *&slot)
{
    // 每个 window 中每个 owner 有独立槽位，多个 owner 因此可以并行写同一 rank，
    // 又不会互相覆盖。窗口复用前由 SLOT_CONSUMED 协议保证旧数据已用完。
    uint64_t slotIndex = static_cast<uint64_t>(tile.windowIndex) * plan.ownerCount + tile.ownerIndex;
    if (slotIndex > UINT64_MAX / plan.tileBytes) {
        HCCL_ERROR("slot offset overflow, window=%u owner=%u tile=%lu", tile.windowIndex,
            tile.ownerIndex, plan.tileBytes);
        return HCCL_E_PARA;
    }
    uint64_t slotOffset = slotIndex * plan.tileBytes;
    if (slotOffset > bufferSize || tile.bytes > bufferSize - slotOffset) {
        HCCL_ERROR("slot out of range, window=%u owner=%u offset=%lu bytes=%lu buffer=%lu local=%lu",
            tile.windowIndex, tile.ownerIndex, slotOffset, tile.bytes, bufferSize, resource.localBuffer.size);
        return HCCL_E_PARA;
    }
    slot = OffsetPtr(bufferBase, slotOffset);
    return HCCL_SUCCESS;
}

HcclResult SubmitSendTile(ThreadHandle thread, const ChannelInfo &channel, const AlgResourceCtx &resource,
    const ExecutionPlan &plan, const TileDesc &tile)
{
    void *localSlot = nullptr;
    void *remoteSlot = nullptr;
    CHK_RET(GetSlotAddress(resource, plan, tile, resource.localBuffer.addr, resource.localBuffer.size, localSlot));
    CHK_RET(GetSlotAddress(resource, plan, tile, channel.remoteCclMem.addr, channel.remoteCclMem.size, remoteSlot));
    // Peer 槽位首次使用时天然空闲；同一 window 再次使用前，上一次调用末尾
    // 已等待 SLOT_CONSUMED。因此发送前不再增加一组 Record/Wait，避免任务图
    // 以跨 rank Wait 开头。Write + Fence 完成后才发布 DATA_READY。
    CHK_RET(CheckHcommRet(HcommWriteOnThread(thread, channel.handle, remoteSlot, localSlot, tile.bytes),
        "HcommWriteOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelFenceOnThread(thread, channel.handle), "HcommChannelFenceOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, channel.handle,
        DataReadyNotifyIndex(tile.windowIndex)), "HcommChannelNotifyRecordOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
        SlotConsumedNotifyIndex(plan, tile.windowIndex), CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
    return HCCL_SUCCESS;
}

HcclResult SubmitReceiveTileData(ThreadHandle thread, const ChannelInfo &channel, const OpParam &param,
    const AlgResourceCtx &resource, const ExecutionPlan &plan, const TileDesc &tile)
{
    void *localSlot = nullptr;
    CHK_RET(GetSlotAddress(resource, plan, tile, resource.localBuffer.addr, resource.localBuffer.size, localSlot));
    // 接收协议的前半段：先允许对端写入，再等 DATA_READY，最后把本地 HCCL
    // 槽位复制到用户 Buffer 的最终 offset。完成 ACK 由调用者在合适时机发送。
    CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, channel.handle,
        SlotConsumedNotifyIndex(plan, tile.windowIndex)), "HcommChannelNotifyRecordOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
        DataReadyNotifyIndex(tile.windowIndex), CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
    CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(thread, OffsetPtr(param.outputPtr, tile.offset),
        localSlot, tile.bytes), "HcommLocalCopyOnThread"));
    return HCCL_SUCCESS;
}

HcclResult SubmitReceiveTile(ThreadHandle thread, const ChannelInfo &channel, const OpParam &param,
    const AlgResourceCtx &resource, const ExecutionPlan &plan, const TileDesc &tile)
{
    void *localSlot = nullptr;
    CHK_RET(GetSlotAddress(resource, plan, tile, resource.localBuffer.addr,
        resource.localBuffer.size, localSlot));
    // Peer 接收方只需等待 DATA_READY。复制完成后的 SLOT_CONSUMED 既确认本轮
    // 数据已使用，也为发送方下一次复用同一个 window 提供许可。
    CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
        DataReadyNotifyIndex(tile.windowIndex), CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
    CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(thread, OffsetPtr(param.outputPtr, tile.offset),
        localSlot, tile.bytes), "HcommLocalCopyOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, channel.handle,
        SlotConsumedNotifyIndex(plan, tile.windowIndex)), "HcommChannelNotifyRecordOnThread"));
    return HCCL_SUCCESS;
}

HcclResult SubmitBidirectionalPeerExchange(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan, ThreadHandle worker, const ChannelInfo &channel,
    const TileDesc &ownTile, const TileDesc &peerTile)
{
    void *ownLocalSlot = nullptr;
    void *ownRemoteSlot = nullptr;
    void *peerLocalSlot = nullptr;
    CHK_RET(GetSlotAddress(resource, plan, ownTile, resource.localBuffer.addr,
        resource.localBuffer.size, ownLocalSlot));
    CHK_RET(GetSlotAddress(resource, plan, ownTile, channel.remoteCclMem.addr,
        channel.remoteCclMem.size, ownRemoteSlot));
    CHK_RET(GetSlotAddress(resource, plan, peerTile, resource.localBuffer.addr,
        resource.localBuffer.size, peerLocalSlot));

    // 双方执行完全相同的顺序，并且在任何跨 rank Wait 之前先发送自己的
    // Tile。首次使用的 peer 槽位天然空闲；窗口复用由本函数末尾等待上一轮
    // SLOT_CONSUMED 保证。Fence 建立 Write 与 DATA_READY 的先后关系。
    CHK_RET(CheckHcommRet(HcommWriteOnThread(worker, channel.handle, ownRemoteSlot,
        ownLocalSlot, ownTile.bytes), "HcommWriteOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelFenceOnThread(worker, channel.handle),
        "HcommChannelFenceOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(worker, channel.handle,
        DataReadyNotifyIndex(ownTile.windowIndex)), "HcommChannelNotifyRecordOnThread"));

    // 收到对端 Tile 后复制到最终用户 Buffer，并发送 SLOT_CONSUMED。最后的
    // Wait 保证 ownTile 对应的远端槽位已被消费，下一轮才可覆盖该 window。
    CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(worker, channel.handle,
        DataReadyNotifyIndex(peerTile.windowIndex), CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
    CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(worker, OffsetPtr(param.outputPtr, peerTile.offset),
        peerLocalSlot, peerTile.bytes), "HcommLocalCopyOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(worker, channel.handle,
        SlotConsumedNotifyIndex(plan, peerTile.windowIndex)), "HcommChannelNotifyRecordOnThread"));
    CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(worker, channel.handle,
        SlotConsumedNotifyIndex(plan, ownTile.windowIndex), CUSTOM_TIMEOUT),
        "HcommChannelNotifyWaitOnThread"));
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectRoot(const OpParam &param, const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    // 串行 Direct 使用主线程和 localBuffer 的第一个槽位。它既是 4B 等
    // 极小包的低开销路径，也是资源不足或整包放不进 Buffer 时的分块保底路径。
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t stripe = 0; stripe < plan.stripeCount; ++stripe) {
        uint64_t offset = stripe * plan.tileBytes;
        uint64_t bytes = std::min(plan.tileBytes, plan.totalBytes - offset);
        void *localSlot = resource.localBuffer.addr;

        CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(thread, localSlot, OffsetPtr(param.inputPtr, offset), bytes),
            "HcommLocalCopyOnThread"));
        for (const auto &channel : resource.channels) {
            CHK_PTR_NULL(channel.remoteCclMem.addr);
            if (channel.remoteCclMem.size < bytes) {
                HCCL_ERROR("remote HCCL buffer too small, remoteRank=%u size=%lu bytes=%lu",
                    channel.remoteRank, channel.remoteCclMem.size, bytes);
                return HCCL_E_PARA;
            }
            CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
                NOTIFY_IDX_ACK, CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
            CHK_RET(CheckHcommRet(HcommWriteOnThread(thread, channel.handle, channel.remoteCclMem.addr,
                localSlot, bytes), "HcommWriteOnThread"));
            CHK_RET(CheckHcommRet(HcommChannelFenceOnThread(thread, channel.handle),
                "HcommChannelFenceOnThread"));
            CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, channel.handle,
                NOTIFY_IDX_DATA_SIGNAL), "HcommChannelNotifyRecordOnThread"));
            CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
                NOTIFY_IDX_ACK, CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectPeer(const OpParam &param, const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    const ChannelInfo *rootChannel = FindChannelByRemoteRank(resource, param.root);
    CHK_PTR_NULL(rootChannel);

    // 非 root 只与 root 握手：先声明槽位空闲，收到 DATA_READY 后复制到用户
    // Buffer，再返回 consumed ACK。
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t stripe = 0; stripe < plan.stripeCount; ++stripe) {
        uint64_t offset = stripe * plan.tileBytes;
        uint64_t bytes = std::min(plan.tileBytes, plan.totalBytes - offset);
        CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle,
            NOTIFY_IDX_ACK), "HcommChannelNotifyRecordOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, rootChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
        CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(thread, OffsetPtr(param.outputPtr, offset),
            resource.localBuffer.addr, bytes), "HcommLocalCopyOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle,
            NOTIFY_IDX_ACK), "HcommChannelNotifyRecordOnThread"));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectFanout(const OpParam &param, const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    if (plan.totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    return param.myRank == param.root ? ExecuteDirectRoot(param, resource, plan) :
                                       ExecuteDirectPeer(param, resource, plan);
}

HcclResult NotifyWorkerDone(ThreadHandle worker, ThreadHandle mainThread, uint32_t workerIndex)
{
    // workerIndex 同时作为主线程上的完成 Notify 下标，因此各 worker 不会串信号。
    return CheckHcommRet(HcommThreadNotifyRecordOnThread(worker, mainThread, workerIndex),
        "HcommThreadNotifyRecordOnThread");
}

HcclResult StartAllWorkers(const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    // Checker 和真实 TS 都从 AICPU 主线程开始推进任务。显式建立
    // main -> worker 的启动边，否则主线程首先执行 WaitAllWorkers，而 worker
    // 流只有末尾的 worker -> main 完成边，整个子图会形成不可达的闭环。
    //
    // 逐 stripe 的 start/done Notify 已按 window 隔离；一次性启动使用布局末尾
    // 的独立索引，避免与任何 window 信号冲突。
    uint32_t launchNotifyIndex = WorkerLaunchNotifyIndex(resource.workerCount, plan.pipelineDepth);
    if (launchNotifyIndex >= resource.notifyNumPerThread) {
        HCCL_ERROR("worker launch notify out of range, index=%u count=%u",
            launchNotifyIndex, resource.notifyNumPerThread);
        return HCCL_E_PARA;
    }
    for (uint32_t workerIndex = 1; workerIndex <= resource.workerCount; ++workerIndex) {
        CHK_RET(CheckHcommRet(HcommThreadNotifyRecordOnThread(resource.aicpuThread,
            resource.threads[workerIndex], launchNotifyIndex), "HcommThreadNotifyRecordOnThread"));
    }
    for (uint32_t workerIndex = 1; workerIndex <= resource.workerCount; ++workerIndex) {
        CHK_RET(CheckHcommRet(HcommThreadNotifyWaitOnThread(resource.threads[workerIndex],
            launchNotifyIndex, CUSTOM_TIMEOUT), "HcommThreadNotifyWaitOnThread"));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitAllWorkers(const AlgResourceCtx &resource)
{
    // 主线程必须等所有 worker 的最后一条任务完成，才能通知 Host 本次 Kernel
    // 编排结束，否则后续调用可能复用仍在工作的资源。
    for (uint32_t workerIndex = 1; workerIndex <= resource.workerCount; ++workerIndex) {
        CHK_RET(CheckHcommRet(HcommThreadNotifyWaitOnThread(resource.aicpuThread, workerIndex, CUSTOM_TIMEOUT),
            "HcommThreadNotifyWaitOnThread"));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteParallelDirectRoot(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan)
{
    ThreadHandle mainThread = resource.aicpuThread;
    void *localBuffer = resource.localBuffer.addr;

    // 共享源 Buffer 只由主线程写一次。StartAllWorkers 中所有 main -> worker
    // launch Record 都排在这条 LocalCopy 之后，因此每个 worker 读取前都有
    // 明确的同线程顺序边；worker 自身只读 localBuffer，不会覆盖共享数据。
    CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(mainThread, localBuffer,
        param.inputPtr, plan.totalBytes), "HcommLocalCopyOnThread"));
    CHK_RET(StartAllWorkers(resource, plan));

    // 每条 Channel 使用资源规划时已经绑定的 worker。当前 16 rank 默认一条
    // Channel 对应一个 worker；把 Host 侧 kMaxWorkerCount 调为 4/8 时，多条
    // Channel 会按 workerIndex 分组，但不同 worker 之间仍然并行。
    for (const auto &channel : resource.channels) {
        ThreadHandle worker = resource.threads[channel.workerIndex];
        CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(worker, channel.handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
        CHK_RET(CheckHcommRet(HcommWriteOnThread(worker, channel.handle,
            channel.remoteCclMem.addr, localBuffer, plan.totalBytes), "HcommWriteOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelFenceOnThread(worker, channel.handle),
            "HcommChannelFenceOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(worker, channel.handle,
            NOTIFY_IDX_DATA_SIGNAL), "HcommChannelNotifyRecordOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(worker, channel.handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT), "HcommChannelNotifyWaitOnThread"));
    }

    // 每个已启动 worker 恰好发送一个唯一完成 Notify。即使后续把 worker 数
    // 调大、最后有 worker 没分到 Channel，也仍会产生与 WaitAllWorkers 匹配
    // 的完成信号，不会留下悬空 Wait。
    for (uint32_t workerIndex = 1; workerIndex <= resource.workerCount; ++workerIndex) {
        CHK_RET(NotifyWorkerDone(resource.threads[workerIndex], mainThread, workerIndex));
    }
    return WaitAllWorkers(resource);
}

HcclResult ExecuteParallelDirectFanout(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan)
{
    if (plan.totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    // 非 root 暂时保持原 Direct 协议：初始 ACK -> DATA_READY -> LocalCopy ->
    // 最终 ACK。只有 root 的 15 条发送 Channel 被分散到 worker。
    return param.myRank == param.root ? ExecuteParallelDirectRoot(param, resource, plan) :
                                       ExecuteDirectPeer(param, resource, plan);
}

HcclResult ExecuteDistributedRoot(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan)
{
    CHK_RET(StartAllWorkers(resource, plan));
    // 第一阶段：root Scatter。
    // 每个 root worker 只负责一个 owner：从用户 buf 取出该 owner 的 block-cyclic
    // Tile，写入 owner rank 对应的远端槽位。所有 owner worker 可并行执行。
    for (const auto &channel : resource.channels) {
        uint32_t ownerIndex = 0;
        if (!RankToOwnerIndex(channel.remoteRank, param.root, ownerIndex)) {
            return HCCL_E_INTERNAL;
        }
        ThreadHandle worker = resource.threads[channel.workerIndex];
        for (uint64_t baseStripe = 0; baseStripe < plan.stripeCount; baseStripe += plan.pipelineDepth) {
            // 一个 batch 最多包含 pipelineDepth 个 stripe，对应不同窗口。
            uint64_t batchEnd = std::min(plan.stripeCount, baseStripe + plan.pipelineDepth);
            // 本批所有有效 Tile 发出后，再等待 consumed，保证下一批复用窗口安全。
            for (uint64_t stripe = baseStripe; stripe < batchEnd; ++stripe) {
                TileDesc tile = MakeTileDesc(param, plan, stripe, ownerIndex);
                if (!tile.valid) {
                    continue;
                }
                void *localSlot = nullptr;
                CHK_RET(GetSlotAddress(resource, plan, tile, resource.localBuffer.addr,
                    resource.localBuffer.size, localSlot));
                CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(worker, localSlot,
                    OffsetPtr(param.inputPtr, tile.offset), tile.bytes), "HcommLocalCopyOnThread"));

                void *remoteSlot = nullptr;
                CHK_RET(GetSlotAddress(resource, plan, tile, channel.remoteCclMem.addr,
                    channel.remoteCclMem.size, remoteSlot));
                CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(worker, channel.handle,
                    SlotConsumedNotifyIndex(plan, tile.windowIndex), CUSTOM_TIMEOUT),
                    "HcommChannelNotifyWaitOnThread"));
                CHK_RET(CheckHcommRet(HcommWriteOnThread(worker, channel.handle, remoteSlot,
                    localSlot, tile.bytes), "HcommWriteOnThread"));
                CHK_RET(CheckHcommRet(HcommChannelFenceOnThread(worker, channel.handle),
                    "HcommChannelFenceOnThread"));
                CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(worker, channel.handle,
                    DataReadyNotifyIndex(tile.windowIndex)), "HcommChannelNotifyRecordOnThread"));
            }
            for (uint64_t stripe = baseStripe; stripe < batchEnd; ++stripe) {
                TileDesc tile = MakeTileDesc(param, plan, stripe, ownerIndex);
                if (!tile.valid) {
                    continue;
                }
                CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(worker, channel.handle,
                    SlotConsumedNotifyIndex(plan, tile.windowIndex), CUSTOM_TIMEOUT),
                    "HcommChannelNotifyWaitOnThread"));
            }
        }
        CHK_RET(NotifyWorkerDone(worker, resource.aicpuThread, channel.workerIndex));
    }
    return WaitAllWorkers(resource);
}

HcclResult SubmitPeerExchange(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan, ThreadHandle worker, const ChannelInfo &channel,
    const TileDesc &ownTile, const TileDesc &peerTile)
{
    if (ownTile.valid && peerTile.valid) {
        CHK_RET(SubmitBidirectionalPeerExchange(param, resource, plan, worker, channel,
            ownTile, peerTile));
    } else if (ownTile.valid) {
        // 最后一个 stripe 可能只有一侧有 Tile，只提交真实存在的一半协议。
        CHK_RET(SubmitSendTile(worker, channel, resource, plan, ownTile));
    } else if (peerTile.valid) {
        CHK_RET(SubmitReceiveTile(worker, channel, param, resource, plan, peerTile));
    }
    return HCCL_SUCCESS;
}

HcclResult SubmitOwnerStripeStart(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan, uint32_t ownOwnerIndex, const ChannelInfo &rootChannel,
    uint64_t stripe)
{
    ThreadHandle coordinator = resource.threads[rootChannel.workerIndex];
    // coordinator 先从 root 接收本 owner 的 Tile，并复制到最终用户 Buffer。
    // 此时不能向 root 返回 consumed，因为 peer worker 还要读取同一个 owner 槽位。
    TileDesc ownTile = MakeTileDesc(param, plan, stripe, ownOwnerIndex);
    if (ownTile.valid) {
        CHK_RET(SubmitReceiveTileData(coordinator, rootChannel, param, resource, plan, ownTile));
    }
    // 无论本 stripe 的 ownTile 是否有效，都启动所有 peer worker。最后一个
    // stripe 中 worker 会根据 ownTile/peerTile.valid 只执行需要的一侧协议，
    // 然后仍然返回完成信号，避免 coordinator 产生悬空 Wait。
    for (const auto &channel : resource.channels) {
        if (channel.remoteRank == param.root) {
            continue;
        }
        CHK_RET(CheckHcommRet(HcommThreadNotifyRecordOnThread(coordinator,
            resource.threads[channel.workerIndex], WorkerStartNotifyIndex(ownTile.windowIndex)),
            "HcommThreadNotifyRecordOnThread"));
    }
    return HCCL_SUCCESS;
}

HcclResult SubmitOwnerStripeFinish(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan, uint32_t ownOwnerIndex, const ChannelInfo &rootChannel,
    uint64_t stripe)
{
    ThreadHandle coordinator = resource.threads[rootChannel.workerIndex];
    TileDesc ownTile = MakeTileDesc(param, plan, stripe, ownOwnerIndex);
    // 等 14 个 peer worker 都结束本 stripe，意味着：
    //   - ownTile 已经 fanout 给所有其他非 root rank；
    //   - 从其他 owner 接收的 Tile 也已复制到本 rank 用户 Buffer。
    for (const auto &channel : resource.channels) {
        if (channel.remoteRank == param.root) {
            continue;
        }
        CHK_RET(CheckHcommRet(HcommThreadNotifyWaitOnThread(coordinator,
            WorkerDoneNotifyIndex(plan, channel.workerIndex, ownTile.windowIndex), CUSTOM_TIMEOUT),
            "HcommThreadNotifyWaitOnThread"));
    }
    if (ownTile.valid) {
        // 到这里已没有 worker 继续读取 ownTile 槽位，可以安全允许 root 覆盖它。
        CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(coordinator, rootChannel.handle,
            SlotConsumedNotifyIndex(plan, ownTile.windowIndex)), "HcommChannelNotifyRecordOnThread"));
    }
    return HCCL_SUCCESS;
}

HcclResult SubmitPeerWorkerStripe(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan, uint32_t ownOwnerIndex, const ChannelInfo &channel,
    ThreadHandle coordinator, uint64_t stripe)
{
    uint32_t peerOwnerIndex = 0;
    if (!RankToOwnerIndex(channel.remoteRank, param.root, peerOwnerIndex)) {
        return HCCL_E_INTERNAL;
    }
    ThreadHandle worker = resource.threads[channel.workerIndex];
    TileDesc ownTile = MakeTileDesc(param, plan, stripe, ownOwnerIndex);
    TileDesc peerTile = MakeTileDesc(param, plan, stripe, peerOwnerIndex);
    // 每条 peer Channel 独占一个 worker。先等 coordinator 确认自己的 owner Tile
    // 已就绪，再与该 peer 同时完成“发送 ownTile + 接收 peerTile”。
    CHK_RET(CheckHcommRet(HcommThreadNotifyWaitOnThread(worker,
        WorkerStartNotifyIndex(ownTile.windowIndex), CUSTOM_TIMEOUT), "HcommThreadNotifyWaitOnThread"));
    CHK_RET(SubmitPeerExchange(param, resource, plan, worker, channel, ownTile, peerTile));
    return CheckHcommRet(HcommThreadNotifyRecordOnThread(worker, coordinator,
        WorkerDoneNotifyIndex(plan, channel.workerIndex, ownTile.windowIndex)),
        "HcommThreadNotifyRecordOnThread");
}

HcclResult ExecuteDistributedOwner(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan)
{
    uint32_t ownOwnerIndex = 0;
    if (!RankToOwnerIndex(param.myRank, param.root, ownOwnerIndex)) {
        return HCCL_E_INTERNAL;
    }
    const ChannelInfo *rootChannel = FindChannelByRemoteRank(resource, param.root);
    CHK_PTR_NULL(rootChannel);
    ThreadHandle coordinator = resource.threads[rootChannel->workerIndex];
    CHK_RET(StartAllWorkers(resource, plan));

    // 第二阶段：owner Fanout。
    //
    // 必须按“有界 batch”交错地向 coordinator 和全部 peer worker 提交任务。
    // Ascend 950 可能在 BatchModeEnd 之前就把某条流积累的 SQE 提前下发；如果
    // 先把 coordinator 的所有 Wait 全部塞完、最后才提交 worker 的 Record，真实
    // 硬件会出现 SQ 反压死锁。下面三个阶段确保释放当前 Wait 的任务已及时进入
    // 对应 worker 流：
    //   A. coordinator 接收本批 Tile，并启动 worker；
    //   B. 所有 worker 提交 pairwise fanout/receive；
    //   C. coordinator 等 worker 完成，再向 root 归还槽位。
    for (uint64_t baseStripe = 0; baseStripe < plan.stripeCount; baseStripe += plan.pipelineDepth) {
        uint64_t batchEnd = std::min(plan.stripeCount, baseStripe + plan.pipelineDepth);
        // 阶段 A：准备本批最多两个窗口。
        for (uint64_t stripe = baseStripe; stripe < batchEnd; ++stripe) {
            CHK_RET(SubmitOwnerStripeStart(param, resource, plan, ownOwnerIndex, *rootChannel, stripe));
        }
        // 阶段 B：把每个 stripe 的点对点任务分散到 14 条 worker 流。
        for (uint64_t stripe = baseStripe; stripe < batchEnd; ++stripe) {
            for (const auto &channel : resource.channels) {
                if (channel.remoteRank == param.root) {
                    continue;
                }
                CHK_RET(SubmitPeerWorkerStripe(param, resource, plan, ownOwnerIndex,
                    channel, coordinator, stripe));
            }
        }
        // 阶段 C：Drain 本批窗口，确认后才允许下一批复用相同槽位。
        for (uint64_t stripe = baseStripe; stripe < batchEnd; ++stripe) {
            CHK_RET(SubmitOwnerStripeFinish(param, resource, plan, ownOwnerIndex, *rootChannel, stripe));
        }
    }

    // 所有 stripe 编排完成后，每个 peer worker 和 coordinator 各向主线程发送
    // 一次最终完成通知；随后主线程统一 WaitAllWorkers。
    for (const auto &channel : resource.channels) {
        if (channel.remoteRank == param.root) {
            continue;
        }
        CHK_RET(NotifyWorkerDone(resource.threads[channel.workerIndex], resource.aicpuThread,
            channel.workerIndex));
    }
    CHK_RET(NotifyWorkerDone(coordinator, resource.aicpuThread, rootChannel->workerIndex));
    return WaitAllWorkers(resource);
}

HcclResult ExecuteDistributedScatterFanout(const OpParam &param, const AlgResourceCtx &resource,
    const ExecutionPlan &plan)
{
    if (plan.totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    // root 只执行 Scatter；其余 rank 同时扮演一个 owner 和所有其他 owner 的接收方。
    return param.myRank == param.root ? ExecuteDistributedRoot(param, resource, plan) :
                                       ExecuteDistributedOwner(param, resource, plan);
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    // 所有算法选择和动态 Tile 计算集中在 BuildExecutionPlan，执行函数不再自行
    // 猜测阈值或 Buffer 大小。
    ExecutionPlan plan;
    CHK_RET(BuildExecutionPlan(param, resCtx, plan));
    HCCL_INFO("broadcast plan rank=%u root=%u rankSize=%u bytes=%lu algorithm=%s tile=%lu stripes=%lu "
              "pipeline=%u workers=%u channels=%lu localBuffer=%lu",
        param.myRank, param.root, param.rankSize, plan.totalBytes, AlgorithmName(plan.algorithm),
        plan.tileBytes, plan.stripeCount, plan.pipelineDepth, resCtx.workerCount, resCtx.channels.size(),
        resCtx.localBuffer.size);

    switch (plan.algorithm) {
        case AlgorithmKind::DIRECT_FANOUT:
            return ExecuteDirectFanout(param, resCtx, plan);
        case AlgorithmKind::PARALLEL_DIRECT_FANOUT:
            return ExecuteParallelDirectFanout(param, resCtx, plan);
        case AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT:
            return ExecuteDistributedScatterFanout(param, resCtx, plan);
        default:
            return HCCL_E_INTERNAL;
    }
}
} // namespace ops_hccl
