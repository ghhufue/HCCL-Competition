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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
enum class AlgorithmKind : uint32_t {
    DIRECT_FANOUT = 0,
    DISTRIBUTED_SCATTER_FANOUT = 1,
};

enum class ChannelNotifyIndex : uint32_t {
    SLOT_READY = NOTIFY_IDX_ACK,
    DATA_READY = NOTIFY_IDX_DATA_SIGNAL,
};

struct AlgorithmConfig {
    static constexpr uint64_t kDirectThresholdBytes = 1ULL << 20;
    static constexpr uint64_t kPreferredTileBytes = 4ULL << 20;
    static constexpr uint64_t kMinimumTileBytes = 4ULL << 10;
    static constexpr uint32_t kPreferredPipelineDepth = 1;
};

struct ExecutionPlan {
    AlgorithmKind algorithm = AlgorithmKind::DIRECT_FANOUT;
    uint64_t totalBytes = 0;
    uint64_t tileBytes = 0;
    uint64_t stripeCount = 0;
    uint32_t ownerCount = 0;
    uint32_t pipelineDepth = 1;
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

HcclResult ValidateExecutionContext(const OpParam &param, const AlgResourceCtx &resource, uint64_t &totalBytes)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resource.localBuffer.addr);

    if (param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize) {
        HCCL_ERROR("invalid rank info, myRank=%u root=%u rankSize=%u", param.myRank, param.root, param.rankSize);
        return HCCL_E_PARA;
    }
    if (resource.layoutVersion != static_cast<uint32_t>(ResourceLayoutVersion::VERSION_1)) {
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

HcclResult BuildExecutionPlan(const OpParam &param, const AlgResourceCtx &resource, ExecutionPlan &plan)
{
    uint64_t totalBytes = 0;
    CHK_RET(ValidateExecutionContext(param, resource, totalBytes));
    plan.totalBytes = totalBytes;
    if (totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    uint64_t minRemoteBuffer = GetMinRemoteBufferSize(resource);
    uint64_t maxTileBytes = std::min(resource.localBuffer.size, minRemoteBuffer);
    if (maxTileBytes == 0) {
        HCCL_ERROR("invalid HCCL buffer size, local=%lu remoteMin=%lu", resource.localBuffer.size, minRemoteBuffer);
        return HCCL_E_PARA;
    }

    auto iter = SIZE_TABLE.find(param.dataType);
    uint64_t elementSize = iter == SIZE_TABLE.end() ? 1 : iter->second;
    uint64_t tileBytes = std::min(AlgorithmConfig::kPreferredTileBytes, maxTileBytes);
    tileBytes = std::max(tileBytes, std::min(AlgorithmConfig::kMinimumTileBytes, maxTileBytes));
    tileBytes -= tileBytes % elementSize;
    if (tileBytes == 0) {
        HCCL_ERROR("HCCL buffer too small, maxTileBytes=%lu elementSize=%lu", maxTileBytes, elementSize);
        return HCCL_E_PARA;
    }

    plan.tileBytes = tileBytes;
    plan.ownerCount = param.rankSize - 1;
    plan.pipelineDepth = std::max<uint32_t>(1, std::min(resource.pipelineDepth, AlgorithmConfig::kPreferredPipelineDepth));
    plan.stripeCount = (totalBytes + tileBytes - 1) / tileBytes;
    plan.algorithm = totalBytes <= AlgorithmConfig::kDirectThresholdBytes ? AlgorithmKind::DIRECT_FANOUT :
                                                                                  AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT;

    if (plan.algorithm == AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT) {
        HCCL_WARNING("distributed algorithm is reserved in v1, fallback to direct fanout");
        plan.algorithm = AlgorithmKind::DIRECT_FANOUT;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectRoot(const OpParam &param, const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t stripe = 0; stripe < plan.stripeCount; ++stripe) {
        uint64_t offset = stripe * plan.tileBytes;
        uint64_t bytes = std::min(plan.tileBytes, plan.totalBytes - offset);
        void *localSlot = resource.localBuffer.addr;
        const void *src = OffsetPtr(param.inputPtr, offset);

        CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(thread, localSlot, src, bytes), "HcommLocalCopyOnThread"));
        for (const auto &channel : resource.channels) {
            CHK_PTR_NULL(channel.remoteCclMem.addr);
            if (channel.remoteCclMem.size < bytes) {
                HCCL_ERROR("remote HCCL buffer too small, remoteRank=%u size=%lu bytes=%lu",
                    channel.remoteRank, channel.remoteCclMem.size, bytes);
                return HCCL_E_PARA;
            }
            CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
                static_cast<uint32_t>(ChannelNotifyIndex::SLOT_READY), CUSTOM_TIMEOUT),
                "HcommChannelNotifyWaitOnThread"));
            CHK_RET(CheckHcommRet(HcommWriteOnThread(thread, channel.handle, channel.remoteCclMem.addr,
                localSlot, bytes), "HcommWriteOnThread"));
            CHK_RET(CheckHcommRet(HcommChannelFenceOnThread(thread, channel.handle), "HcommChannelFenceOnThread"));
            CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, channel.handle,
                static_cast<uint32_t>(ChannelNotifyIndex::DATA_READY)), "HcommChannelNotifyRecordOnThread"));
            CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, channel.handle,
                static_cast<uint32_t>(ChannelNotifyIndex::SLOT_READY), CUSTOM_TIMEOUT),
                "HcommChannelNotifyWaitOnThread"));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectPeer(const OpParam &param, const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    const ChannelInfo *rootChannel = FindChannelByRemoteRank(resource, param.root);
    CHK_PTR_NULL(rootChannel);

    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t stripe = 0; stripe < plan.stripeCount; ++stripe) {
        uint64_t offset = stripe * plan.tileBytes;
        uint64_t bytes = std::min(plan.tileBytes, plan.totalBytes - offset);
        void *dst = OffsetPtr(param.outputPtr, offset);

        CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle,
            static_cast<uint32_t>(ChannelNotifyIndex::SLOT_READY)), "HcommChannelNotifyRecordOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelNotifyWaitOnThread(thread, rootChannel->handle,
            static_cast<uint32_t>(ChannelNotifyIndex::DATA_READY), CUSTOM_TIMEOUT),
            "HcommChannelNotifyWaitOnThread"));
        CHK_RET(CheckHcommRet(HcommLocalCopyOnThread(thread, dst, resource.localBuffer.addr, bytes),
            "HcommLocalCopyOnThread"));
        CHK_RET(CheckHcommRet(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle,
            static_cast<uint32_t>(ChannelNotifyIndex::SLOT_READY)), "HcommChannelNotifyRecordOnThread"));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectFanout(const OpParam &param, const AlgResourceCtx &resource, const ExecutionPlan &plan)
{
    if (plan.totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (param.myRank == param.root) {
        return ExecuteDirectRoot(param, resource, plan);
    }
    return ExecuteDirectPeer(param, resource, plan);
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    ExecutionPlan plan;
    CHK_RET(BuildExecutionPlan(param, resCtx, plan));
    HCCL_INFO("broadcast plan rank=%u root=%u rankSize=%u bytes=%lu algorithm=%u tile=%lu stripes=%lu channels=%lu",
        param.myRank, param.root, param.rankSize, plan.totalBytes, static_cast<uint32_t>(plan.algorithm),
        plan.tileBytes, plan.stripeCount, resCtx.channels.size());

    switch (plan.algorithm) {
        case AlgorithmKind::DIRECT_FANOUT:
            return ExecuteDirectFanout(param, resCtx, plan);
        case AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT:
        default:
            return ExecuteDirectFanout(param, resCtx, plan);
    }
}
} // namespace ops_hccl
