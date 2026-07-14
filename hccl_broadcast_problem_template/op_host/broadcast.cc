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
#include <cstdio>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
// 资源 Tag 只描述静态资源布局，不包含 root/count/buf。这样不同调用可以复用
// Thread、Channel 和 Engine Context。布局字段变化时必须同步升级 Tag 和版本号。
constexpr char kResourceTag[] = "hccl_custom_broadcast_v2";

// 双缓冲的 Channel Notify 布局：
//   [0, 1] -> 两个窗口的 DATA_READY
//   [2, 3] -> 两个窗口的 SLOT_CONSUMED
constexpr uint32_t kChannelNotifyNum = 4;
constexpr uint32_t kMinimumThreadNotifyNum = 1;
constexpr uint32_t kPipelineDepth = 2;
// 题目拓扑最多 16 rank，因此最多需要 15 个 peer worker。
constexpr uint32_t kMaxWorkerCount = 15;
constexpr uint32_t kMaxNetLayer = 3;

// Host 侧资源预算。它只决定“申请多少资源”，实际使用 Direct 还是 Distributed
// 由 AICPU 侧根据本次 totalBytes 和 Buffer 容量决定。
struct ResourcePlan {
    uint32_t threadNum = 1;
    uint32_t workerCount = 0;
    uint32_t notifyNumPerThread = kMinimumThreadNotifyNum;
    uint32_t channelNotifyNum = kChannelNotifyNum;
    uint32_t pipelineDepth = kPipelineDepth;
};

HcclResult ValidateBroadcastParam(const OpParam &param)
{
    if (param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize) {
        HCCL_ERROR("invalid rank info, myRank=%u root=%u rankSize=%u", param.myRank, param.root, param.rankSize);
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
    return HCCL_SUCCESS;
}

ResourcePlan BuildResourcePlan(uint32_t rankSize)
{
    ResourcePlan plan;
    if (rankSize > 1) {
        // 当前设计为每个远端 rank 分配一个 worker，便于 15 个 owner/peer 并发。
        // 如果将来 rankSize 增大，超过上限的 Channel 需要改为共享 worker。
        plan.workerCount = std::min(rankSize - 1, kMaxWorkerCount);
        plan.threadNum = plan.workerCount + 1;
        // 每个 worker 的 Notify 0 用作启动信号；主线程上的 [1, workerCount]
        // 分别接收对应 worker 的完成信号。所有线程统一按最大数量申请。
        plan.notifyNumPerThread = plan.workerCount + 1;
    }
    return plan;
}

// rank graph 可能在多个网络层为同一对 rank 提供 Link。按层从近到远选择
// 第一个真实协议，避免把 RESERVED 占位 Link 用来创建 Channel。
HcclResult QueryBestLinkToPeer(HcclComm comm, uint32_t localRank, uint32_t remoteRank, CommLink &selectedLink)
{
    HcclResult lastRet = HCCL_E_NOT_FOUND;
    for (uint32_t layer = 0; layer < kMaxNetLayer; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, localRank, remoteRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS) {
            lastRet = ret;
            continue;
        }
        for (uint32_t idx = 0; idx < linkNum; ++idx) {
            if (links[idx].linkAttr.linkProtocol != COMM_PROTOCOL_RESERVED) {
                selectedLink = links[idx];
                return HCCL_SUCCESS;
            }
        }
        lastRet = HCCL_E_NOT_FOUND;
    }
    HCCL_ERROR("failed to query link, localRank=%u remoteRank=%u ret=%d", localRank, remoteRank, lastRet);
    return lastRet;
}

HcclResult AcquireThreads(HcclComm comm, CommEngine engine, const ResourcePlan &plan, AlgResourceCtx &resource)
{
    // threads[0] 固定为 AICPU 主线程，后续 Channel worker 从 1 开始编号。
    resource.threads.resize(plan.threadNum);
    CHK_RET(HcclThreadAcquire(comm, engine, plan.threadNum, plan.notifyNumPerThread, resource.threads.data()));
    resource.aicpuThread = resource.threads[0];
    resource.workerCount = plan.workerCount;
    resource.notifyNumPerThread = plan.notifyNumPerThread;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, CommEngine engine, const ResourcePlan &plan,
    AlgResourceCtx &resource)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channelDescs(param.rankSize - 1);
    std::vector<uint32_t> remoteRanks;
    remoteRanks.reserve(param.rankSize - 1);
    CHK_RET(HcclChannelDescInit(channelDescs.data(), static_cast<uint32_t>(channelDescs.size())));

    // 按 remoteRank 升序构造描述符，因此序列化后的 channels 也保持稳定顺序。
    // Direct 只使用 root 相关 Channel；Distributed 会使用全部非 root peer Channel。
    uint32_t descIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        CommLink selectedLink;
        CHK_RET(QueryBestLinkToPeer(comm, param.myRank, remoteRank, selectedLink));

        HcclChannelDesc &desc = channelDescs[descIndex];
        desc.remoteRank = remoteRank;
        desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        desc.localEndpoint = selectedLink.srcEndpointDesc;
        desc.remoteEndpoint = selectedLink.dstEndpointDesc;
        desc.notifyNum = plan.channelNotifyNum;
        remoteRanks.push_back(remoteRank);
        ++descIndex;
    }

    std::vector<ChannelHandle> handles(channelDescs.size());
    CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()),
        handles.data()));

    resource.channels.resize(handles.size());
    for (uint32_t idx = 0; idx < handles.size(); ++idx) {
        ChannelInfo &channel = resource.channels[idx];
        channel.remoteRank = remoteRanks[idx];
        // 当前 workerCount == rankSize - 1，因此通常是一条 Channel 对应一个 worker。
        channel.workerIndex = plan.workerCount == 0 ? 0 : 1 + (idx % plan.workerCount);
        channel.notifyNum = plan.channelNotifyNum;
        channel.handle = handles[idx];
        CHK_RET(HcclChannelGetHcclBuffer(comm, channel.handle, &channel.remoteCclMem.addr,
            &channel.remoteCclMem.size));
    }
    return HCCL_SUCCESS;
}

HcclResult CreateAndStoreEngineContext(HcclComm comm, OpParam &param, CommEngine aicpuTsEngine,
    CommEngine cpuTsEngine, AlgResourceCtx &resource)
{
    // AICPU Engine Context 保存完整静态资源；CPU_TS Context 只保存 AICPU 主线程
    // 句柄，供后续调用重新导出并建立 Host <-> AICPU 启动/完成握手。
    std::vector<char> seq = resource.Serialize();
    param.ctxSize = seq.size();
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));

    void *hostCtx = nullptr;
    uint64_t hostCtxSize = sizeof(ThreadHandle);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
    CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &resource.aicpuThread, hostCtxSize, 0));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // OpParam 是本次调用的数据：用户 Buffer、count、root 等不会写入长期资源。
    OpParam param;
    int ret = snprintf(param.tag, TAG_LENGTH, "%s", kResourceTag);
    CHK_PRT_RET(ret < 0 || static_cast<uint32_t>(ret) >= TAG_LENGTH, HCCL_ERROR("failed to format resource tag"),
        HCCL_E_PARA);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(ValidateBroadcastParam(param));
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // 热路径：同一通信域已经创建过静态资源，本次只复用 Context 并重新导出
        // Host 可见的 AICPU 主线程句柄。root 和数据量可以与上次不同。
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtxSize != sizeof(ThreadHandle), HCCL_ERROR("invalid host ctx size=%lu", hostCtxSize),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // 冷路径：首次调用时一次性申请全连接 Channel 和所有 worker。
        AlgResourceCtx resCtxHost;
        resCtxHost.layoutVersion = static_cast<uint32_t>(ResourceLayoutVersion::VERSION_2);
        resCtxHost.rankSize = param.rankSize;
        resCtxHost.channelNotifyNum = kChannelNotifyNum;
        resCtxHost.pipelineDepth = kPipelineDepth;

        // 本地 HCCL Buffer 的真实大小由运行环境决定，AICPU 侧据此动态计算
        // Tile 容量，不能假设它一定等于 400MB。
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        ResourcePlan plan = BuildResourcePlan(param.rankSize);
        CHK_RET(AcquireThreads(comm, aicpuTsEngine, plan, resCtxHost));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine,
            &param.aicpuThreadOnCpu));
        CHK_RET(AcquireChannels(comm, param, aicpuTsEngine, plan, resCtxHost));
        CHK_RET(CreateAndStoreEngineContext(comm, param, aicpuTsEngine, cpuTsEngine, resCtxHost));
        HCCL_INFO("created broadcast resource, rank=%u rankSize=%u threads=%lu channels=%lu localBuffer=%lu",
            param.myRank, param.rankSize, resCtxHost.threads.size(), resCtxHost.channels.size(),
            resCtxHost.localBuffer.size);
    }

    // Kernel 内只负责把通信任务编排到已申请的线程/Channel 上；真正执行仍由
    // HCOMM/TS 按用户 stream 的依赖关系完成。
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
