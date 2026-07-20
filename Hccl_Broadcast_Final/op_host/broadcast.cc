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
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <ccu/ccu_launch.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;

HcclResult ValidateBroadcastParam(const OpParam &param)
{
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[ValidateBroadcastParam] only FP32 is supported, dataType=%d", param.dataType), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[ValidateBroadcastParam] invalid rankSize=%u", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("[ValidateBroadcastParam] invalid rank=%u for rankSize=%u", param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[ValidateBroadcastParam] invalid root=%u for rankSize=%u", param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeIt->second,
        HCCL_ERROR("[ValidateBroadcastParam] byte size overflows uint64_t"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult GetEndpointDieId(
    HcclComm comm, uint32_t rankId, const EndpointDesc &endpoint, uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId, &endpoint,
        ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
    dieId = endpointDieId;
    return dieId < BROADCAST_CCU_DIE_NUM ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}

int CompareEndpointAddress(const EndpointDesc &left, const EndpointDesc &right)
{
    return std::memcmp(left.commAddr.eid, right.commAddr.eid, COMM_ADDR_EID_LEN);
}

bool SymmetricLinkLess(const CommLink &left, const CommLink &right)
{
    const EndpointDesc *leftFirst = &left.srcEndpointDesc;
    const EndpointDesc *leftSecond = &left.dstEndpointDesc;
    if (CompareEndpointAddress(*leftFirst, *leftSecond) > 0) {
        std::swap(leftFirst, leftSecond);
    }
    const EndpointDesc *rightFirst = &right.srcEndpointDesc;
    const EndpointDesc *rightSecond = &right.dstEndpointDesc;
    if (CompareEndpointAddress(*rightFirst, *rightSecond) > 0) {
        std::swap(rightFirst, rightSecond);
    }
    const int firstCompare = CompareEndpointAddress(*leftFirst, *rightFirst);
    return firstCompare < 0 ||
        (firstCompare == 0 && CompareEndpointAddress(*leftSecond, *rightSecond) < 0);
}

HcclResult QueryBestCcuLinkToPeer(
    HcclComm comm, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &desc, uint32_t &localDieId)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[QueryBestCcuLinkToPeer] no network layer for rank %u", remoteRank), HCCL_E_NOT_FOUND);

    const CommProtocol preferredProtocols[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };
    bool found = false;
    CommLink selected{};
    for (const auto protocol : preferredProtocols) {
        if (protocol == CommProtocol::COMM_PROTOCOL_RESERVED) {
            continue;
        }
        for (uint32_t layerIdx = 0; layerIdx < netLayerNum && !found; ++layerIdx) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], myRank, remoteRank, &links, &linkNum));
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                if (links[linkIdx].linkAttr.linkProtocol != protocol) {
                    continue;
                }
                uint32_t candidateDieId = 0;
                if (GetEndpointDieId(
                        comm, myRank, links[linkIdx].srcEndpointDesc, candidateDieId) != HCCL_SUCCESS) {
                    continue;
                }
                if (!found || SymmetricLinkLess(links[linkIdx], selected)) {
                    selected = links[linkIdx];
                }
                found = true;
            }
        }
        if (found) {
            break;
        }
    }

    if (!found) {
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(GetEndpointDieId(comm, myRank, selected.srcEndpointDesc, localDieId));
    CHK_PRT_RET(localDieId >= BROADCAST_CCU_DIE_NUM,
        HCCL_ERROR("[QueryBestCcuLinkToPeer] invalid local die=%u", localDieId), HCCL_E_INTERNAL);
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selected.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selected.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selected.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selected.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selected.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selected.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireAllPeerChannels(
    HcclComm comm, const OpParam &param,
    std::vector<ChannelHandle> (&channelsByDie)[BROADCAST_CCU_DIE_NUM],
    std::vector<uint32_t> (&remoteRanksByDie)[BROADCAST_CCU_DIE_NUM], uint32_t *peerDieByRank)
{
    CHK_PTR_NULL(peerDieByRank);
    const uint32_t channelCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> localDies;
    std::vector<uint32_t> remoteRanks;
    descs.reserve(channelCount);
    localDies.reserve(channelCount);
    remoteRanks.reserve(channelCount);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        uint32_t localDieId = 0;
        CHK_RET(QueryBestCcuLinkToPeer(comm, param.myRank, remoteRank, desc, localDieId));
        descs.push_back(desc);
        localDies.push_back(localDieId);
        remoteRanks.push_back(remoteRank);
    }

    CHK_PRT_RET(descs.size() != channelCount || localDies.size() != channelCount ||
            remoteRanks.size() != channelCount,
        HCCL_ERROR("[AcquireAllPeerChannels] channel map is incomplete"), HCCL_E_INTERNAL);

    std::vector<ChannelHandle> allChannels(channelCount);
    if (channelCount != 0) {
        CHK_RET(HcclChannelAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, descs.data(), channelCount, allChannels.data()));
    }
    for (uint32_t i = 0; i < channelCount; ++i) {
        channelsByDie[localDies[i]].push_back(allChannels[i]);
        remoteRanksByDie[localDies[i]].push_back(remoteRanks[i]);
        peerDieByRank[remoteRanks[i]] = localDies[i];
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<BroadcastKernelArg> BuildKernelArg(const OpParam &param,
    const std::vector<ChannelHandle> &channels, const std::vector<uint32_t> &remoteRanks)
{
    auto arg = std::make_shared<BroadcastKernelArg>();
    arg->rankSize = param.rankSize;
    arg->rankId = param.myRank;
    arg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        arg->channels[i] = channels[i];
        arg->remoteRanks[i] = remoteRanks[i];
    }
    return arg;
}

HcclResult RegisterOneKernel(CcuInsHandle insHandle, uint32_t dieId, const char *name, void *kernelFunc,
    const std::shared_ptr<BroadcastKernelArg> &kernelArg, CcuKernelHandle &kernelHandle)
{
    const void *kernelArgs[] = {kernelArg.get()};
    CcuResult ret = HcommCcuKernelRegister(
        insHandle, dieId, name, kernelFunc, kernelArgs, 1, &kernelHandle);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterOneKernel] failed to register %s, ccuRet=%d", name, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterBroadcastKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelHandle> (&channelsByDie)[BROADCAST_CCU_DIE_NUM],
    const std::vector<uint32_t> (&remoteRanksByDie)[BROADCAST_CCU_DIE_NUM], AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterBroadcastKernels] expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    HcclResult ret = HCCL_SUCCESS;
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM && ret == HCCL_SUCCESS; ++dieId) {
        if (channelsByDie[dieId].empty()) {
            continue;
        }
        resCtx.activeDieMask |= 1U << dieId;
        auto directArg = BuildKernelArg(param, channelsByDie[dieId], remoteRanksByDie[dieId]);
        auto pullArg = BuildKernelArg(param, channelsByDie[dieId], remoteRanksByDie[dieId]);
        ret = RegisterOneKernel(insHandle, dieId, "CcuBroadcastDirectKernel",
            reinterpret_cast<void *>(ops_hccl::CcuBroadcastDirectKernel), directArg,
            resCtx.directKernels[dieId]);
        if (ret == HCCL_SUCCESS) {
            ret = RegisterOneKernel(insHandle, dieId, "CcuBroadcastPullScatterAllGatherKernel",
                reinterpret_cast<void *>(ops_hccl::CcuBroadcastPullScatterAllGatherKernel), pullArg,
                resCtx.pullKernels[dieId]);
        }
    }

    if (ret == HCCL_SUCCESS && resCtx.activeDieMask == 0) {
        ret = HCCL_E_INTERNAL;
    }

    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (endRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterBroadcastKernels] register end failed, ccuRet=%d", endRet);
        return ConvertCcuToHccl(endRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateAndStoreEngineContext(HcclComm comm, const OpParam &param, const AlgResourceCtx &resCtx)
{
    std::vector<char> seq = resCtx.Serialize();
    CHK_PRT_RET(seq.size() != AlgResourceCtx::SerializedSize(),
        HCCL_ERROR("[CreateAndStoreEngineContext] unexpected serialized size=%zu", seq.size()), HCCL_E_INTERNAL);
    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, static_cast<uint64_t>(seq.size()), &ctx));
    CHK_PTR_NULL(ctx);
    CHK_RET(HcclEngineCtxCopy(
        comm, CommEngine::COMM_ENGINE_CCU, param.tag, seq.data(), static_cast<uint64_t>(seq.size()), 0));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    const int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_broadcast");
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[HcclBroadcast] failed to create operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(ValidateBroadcastParam(param));

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    const HcclResult ctxRet = HcclEngineCtxGet(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize);
    if (ctxRet == HCCL_SUCCESS) {
        CHK_PTR_NULL(ctx);
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        CHK_PRT_RET(ctxRet != HCCL_E_NOT_FOUND,
            HCCL_ERROR("[HcclBroadcast] failed to query engine context, ret=%d", ctxRet), ctxRet);
        AlgResourceCtx resCtx;
        resCtx.rankSize = param.rankSize;
        CHK_RET(HcclGetHcclBuffer(comm, &resCtx.localBuffer.addr, &resCtx.localBuffer.size));

        std::vector<ChannelHandle> channelsByDie[BROADCAST_CCU_DIE_NUM];
        std::vector<uint32_t> remoteRanksByDie[BROADCAST_CCU_DIE_NUM];
        CHK_RET(AcquireAllPeerChannels(
            comm, param, channelsByDie, remoteRanksByDie, resCtx.peerDieByRank));
        CHK_RET(RegisterBroadcastKernels(comm, param, channelsByDie, remoteRanksByDie, resCtx));
        CHK_RET(CreateAndStoreEngineContext(comm, param, resCtx));

        CHK_RET(HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize));
        CHK_PTR_NULL(ctx);
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    }

    return ops_hccl::ExecOp(param, stream);
}
