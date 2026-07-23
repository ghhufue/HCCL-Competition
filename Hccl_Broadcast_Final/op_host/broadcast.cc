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
constexpr uint32_t MAIN_THREAD_NOTIFY_NUM =
    1 + BROADCAST_SEGMENT_PIPELINE_DEPTH * BROADCAST_CCU_DIE_NUM * 2;
constexpr uint32_t PUSH_THREAD_NOTIFY_NUM = 1 + BROADCAST_SEGMENT_PIPELINE_DEPTH;
constexpr char RESOURCE_TAG[] = "hccl_custom_broadcast_v18";

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

uint64_t SymmetricLinkKey(const CommLink &link)
{
    const EndpointDesc *first = &link.srcEndpointDesc;
    const EndpointDesc *second = &link.dstEndpointDesc;
    if (CompareEndpointAddress(*first, *second) > 0) {
        std::swap(first, second);
    }
    constexpr uint64_t FNV_OFFSET_BASIS = 1469598103934665603ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET_BASIS;
    for (uint32_t i = 0; i < COMM_ADDR_EID_LEN; ++i) {
        hash = (hash ^ first->commAddr.eid[i]) * FNV_PRIME;
    }
    for (uint32_t i = 0; i < COMM_ADDR_EID_LEN; ++i) {
        hash = (hash ^ second->commAddr.eid[i]) * FNV_PRIME;
    }
    return hash;
}

uint32_t ProtocolPriority(CommProtocol protocol)
{
    if (protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
        return 0;
    }
    if (protocol == CommProtocol::COMM_PROTOCOL_UBC_TP) {
        return 1;
    }
    return 2;
}

const char *ProtocolName(CommProtocol protocol)
{
    switch (protocol) {
        case CommProtocol::COMM_PROTOCOL_UBC_CTP:
            return "CTP";
        case CommProtocol::COMM_PROTOCOL_UBC_TP:
            return "TP";
        default:
            return "OTHER";
    }
}

const char *TopologyName(CommTopo topology)
{
    switch (topology) {
        case COMM_TOPO_CLOS:
            return "CLOS";
        case COMM_TOPO_1DMESH:
            return "MESH";
        case COMM_TOPO_CUSTOM:
            return "CUSTOM";
        default:
            return "OTHER";
    }
}

struct LinkCandidate {
    CommLink link{};
    uint32_t layer = 0;
    CommTopo topology = COMM_TOPO_RESERVED;
    uint32_t localDie = BROADCAST_CCU_DIE_NUM;
    EndpointAttrBwCoeff bandwidthCoeff = 0;
    bool bandwidthAvailable = false;
};

bool QueryEndpointDieOptional(
    HcclComm comm, uint32_t rankId, const EndpointDesc &endpoint, uint32_t &dieId)
{
    EndpointAttrDieId value = 0;
    const HcclResult ret = HcclRankGraphGetEndpointInfo(
        comm, rankId, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(value), &value);
    if (ret != HCCL_SUCCESS || value >= BROADCAST_CCU_DIE_NUM) {
        return false;
    }
    dieId = value;
    return true;
}

bool BetterLinkCandidate(const LinkCandidate &left, const LinkCandidate &right)
{
    const uint32_t leftProtocol = ProtocolPriority(left.link.linkAttr.linkProtocol);
    const uint32_t rightProtocol = ProtocolPriority(right.link.linkAttr.linkProtocol);
    if (leftProtocol != rightProtocol) {
        return leftProtocol < rightProtocol;
    }
    if (left.bandwidthAvailable != right.bandwidthAvailable) {
        return left.bandwidthAvailable;
    }
    if (left.bandwidthAvailable && left.bandwidthCoeff != right.bandwidthCoeff) {
        return left.bandwidthCoeff > right.bandwidthCoeff;
    }
    if (left.link.linkAttr.hop != right.link.linkAttr.hop) {
        return left.link.linkAttr.hop < right.link.linkAttr.hop;
    }
    if (left.layer != right.layer) {
        return left.layer < right.layer;
    }
    return SymmetricLinkLess(left.link, right.link);
}

bool SameLinkCapability(const LinkCandidate &left, const LinkCandidate &right)
{
    return ProtocolPriority(left.link.linkAttr.linkProtocol) ==
            ProtocolPriority(right.link.linkAttr.linkProtocol) &&
        left.bandwidthAvailable == right.bandwidthAvailable &&
        (!left.bandwidthAvailable || left.bandwidthCoeff == right.bandwidthCoeff) &&
        left.link.linkAttr.hop == right.link.linkAttr.hop &&
        left.layer == right.layer && left.topology == right.topology;
}

uint64_t SymmetricRankPairKey(uint32_t rankA, uint32_t rankB)
{
    const uint64_t low = std::min(rankA, rankB);
    const uint64_t high = std::max(rankA, rankB);
    uint64_t value = (low << 32U) | high;
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

const char *SelectionReason(
    const LinkCandidate &selected, const LinkCandidate *runnerUp, uint32_t balancedCandidateCount)
{
    if (balancedCandidateCount > 1) {
        return "rank_pair_static_balance";
    }
    if (runnerUp == nullptr) {
        return "only_candidate";
    }
    if (ProtocolPriority(selected.link.linkAttr.linkProtocol) !=
        ProtocolPriority(runnerUp->link.linkAttr.linkProtocol)) {
        return "protocol";
    }
    if (selected.bandwidthAvailable != runnerUp->bandwidthAvailable ||
        (selected.bandwidthAvailable && selected.bandwidthCoeff != runnerUp->bandwidthCoeff)) {
        return "bw_coeff";
    }
    if (selected.link.linkAttr.hop != runnerUp->link.linkAttr.hop) {
        return "hop";
    }
    if (selected.layer != runnerUp->layer) {
        return "network_layer";
    }
    return "canonical_endpoint";
}

HcclResult QueryBestCcuLinkToPeer(
    HcclComm comm, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &desc, uint32_t &localDieId)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[QueryBestCcuLinkToPeer] no network layer for rank %u", remoteRank), HCCL_E_NOT_FOUND);

    std::vector<LinkCandidate> candidates;
    uint32_t candidateDieMask = 0;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], myRank, remoteRank, &links, &linkNum));
        CommTopo topology = COMM_TOPO_RESERVED;
        (void)HcclRankGraphGetTopoTypeByLayer(comm, netLayers[layerIdx], &topology);
        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommProtocol protocol = links[linkIdx].linkAttr.linkProtocol;
            if (protocol != CommProtocol::COMM_PROTOCOL_UBC_CTP &&
                protocol != CommProtocol::COMM_PROTOCOL_UBC_TP) {
                continue;
            }
            LinkCandidate candidate;
            candidate.link = links[linkIdx];
            candidate.layer = netLayers[layerIdx];
            candidate.topology = topology;
            if (!QueryEndpointDieOptional(
                    comm, myRank, candidate.link.srcEndpointDesc, candidate.localDie)) {
                continue;
            }
            // GetEndpointInfo is rank-local in the Finals RankGraph runtime.
            // Querying dstEndpointDesc with remoteRank emits a hard error even
            // when used as an optional probe. Keep remote attributes unknown
            // and fall back to properties both endpoints observe identically.
            candidateDieMask |= 1U << candidate.localDie;
            HCCL_DEBUG("[BroadcastChannelCandidate] rank=%u peer=%u protocol=%s layer=%u topology=%s "
                       "localDie=%u bwCoeff=%u bwAvailable=%u hop=%u channelKey=%016llx",
                myRank, remoteRank, ProtocolName(protocol), candidate.layer,
                TopologyName(candidate.topology), candidate.localDie, candidate.bandwidthCoeff,
                candidate.bandwidthAvailable ? 1U : 0U, candidate.link.linkAttr.hop,
                static_cast<unsigned long long>(SymmetricLinkKey(candidate.link)));
            candidates.push_back(candidate);
        }
    }

    if (candidates.empty()) {
        return HCCL_E_NOT_FOUND;
    }
    std::sort(candidates.begin(), candidates.end(), BetterLinkCandidate);
    uint32_t balancedCandidateCount = 1;
    while (balancedCandidateCount < candidates.size() &&
        SameLinkCapability(candidates.front(), candidates[balancedCandidateCount])) {
        ++balancedCandidateCount;
    }
    const uint32_t selectedIndex = static_cast<uint32_t>(
        SymmetricRankPairKey(myRank, remoteRank) % balancedCandidateCount);
    const LinkCandidate &selectedCandidate = candidates[selectedIndex];
    const CommLink &selected = selectedCandidate.link;
    localDieId = selectedCandidate.localDie;
    CHK_PRT_RET(localDieId >= BROADCAST_CCU_DIE_NUM,
        HCCL_ERROR("[QueryBestCcuLinkToPeer] invalid local die=%u", localDieId), HCCL_E_INTERNAL);
    if (LOG_LEVEL <= LOG_LEVEL_INFO) {
        const LinkCandidate *runnerUp = candidates.size() > balancedCandidateCount ?
            &candidates[balancedCandidateCount] : nullptr;
        const char *pathClass = selected.linkAttr.hop <= 1 ? "DIRECT" : "TRANSIT_ONLY";
        HCCL_INFO("[BroadcastChannelDiag] rank=%u peer=%u protocol=%s layer=%u topology=%s path=%s "
            "localDie=%u bwCoeff=%u bwAvailable=%u candidateCount=%zu balancedCandidateCount=%u "
            "selectedIndex=%u candidateDieMask=0x%x tieBreak=%s channelKey=%016llx",
            myRank, remoteRank, ProtocolName(selected.linkAttr.linkProtocol), selectedCandidate.layer,
            TopologyName(selectedCandidate.topology), pathClass, localDieId,
            selectedCandidate.bandwidthCoeff, selectedCandidate.bandwidthAvailable ? 1U : 0U,
            candidates.size(), balancedCandidateCount, selectedIndex, candidateDieMask,
            SelectionReason(selectedCandidate, runnerUp, balancedCandidateCount),
            static_cast<unsigned long long>(SymmetricLinkKey(selected)));
    }
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
    HCCL_INFO("[AcquireAllPeerChannels] rank=%u die0Peers=%zu die1Peers=%zu",
        param.myRank, channelsByDie[0].size(), channelsByDie[1].size());
    return HCCL_SUCCESS;
}

std::shared_ptr<BroadcastKernelArg> BuildKernelArg(const OpParam &param,
    const std::vector<ChannelHandle> &channels, const std::vector<uint32_t> &remoteRanks,
    uint32_t dieId, uint32_t seedDie, uint32_t activeDieMask,
    const ops_hccl::OwnerWriteConfig &config, uint32_t pushGroup = 0,
    bool dataOnlyGroup = false)
{
    auto arg = std::make_shared<BroadcastKernelArg>();
    arg->rankSize = param.rankSize;
    arg->rankId = param.myRank;
    arg->rootRank = param.root;
    arg->dieId = dieId;
    arg->seedDie = seedDie;
    arg->activeDieMask = activeDieMask;
    arg->pushGroup = pushGroup;
    arg->pushWindowDepth = config.pushWindowDepth;
    arg->enablePushBatchMerge = config.enablePushBatchMerge ? 1U : 0U;
    uint32_t pushPeerIndex = 0;
    for (uint32_t i = 0; i < channels.size(); ++i) {
        const bool isPushPeer = param.myRank == param.root || remoteRanks[i] != param.root;
        const uint32_t peerGroup = isPushPeer ? pushPeerIndex++ % 2U : 0U;
        if (dataOnlyGroup && ((isPushPeer && peerGroup != pushGroup) ||
                (!isPushPeer && pushGroup != 0))) {
            continue;
        }
        const uint32_t argIndex = arg->channelCount++;
        arg->channels[argIndex] = channels[i];
        arg->remoteRanks[argIndex] = remoteRanks[i];
        arg->peerGroups[argIndex] = peerGroup;
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
    const std::vector<uint32_t> (&remoteRanksByDie)[BROADCAST_CCU_DIE_NUM],
    const ops_hccl::OwnerWriteConfig &config, KernelKind algorithm, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterBroadcastKernels] expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    HcclResult ret = HCCL_SUCCESS;
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if (!channelsByDie[dieId].empty()) {
            resCtx.activeDieMask |= 1U << dieId;
        }
        uint32_t pushPeerCount = 0;
        for (uint32_t peerRank : remoteRanksByDie[dieId]) {
            pushPeerCount += param.myRank == param.root || peerRank != param.root ? 1U : 0U;
        }
        if (algorithm == KernelKind::CONTIGUOUS_OWNER_WRITE && pushPeerCount > 1) {
            resCtx.group1DieMask |= 1U << dieId;
        }
    }
    uint32_t seedDie = BROADCAST_CCU_DIE_NUM;
    if (algorithm == KernelKind::CONTIGUOUS_OWNER_WRITE) {
        if (param.myRank != param.root) {
            seedDie = resCtx.peerDieByRank[param.root];
        } else {
            for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
                if ((resCtx.activeDieMask & (1U << dieId)) != 0) {
                    seedDie = dieId;
                    break;
                }
            }
        }
        if (seedDie >= BROADCAST_CCU_DIE_NUM) {
            HCCL_ERROR("[RegisterBroadcastKernels] invalid seed die=%u", seedDie);
            ret = HCCL_E_INTERNAL;
        }
        HCCL_INFO("[BroadcastChannelSummary] rank=%u root=%u die0Peers=%zu die1Peers=%zu seedDie=%u",
            param.myRank, param.root, channelsByDie[0].size(), channelsByDie[1].size(), seedDie);
    }
    const bool useSegmentPipeline = algorithm == KernelKind::CONTIGUOUS_OWNER_WRITE &&
        param.myRank != param.root;
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM && ret == HCCL_SUCCESS; ++dieId) {
        if (channelsByDie[dieId].empty()) {
            continue;
        }
        if (algorithm == KernelKind::SMALL_RECEIVER_PULL) {
            auto smallPullArg = BuildKernelArg(
                param, channelsByDie[dieId], remoteRanksByDie[dieId], dieId, dieId,
                resCtx.activeDieMask, config);
            ret = RegisterOneKernel(insHandle, dieId, "CcuBroadcastSmallReceiverPullKernel",
                reinterpret_cast<void *>(ops_hccl::CcuBroadcastSmallReceiverPullKernel), smallPullArg,
                resCtx.smallPullKernels[dieId]);
        } else {
            auto ownerWriteArg = BuildKernelArg(
                param, channelsByDie[dieId], remoteRanksByDie[dieId], dieId, seedDie,
                resCtx.activeDieMask, config, 0, true);
            if ((resCtx.group1DieMask & (1U << dieId)) != 0) {
                ownerWriteArg->pushWindowDepth = 1;
                HCCL_INFO("[BroadcastPushGroups] rank=%u die=%u group0Peers=%u group1Enabled=1 "
                          "effectiveDepthPerGroup=1 requestedDepth=%u",
                    param.myRank, dieId, ownerWriteArg->channelCount, config.pushWindowDepth);
            }
            const bool rootOwner = param.myRank == param.root;
            const char *kernelName = rootOwner ? "CcuBroadcastRootOwnerWriteKernel" :
                (useSegmentPipeline ? "CcuBroadcastOwnerControlKernel" :
                                      "CcuBroadcastContiguousOwnerWriteKernel");
            void *kernelFunc = rootOwner ?
                reinterpret_cast<void *>(ops_hccl::CcuBroadcastRootOwnerWriteKernel) :
                (useSegmentPipeline ? reinterpret_cast<void *>(ops_hccl::CcuBroadcastOwnerControlKernel) :
                                      reinterpret_cast<void *>(ops_hccl::CcuBroadcastContiguousOwnerWriteKernel));
            ret = RegisterOneKernel(insHandle, dieId, kernelName, kernelFunc, ownerWriteArg,
                resCtx.ownerWriteKernels[dieId]);
            if (ret == HCCL_SUCCESS && (resCtx.group1DieMask & (1U << dieId)) != 0) {
                auto group1Arg = BuildKernelArg(
                    param, channelsByDie[dieId], remoteRanksByDie[dieId], dieId, seedDie,
                    resCtx.activeDieMask, config, 1, true);
                group1Arg->pushWindowDepth = 1;
                const char *group1KernelName = rootOwner ? "CcuBroadcastRootOwnerWriteGroup1Kernel" :
                                                          "CcuBroadcastOwnerControlGroup1Kernel";
                void *group1KernelFunc = rootOwner ?
                    reinterpret_cast<void *>(ops_hccl::CcuBroadcastRootOwnerWriteKernel) :
                    reinterpret_cast<void *>(ops_hccl::CcuBroadcastOwnerControlKernel);
                ret = RegisterOneKernel(insHandle, dieId, group1KernelName, group1KernelFunc,
                    group1Arg, resCtx.segmentPushKernels[dieId]);
            }
            if (ret == HCCL_SUCCESS && !rootOwner && !useSegmentPipeline && dieId == seedDie) {
                auto producerArg = BuildKernelArg(
                    param, channelsByDie[dieId], remoteRanksByDie[dieId], dieId, seedDie,
                    resCtx.activeDieMask, config);
                ret = RegisterOneKernel(insHandle, dieId, "CcuBroadcastOwnerSeedKernel",
                    reinterpret_cast<void *>(ops_hccl::CcuBroadcastOwnerSeedKernel), producerArg,
                    resCtx.seedKernels[dieId]);
            }
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

HcclResult AcquirePushThreads(HcclComm comm, AlgResourceCtx &resCtx)
{
    uint32_t activeCount = 0;
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        activeCount += (resCtx.activeDieMask & (1U << dieId)) != 0 ? 1U : 0U;
    }
    const uint32_t group1Count = static_cast<uint32_t>(
        (resCtx.group1DieMask & 1U) != 0) + static_cast<uint32_t>((resCtx.group1DieMask & 2U) != 0);
    const uint32_t threadCount = activeCount + group1Count;
    CHK_PRT_RET(activeCount == 0 || activeCount > BROADCAST_CCU_DIE_NUM ||
            (resCtx.group1DieMask & ~resCtx.activeDieMask) != 0 ||
            threadCount > BROADCAST_CCU_DIE_NUM * 2,
        HCCL_ERROR("[AcquirePushThreads] invalid active die count=%u", activeCount), HCCL_E_INTERNAL);
    ThreadHandle acquired[BROADCAST_CCU_DIE_NUM * 2]{};
    CHK_RET(HcclThreadAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, threadCount, PUSH_THREAD_NOTIFY_NUM, acquired));
    uint32_t threadIdx = 0;
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((resCtx.activeDieMask & (1U << dieId)) != 0) {
            resCtx.pushThreads[dieId] = acquired[threadIdx++];
        }
    }
    for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
        if ((resCtx.group1DieMask & (1U << dieId)) != 0) {
            resCtx.pushGroup1Threads[dieId] = acquired[threadIdx++];
        }
    }
    resCtx.pushThreadCount = threadCount;
    HCCL_DEBUG("[AcquirePushThreads] activeDieMask=%u group1DieMask=%u pushThreadCount=%u",
        resCtx.activeDieMask, resCtx.group1DieMask, resCtx.pushThreadCount);
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

    ops_hccl::OwnerWriteConfig ownerWriteConfig;
    CHK_RET(ops_hccl::LoadOwnerWriteConfig(ownerWriteConfig));
    OpParam param;
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

    const uint64_t totalBytes = count * SIZE_TABLE.at(dataType);
    ops_hccl::ExecutionPlan executionPlan;
    CHK_RET(ops_hccl::BuildExecutionPlan(totalBytes, param.rankSize, param.myRank, executionPlan));
    const KernelKind algorithm = executionPlan.algorithm;
    const int tagRet = std::snprintf(param.tag, sizeof(param.tag),
        "%s_r%u_a%u_d%u_m%u_t%llu_b%llu", RESOURCE_TAG, root,
        static_cast<uint32_t>(algorithm), ownerWriteConfig.pushWindowDepth,
        ownerWriteConfig.enablePushBatchMerge ? 1U : 0U,
        static_cast<unsigned long long>(ownerWriteConfig.tileSizeBytes),
        static_cast<unsigned long long>(ownerWriteConfig.maxPushBatchBytes));
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[HcclBroadcast] failed to create operation tag"), HCCL_E_INTERNAL);

    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, MAIN_THREAD_NOTIFY_NUM, &param.cpuThread));

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
        resCtx.rootRank = param.root;
        resCtx.pushWindowDepth = ownerWriteConfig.pushWindowDepth;
        resCtx.enablePushBatchMerge = ownerWriteConfig.enablePushBatchMerge ? 1U : 0U;
        CHK_RET(HcclGetHcclBuffer(comm, &resCtx.localBuffer.addr, &resCtx.localBuffer.size));

        std::vector<ChannelHandle> channelsByDie[BROADCAST_CCU_DIE_NUM];
        std::vector<uint32_t> remoteRanksByDie[BROADCAST_CCU_DIE_NUM];
        CHK_RET(AcquireAllPeerChannels(
            comm, param, channelsByDie, remoteRanksByDie, resCtx.peerDieByRank));
        CHK_RET(RegisterBroadcastKernels(
            comm, param, channelsByDie, remoteRanksByDie, ownerWriteConfig, algorithm, resCtx));
        CHK_RET(AcquirePushThreads(comm, resCtx));
        CHK_RET(CreateAndStoreEngineContext(comm, param, resCtx));

        CHK_RET(HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize));
        CHK_PTR_NULL(ctx);
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    }

    return ops_hccl::ExecOp(param, stream);
}
