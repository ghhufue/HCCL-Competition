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

#include <cstdint>
#include <memory>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t RESOURCE_LAYOUT_VERSION = 5;
constexpr uint32_t BROADCAST_CCU_DIE_NUM = 2;
constexpr uint32_t DIRECT_PHASE_COUNT = 3;
constexpr uint32_t PULL_PHASE_COUNT = 7;

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
    uint32_t remoteRanks[MAX_RANK_SIZE];
};

enum class KernelKind : uint32_t {
    DIRECT = 0,
    PULL_SCATTER_ALLGATHER = 1,
};

enum class PullPhase : uint64_t {
    PRESYNC_PUBLISH = 0,
    PRESYNC_WAIT = 1,
    SEED = 2,
    PHASE2_START = 3,
    ALLGATHER = 4,
    READ_DONE = 5,
    GLOBAL_DONE = 6,
};

enum class DirectPhase : uint64_t {
    PRESYNC_PUBLISH = 0,
    PRESYNC_WAIT = 1,
    DATA = 2,
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
    uint32_t activeDieMask = 0;
    uint32_t peerDieByRank[MAX_RANK_SIZE]{};
    CommBuffer localBuffer{nullptr, 0};
    CcuKernelHandle directKernels[BROADCAST_CCU_DIE_NUM]{};
    CcuKernelHandle pullKernels[BROADCAST_CCU_DIE_NUM]{};

    static constexpr uint64_t SerializedSize()
    {
        return sizeof(version) + sizeof(rankSize) + sizeof(activeDieMask) + sizeof(peerDieByRank) +
            sizeof(localBuffer) +
            sizeof(directKernels) + sizeof(pullKernels);
    }

    std::vector<char> Serialize() const
    {
        BinaryStream binaryStream;
        binaryStream << version;
        binaryStream << rankSize;
        binaryStream << activeDieMask;
        for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
            binaryStream << peerDieByRank[rank];
        }
        binaryStream << localBuffer;
        for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
            binaryStream << directKernels[dieId];
            binaryStream << pullKernels[dieId];
        }
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> version;
        binaryStream >> rankSize;
        binaryStream >> activeDieMask;
        for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
            binaryStream >> peerDieByRank[rank];
        }
        binaryStream >> localBuffer;
        for (uint32_t dieId = 0; dieId < BROADCAST_CCU_DIE_NUM; ++dieId) {
            binaryStream >> directKernels[dieId];
            binaryStream >> pullKernels[dieId];
        }
    }
};

#endif // OPS_HCCL_CUSTOM_H
