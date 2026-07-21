/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <cstdint>

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint32_t MAX_PUSH_WINDOW_DEPTH = 4;
static_assert(MAX_RANK_SIZE <= 16, "CCU completion masks are 16-bit rank masks");

constexpr uint32_t BUFFER_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CKE_PRESYNC = 0;
constexpr uint32_t CKE_PHASE = 1;
constexpr uint32_t MASK_BUFFER_READY = 1U << BUFFER_XN_ID;
constexpr uint32_t MASK_TOKEN_READY = 1U << TOKEN_XN_ID;

enum BroadcastNotifyMask : uint32_t {
    NOTIFY_SMALL_READ_DONE = 1U << 0,
    NOTIFY_SMALL_GLOBAL_DONE = 1U << 1,
    NOTIFY_OWNER_DONE = 1U << 2,
    NOTIFY_OWNER_GLOBAL_DONE = 1U << 3,
};

struct BroadcastContext {
    const BroadcastKernelArg *arg;
    ccu::Variable buffer[MAX_RANK_SIZE];
    ccu::Variable token[MAX_RANK_SIZE];
    ccu::Variable root;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable ownerOffset;
    ccu::Variable ownerBytes;
    ccu::Variable tileSizeBytes;
    ccu::Variable seedFullTileCount;
    ccu::Variable seedFullBytes;
    ccu::Variable seedTailBytes;
    ccu::Variable enablePushBatchMerge;
    ccu::Variable maxPushBatchBytes;
    ccu::Variable pushMergeFactor;
    ccu::Variable pushWindowDepth;
    ccu::Variable pushFirstBytes;
    ccu::Variable pushLoopOffset;
    ccu::Variable pushLoopBytes;
    ccu::Variable pushLoopParam;
    ccu::Variable pushLoopRemainder;
    ccu::Variable pushTailOffset;
    ccu::Variable pushTailBytes;
    ccu::Variable pushTailReadyTiles;
    ccu::Variable kernelPhase;
    ccu::Event pushEvents[MAX_PUSH_WINDOW_DEPTH];
};

CcuResult InitBroadcastResource(BroadcastContext &ctx, const BroadcastKernelArg *arg);
CcuResult LoadSmallBroadcastArgs(BroadcastContext &ctx);
CcuResult LoadOwnerWriteArgs(BroadcastContext &ctx);
CcuResult PublishBufferInfo(BroadcastContext &ctx);
CcuResult WaitBufferInfo(BroadcastContext &ctx);
CcuResult PreSyncBufferInfo(BroadcastContext &ctx);
CcuResult CcuBroadcastSmallReceiverPullKernel(CcuKernelArg arg);
CcuResult CcuBroadcastOwnerSeedKernel(CcuKernelArg arg);
CcuResult CcuBroadcastContiguousOwnerWriteKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
