/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include <cstdint>
#include <vector>

#include <hccl/hcomm_primitives.h>

#include "common.h"
#include "custom.h"

namespace ops_hccl {

struct OwnerWriteConfig {
    uint64_t tileSizeBytes = 4ULL * 1024ULL * 1024ULL;
    bool enablePushBatchMerge = false;
    uint64_t maxPushBatchBytes = 8ULL * 1024ULL * 1024ULL;
    uint32_t pushWindowDepth = 2;
};

struct ChunkDesc {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    OwnerBlock owner;
    uint64_t tileSizeBytes = 0;
    uint64_t seedFullTileCount = 0;
    uint64_t seedTailBytes = 0;
    bool enablePushBatchMerge = false;
    uint64_t maxPushBatchBytes = 0;
    uint32_t pushWindowDepth = 2;
    PushBatchPlan push;
};

struct ExecutionPlan {
    KernelKind algorithm = KernelKind::CONTIGUOUS_OWNER_WRITE;
    std::vector<ChunkDesc> chunks;
};

HcclResult LoadOwnerWriteConfig(OwnerWriteConfig &config);
HcclResult BuildExecutionPlan(uint64_t totalBytes, uint32_t rankSize, uint32_t rankId, ExecutionPlan &plan);
HcclResult LaunchSmallReceiverPullChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk);
HcclResult LaunchContiguousOwnerWriteChunk(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, const ChunkDesc &chunk);
HcclResult ExecOp(const OpParam &param, aclrtStream stream);

} // namespace ops_hccl
#endif // OPS_HCCL_CCU_EXEC_OP_H
