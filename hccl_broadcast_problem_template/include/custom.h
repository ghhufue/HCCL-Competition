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

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

enum class ResourceLayoutVersion : uint32_t {
    VERSION_1 = 1,
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t workerIndex = 0;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem = {nullptr, 0};
};

struct AlgResourceCtx {
    uint32_t layoutVersion = static_cast<uint32_t>(ResourceLayoutVersion::VERSION_1);
    uint32_t rankSize = 0;
    uint32_t workerCount = 0;
    uint32_t notifyNumPerThread = 0;
    uint32_t channelNotifyNum = 0;
    uint32_t pipelineDepth = 0;
    ThreadHandle aicpuThread = 0;
    CommBuffer localBuffer = {nullptr, 0};
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << layoutVersion;
        binaryStream << rankSize;
        binaryStream << workerCount;
        binaryStream << notifyNumPerThread;
        binaryStream << channelNotifyNum;
        binaryStream << pipelineDepth;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;

        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> layoutVersion;
        binaryStream >> rankSize;
        binaryStream >> workerCount;
        binaryStream >> notifyNumPerThread;
        binaryStream >> channelNotifyNum;
        binaryStream >> pipelineDepth;
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
