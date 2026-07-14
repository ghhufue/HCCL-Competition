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
    // HCCL 通信 Buffer 的基地址和可用字节数。Host 侧查询，AICPU 侧只消费。
    void *addr;
    uint64_t size;
} CommBuffer;

// Engine Context 的序列化布局版本。Host 和 AICPU 必须使用完全相同的版本。
enum class ResourceLayoutVersion : uint32_t {
    VERSION_1 = 1,
    VERSION_2 = 2,
};

// 与一个远端 rank 对应的静态 Channel 资源。
//
// 这里不保存 root、数据长度、Tile 等单次 Broadcast 参数。这样同一个通信域
// 可以在 root 和数据量变化时继续复用同一份 Engine Context。
struct ChannelInfo {
    // 此 Channel 连接的对端 rank。channels 始终按 remoteRank 升序保存。
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    // 执行此 Channel 任务的 worker 在线程数组中的下标；0 固定为主线程。
    uint32_t workerIndex = 0;
    // Channel 创建时申请的本地/远端 Notify 数量。
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    // 对端 HCCL Buffer。单边 Write 的目的地址由这个基址加槽位偏移得到。
    CommBuffer remoteCclMem = {nullptr, 0};
};

// Broadcast 在一个通信域内长期复用的静态资源。
//
// 生命周期：Host 首次调用时申请 -> 序列化到 AICPU Engine Context -> 每次
// Kernel 启动时反序列化。这里不能放用户 buf、count、root 等调用级数据。
struct AlgResourceCtx {
    uint32_t layoutVersion = static_cast<uint32_t>(ResourceLayoutVersion::VERSION_2);
    uint32_t rankSize = 0;
    // workerCount 不包含 threads[0] 主线程；当前 16 rank 时为 15。
    uint32_t workerCount = 0;
    // 每个 ThreadHandle 拥有的 Notify 槽数，包含主从启动/完成通知。
    uint32_t notifyNumPerThread = 0;
    // 每条 Channel 的 Notify 槽数。双缓冲时需要 2 个 DATA_READY + 2 个 CONSUMED。
    uint32_t channelNotifyNum = 0;
    // Host 申请资源时允许的最大流水窗口数；实际值还会在执行计划中校验。
    uint32_t pipelineDepth = 0;
    // threads[0]，同时承担 AICPU Kernel 与 Host 的启动/完成握手。
    ThreadHandle aicpuThread = 0;
    // 本 rank 的 HCCL Buffer，Distributed 模式会切分成 window x owner 槽位。
    CommBuffer localBuffer = {nullptr, 0};
    // threads[0] 是主线程，threads[1..workerCount] 是各 Channel worker。
    std::vector<ThreadHandle> threads;
    // 与所有其他 rank 的全连接 Channel；大包 owner fanout 需要非 root 间互传。
    std::vector<ChannelInfo> channels;

    // 注意：Serialize 与 DeSerialize 的字段顺序是协议的一部分，修改一侧时必须
    // 同步修改另一侧并升级 layoutVersion。
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
