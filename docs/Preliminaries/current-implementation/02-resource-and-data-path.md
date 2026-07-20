# 资源与数据通路

## 资源总表

| 对象 | 代码对象 | 定义/创建位置 | 数量和大小 | 所有者 | 生命周期 | 作用 |
|---|---|---|---|---|---|---|
| rank | `OpParam::myRank/rankSize/root` | 文件：`include/common.h`；符号：`OpParam`；由 `HcclGetRankId/Size` 填充 | 每次调用各 1 个标量 | Host 参数；AICPU 只读副本 | 单次调用 | 决定 root/owner/peer 分支 |
| link | `CommLink selectedLink` | 文件：`op_host/broadcast.cc`；符号：`QueryBestLinkToPeer` | 每 peer 最终选 1 条；搜索最多 3 个网络层 | HCCL rank graph 返回 | Channel 创建期间的临时值 | 提供 Channel 协议和端点 |
| Channel | `ChannelInfo`、`resource.channels` | 文件：`include/custom.h`、`op_host/broadcast.cc`；符号：`AcquireChannels` | 本 rank 到每个其他 rank 1 条，即 `rankSize-1` | Engine Context | 通信域内跨调用复用 | 远端 Write、Channel Notify、远端 HCCL Buffer 地址 |
| Thread | `AlgResourceCtx::threads` | 文件：`op_host/broadcast.cc`；符号：`AcquireThreads` | `1 + min(rankSize-1,15)`；16 rank 为 16 条 | AICPU_TS Engine Context | 通信域内跨调用复用 | 主线程与 Channel worker 任务流 |
| Tile | `TileDesc` | 文件：`op_kernel_aicpu/exec_op.cc`；符号：`MakeTileDesc` | 动态；`ceil(totalBytes/tileBytes)` 个有效 Tile | 执行计划计算值 | 单次函数调用/stripe | 描述连续用户数据片段 |
| Window | `TileDesc::windowIndex` | `MakeTileDesc` | Distributed 通常 2 个：0、1；Direct 为 1 | 执行计划 | 单次调用 | 隔离相邻 stripe 的槽和信号 |
| Slot | [分析命名] `slotIndex/slotOffset` | 文件：`op_kernel_aicpu/exec_op.cc`；符号：`GetSlotAddress` | Distributed 为 `pipelineDepth * ownerCount`；每 Slot `tileBytes` 容量 | 每个 rank 的 HCCL Buffer | 资源长期存在，内容按 batch 复用 | 暂存 root scatter/owner fanout Tile |
| 用户 Buffer | `OpParam::inputPtr/outputPtr` | 文件：`op_host/broadcast.cc`；符号：`HcclBroadcast` | `totalBytes`；两者都等于 `buf` | 调用者 | 用户管理；单次调用引用 | root 源数据与非 root 最终输出 |
| 本地通信 Buffer | `AlgResourceCtx::localBuffer` | `HcclGetHcclBuffer`，保存于 `AlgResourceCtx` | 运行时返回，大小不在仓库写死 | 当前 rank/HCCL | Engine Context 复用 | LocalCopy 暂存与远端 Write 源/目标 |
| 远端通信 Buffer | `ChannelInfo::remoteCclMem` | `HcclChannelGetHcclBuffer` | 每 Channel 1 个 `{addr,size}` | 对端 rank/HCCL；本 rank持有地址描述 | Channel 生命周期 | `HcommWriteOnThread` 的远端目标 |
| Channel Notify | `channelNotifyNum`、索引 0..3 | `BuildResourcePlan`、`AcquireChannels` | 每 Channel 4 个 | Channel 两端 | Channel 生命周期内复用 | 数据到达和 Slot 消费确认 |
| Thread Notify | `notifyNumPerThread` | `BuildResourcePlan`、`HcclThreadAcquire` | `D*(workerCount+1)+1`；16 rank、D=2 时 33 | 每个目标 ThreadHandle | Thread 生命周期内复用 | Host/AICPU、main/worker、coordinator/worker 同步 |
| stream | `aclrtStream stream`、`cpuThread` | `HcclBroadcast`、`HcclThreadAcquireWithStream` | 每次调用绑定 1 个用户 stream CPU Thread | 调用者/CPU_TS | 单次调用导出句柄 | 保持算子与用户 stream 顺序 |
| Kernel task | ACL Kernel 参数/单 block launch | `ops_hccl::LaunchAICPUKernel` | 每次调用 1 次 Kernel launch | ACL Runtime | 单次调用 | 在 AICPU 上生成通信任务 |
| HCOMM task | LocalCopy/Write/Fence/Notify | `exec_op.cc` 各 Submit/Execute 函数 | 随 Tile、peer 数动态增长 | 各 Thread 任务流 | BatchMode/stream 执行期 | 实际搬运与依赖 |

## 实际计算表达式

设：

```text
elementSize = SIZE_TABLE[dataType]                         // Byte/元素
totalBytes = count * elementSize                          // Byte
ownerCount = rankSize - 1                                 // 个数
pipelineDepth = max(1, min(resource.pipelineDepth, 2))    // 个数
commonBufferBytes = min(localBuffer.size, min(remoteCclMem.size)) // Byte
```

### Distributed

```text
slotCount = ownerCount * pipelineDepth                    // Slot 个数
slotCapacity = commonBufferBytes / slotCount              // Byte/Slot
tileSize = AlignDown(min(4 MiB, slotCapacity), elementSize) // Byte
tileCount = ceil(totalBytes / tileSize)                   // Tile 个数
stripeCount = ceil(tileCount / ownerCount)                // stripe 个数

globalTileIndex = stripeIndex * ownerCount + ownerIndex   // Tile 索引
windowId = stripeIndex % pipelineDepth                    // Window 索引
slotId = windowId * ownerCount + ownerIndex               // [分析命名] Slot 索引
bufferOffset = slotId * tileSize                          // HCCL Buffer 字节偏移
userOffset = globalTileIndex * tileSize                   // 用户 Buffer 字节偏移
actualTileSize = min(tileSize, totalBytes - userOffset)   // Byte
```

代码依据：

- 文件：`op_kernel_aicpu/exec_op.cc`
- 符号：`BuildExecutionPlan`、`MakeTileDesc`、`GetSlotAddress`
- 关键变量：`slotCount`、`distributedTileBytes`、`globalTileIndex`、`tile.offset`、`tile.bytes`、`slotIndex`、`slotOffset`
- 行号：L260-L382

### Direct / Parallel Direct

```text
directTileSize = AlignDown(min(4 MiB, commonBufferBytes), elementSize) // Byte
directStripeCount = ceil(totalBytes / directTileSize)                 // 块数
offset = stripe * directTileSize                                      // Byte
actualBytes = min(directTileSize, totalBytes - offset)                // Byte
```

- Serial Direct 可以分块，始终使用 `localBuffer.addr` 和对端 Buffer 的基址，不做 window/owner Slot 偏移。
- Parallel Direct 只有在 `totalBytes <= commonBufferBytes` 时选择，`tileBytes=totalBytes`、`stripeCount=1`，root 先把整包复制到共享本地 HCCL Buffer。

## 普通 Tile 的完整数据路径

选择真实测试拓扑 `rankSize=16`、`root=7`，跟踪 `globalTileIndex=0`：`ownerIndex=0`、`ownerRank=0`、`stripe=0`、`window=0`。再选择 rank 8 作为一个最终目标。

| 步骤 | Rank | Tile | 源 Buffer/偏移 | 目标 Buffer/偏移 | 长度 | 发起方 | 搬运接口 | 代码位置 |
|---:|---:|---:|---|---|---:|---|---|---|
| 1 | 7 root | 0 | root `inputPtr + 0` | root `localBuffer + slot(0,0)` | `tile.bytes` | root 的 owner0 worker | `HcommLocalCopyOnThread` | `ExecuteDistributedRoot` L636-L689 |
| 2 | 7 -> 0 | 0 | root `localBuffer + slot(0,0)` | rank0 `localBuffer + slot(0,0)` | `tile.bytes` | root 主动写 | `HcommWriteOnThread` | `ExecuteDistributedRoot` L670-L676 |
| 3 | 0 owner | 0 | rank0 `localBuffer + slot(0,0)` | rank0 `outputPtr + 0` | `tile.bytes` | rank0 coordinator | `HcommLocalCopyOnThread` | `SubmitReceiveTileData` L405-L418 |
| 4 | 0 -> 8 | 0 | rank0 `localBuffer + slot(0,0)` | rank8 `localBuffer + slot(0,0)` | `tile.bytes` | rank0 到 rank8 的 worker 主动写 | `HcommWriteOnThread` | `SubmitBidirectionalPeerExchange` L438-L473；尾部单向时为 `SubmitSendTile` |
| 5 | 8 | 0 | rank8 `localBuffer + slot(0,0)` | rank8 `outputPtr + 0` | `tile.bytes` | rank8 到 rank0 的 worker | `HcommLocalCopyOnThread` | `SubmitBidirectionalPeerExchange` L464-L469；单向时为 `SubmitReceiveTile` |

[确认] root 自己的 `inputPtr` 与 `outputPtr` 都是原始 `buf`，Distributed 不把 owner Tile 再发回 root；root 的最终数据本来就在用户 Buffer 中。

## 中间 rank 的本地复制与转发顺序

[确认] 在 `SubmitOwnerStripeStart` 中，coordinator 先调用 `SubmitReceiveTileData`，其顺序是：

1. 向 root Record 初始 `SLOT_CONSUMED`，声明 Slot 可写；
2. Wait `DATA_READY`；
3. 把 own Tile 从本地 HCCL Slot 复制到用户 Buffer；
4. coordinator 再向 peer worker Record `WorkerStart`。

因此任务图中的本地复制排在 worker 启动之前。peer worker 随后只读 own Tile Slot做 fanout；本地复制和 fanout都不写该 Slot。最终 coordinator 等所有 peer worker 完成后，才向 root 发最终 `SLOT_CONSUMED`。

依据：文件：`op_kernel_aicpu/exec_op.cc`；符号：`SubmitReceiveTileData`、`SubmitOwnerStripeStart`、`SubmitOwnerStripeFinish`；行号：L405-L418、L708-L756。

## Window 切换与 Slot 覆盖

- [确认] stripe 0/1 分别用 window 0/1；stripe 2 再回到 window 0。
- [确认] root 对每个 owner 以最多两个 stripe 为一批：先对本批有效 Tile完成“初始 consumed -> Write -> DATA_READY”，再逐个等待最终 consumed。只有两次最终 Wait 都已排入相应 worker 流后，才进入下一批并复用 window。
- [确认] owner 按 A/B/C 三阶段处理一批：A 接收 own Tile 并启动 worker；B 提交 peer 交换；C 等所有 worker 完成并向 root 归还 Slot。C 完成后才开始下一批。
- [确认] owner-peer 的每个发送函数末尾都等待远端 `SLOT_CONSUMED`，因此同一 Channel、同一 window 的远端 Slot在下次覆盖前已有消费依赖。
- [推测] “Wait 调用返回”在这里表示任务成功编排到 Thread 流，而不是 Host 立即阻塞到硬件完成；Slot 安全性依赖 HCOMM 对同一 Thread 上任务顺序和 Notify 的定义。

## 尾 Tile 与 4B 通信

[确认] `MakeTileDesc` 先计算字节偏移，再用：

```text
tile.bytes = min(plan.tileBytes, plan.totalBytes - tile.offset)
```

因此最后不足一个 Tile 的部分不会按固定 Tile 长度越界访问。

对真实用例 `400 MiB + 4 B = 419,430,404 B`：

- 若 `commonBufferBytes >= 120 MiB`，则 16-rank、双窗口下单 Slot 容量至少 4 MiB，`tileSize=4 MiB`。
- 此时 `tileCount=101`、`stripeCount=7`。
- 最后 Tile 是 `globalTileIndex=100`：`stripe=6`、`ownerIndex=10`、`ownerRank=11`（因为 root=7）、`window=0`、`actualTileSize=4 B`。
- [确认] 该 4B Tile 不是附着在前一个 Tile 上；它独立执行一次 root->owner Scatter，并由 owner rank 11 对其他非 root rank执行完整的有效半边/双边 fanout 协议。
- [确认] 同一最后 stripe 中超出 `totalBytes` 的 owner Tile被标记 `valid=false`；两端根据 `ownTile.valid/peerTile.valid` 选择双向、仅发送、仅接收或不通信，避免为不存在的 Tile制造 Channel Wait。

[待确认] `commonBufferBytes` 的运行时实际值不在仓库中；若不足 120 MiB，`tileSize` 会按上述公式缩小，因此具体的最后 `globalTileIndex` 和 owner 也会变化，但尾块长度与协议选择公式不变。
