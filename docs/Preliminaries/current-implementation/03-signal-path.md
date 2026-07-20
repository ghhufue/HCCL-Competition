# Notify 与同步通路

## Notify 编号基础

### Channel Notify

| [分析命名] 信号 | ID 公式 | 源码定义 |
|---|---:|---|
| Direct `ACK` | `0` | `NOTIFY_IDX_ACK`，文件：`include/common.h`，L20 |
| Direct `DATA_READY` | `1` | `NOTIFY_IDX_DATA_SIGNAL`，文件：`include/common.h`，L21 |
| Distributed `DATA_READY[window]` | `windowIndex` | `DataReadyNotifyIndex`，文件：`op_kernel_aicpu/exec_op.cc`，L133-L137 |
| Distributed `SLOT_CONSUMED[window]` | `pipelineDepth + windowIndex` | `SlotConsumedNotifyIndex`，同文件 L139-L144 |

当前 `pipelineDepth=2`，所以 Distributed 使用 `DATA_READY={0,1}`、`SLOT_CONSUMED={2,3}`。每条 Channel 各自申请 4 个槽，peer 维度由 ChannelHandle 隔离。

### Thread Notify

| [分析命名] 信号 | 目标 Thread 上的 ID 公式 | 源码定义 |
|---|---:|---|
| Host -> AICPU 启动 | `0` | `LaunchAICPUKernel` / `HcclAICPUKernel` |
| AICPU -> Host 编排完成 | `0` | 同上，目标是另一个 ThreadHandle |
| coordinator -> peer worker 的 window 启动 | `windowIndex` | `WorkerStartNotifyIndex` |
| peer worker -> coordinator 的 window 完成 | `pipelineDepth + windowIndex * ownerCount + (workerIndex - 1)` | `WorkerDoneNotifyIndex` |
| main -> 全 worker 一次性启动 | `pipelineDepth * (workerCount + 1)` | `WorkerLaunchNotifyIndex` |
| worker -> main 最终完成 | `workerIndex` | `NotifyWorkerDone` / `WaitAllWorkers` |

[确认] 16 rank、`workerCount=ownerCount=15`、`pipelineDepth=2` 时，每个 Thread 申请 33 个槽：window start 0/1，window done 2..31，launch 32。最终 worker done 1..15 和 Host/AICPU 0 以主线程为目标；相同数字在其他目标 Thread 上不冲突。

## Notify 总表

| 信号含义 | ID 公式 | 申请位置 | Post 方/位置 | Wait 方/位置 | 区分维度 | 复用条件 |
|---|---|---|---|---|---|---|
| Host 启动 AICPU | Thread 0 | `HcclThreadAcquireWithStream` 与 `HcclThreadAcquire` | CPU Thread；`LaunchAICPUKernel` L53 | AICPU 主线程；`HcclAICPUKernel` L36 | 目标 Thread、单次调用 | 本次 Wait 已消费；调用按用户 stream 排序 |
| AICPU 通知 Host 编排完成 | Thread 0 | 同上 | AICPU 主线程；`HcclAICPUKernel` L48 | CPU Thread；`LaunchAICPUKernel` L79 | 目标 Thread、单次调用 | 本次 Host Wait 已消费 |
| [分析命名] WORKER_LAUNCH | `D*(workerCount+1)` | `BuildResourcePlan` 计算 `notifyNumPerThread` | AICPU 主线程；`StartAllWorkers` | 每个 worker；`StartAllWorkers` | 目标 worker、调用轮次 | 每 worker 的 launch Wait 已排入其流；下一次调用再复用 |
| [分析命名] WORKER_START | `window` | 同上 | owner coordinator；`SubmitOwnerStripeStart` | 对应 peer worker；`SubmitPeerWorkerStripe` | 目标 worker、window、stripe 轮次 | 同 window 的上一批 WORKER_DONE 已被 coordinator 消费 |
| [分析命名] WORKER_DONE | `D + window*ownerCount + workerIndex-1` | 同上 | peer worker；`SubmitPeerWorkerStripe` | owner coordinator；`SubmitOwnerStripeFinish` | worker、window、stripe 轮次 | stage C Wait 完成本批后才进入下一批 |
| [分析命名] WORKER_FINAL_DONE | `workerIndex` | 同上 | 每个 worker；`NotifyWorkerDone` | AICPU 主线程；`WaitAllWorkers` | worker、调用轮次 | 本次 main Wait 已消费；下一次调用再复用 |
| Direct 初始写许可 | Channel 0 | `desc.notifyNum=4`；`AcquireChannels` | 非 root；`ExecuteDirectPeer` 首个 Record ACK | root；`ExecuteDirectRoot`/`ExecuteParallelDirectRoot` 首个 Wait ACK | Channel(peer)、块 | root 完成该 Wait 后才能 Write |
| Direct 数据到达 | Channel 1 | 同上 | root；Write+Fence 后 Record DATA | 非 root；Wait DATA | Channel(peer)、块 | 非 root Wait 后再复制；下一块需完成最终 ACK |
| Direct 最终消费 | Channel 0 | 同上 | 非 root；LocalCopy 后第二个 Record ACK | root；第二个 Wait ACK | Channel(peer)、块 | root Wait 后才能复用 Buffer；与初始许可在同 ID 上串行使用 |
| root->owner 初始写许可 | Channel `D+window` | 同上 | owner coordinator；`SubmitReceiveTileData` 首个 Record | root owner worker；`ExecuteDistributedRoot` 第一个 Wait | Channel(owner)、window、stripe | root 收到许可后写该 Slot |
| root->owner 数据到达 | Channel `window` | 同上 | root owner worker；Write+Fence 后 Record | owner coordinator；`SubmitReceiveTileData` Wait | Channel(owner)、window、stripe | owner Wait 后复制/启动 fanout |
| root->owner 最终 Slot 消费 | Channel `D+window` | 同上 | owner coordinator；所有 peer worker 完成后 `SubmitOwnerStripeFinish` Record | root owner worker；`ExecuteDistributedRoot` 第二个 Wait | Channel(owner)、window、stripe | 本批最终 Wait 后才复用同 window |
| owner<->peer 数据到达 | Channel `window` | 同上 | 每一端的 peer worker；Write+Fence 后 Record | 对端 peer worker；Wait DATA | Channel(peer)、direction、window、stripe | 对端消费 DATA；同 window 下次发送前还需 consumed |
| owner<->peer Slot 消费 | Channel `D+window` | 同上 | 接收方 LocalCopy 后 Record | 原发送方 Wait | Channel(peer)、direction、window、stripe | 原发送方 Wait 后可覆盖远端 Slot |

## Wait/Post 配对核查

### Direct 和 Parallel Direct

每个非 root、每个块在同一 root Channel 上的严格配对是：

```text
peer Record ACK(0)       -> root Wait ACK(0)
root Record DATA(1)      -> peer Wait DATA(1)
peer Record ACK(0)       -> root Wait ACK(0)
```

[确认] root 的 Write 前有初始 ACK，下一块覆盖本地源 Buffer 前，root 已遍历所有 Channel并等待最终 ACK。Parallel Direct 的 root worker 使用同一三步协议；非 root 仍执行 `ExecuteDirectPeer`。

### root 与 owner

每个有效 Tile：

```text
owner Record CONSUMED[w] -> root Wait CONSUMED[w]    // 初始写许可
root Record DATA[w]      -> owner Wait DATA[w]
owner Record CONSUMED[w] -> root Wait CONSUMED[w]    // fanout 完成后最终归还
```

[确认] 两个 `CONSUMED` 使用同一 ID，但 Post/Wait 数量均为 2，且代码顺序固定；第二个 Post 在所有 peer worker done 之后。

### owner 与 peer owner

两侧 Tile都有效时，两端执行对称的 `SubmitBidirectionalPeerExchange`：双方都先 Write/Fence/Record DATA，再 Wait 对端 DATA；随后 LocalCopy、Record 对端 consumed、Wait 自己发送 Tile 的 consumed。因此每个方向都有唯一的一对 DATA 和 consumed。

最后 stripe 只有一侧 Tile有效时：

- 有 Tile 一侧走 `SubmitSendTile`：Post DATA，Wait consumed。
- 对端走 `SubmitReceiveTile`：Wait DATA，Post consumed。
- 两侧都无 Tile 时均不发 Channel Notify。

[确认] `SubmitPeerExchange` 的 `ownTile.valid/peerTile.valid` 四分支使尾部不会留下仅一侧存在的 Channel Wait。

### Thread 信号

- [确认] `StartAllWorkers` 对每个 worker 恰好一条 main Record 和一条 worker Wait。
- [确认] 每个有效或无效 stripe，coordinator 都会启动每个非 root peer worker；该 worker 无论是否有 Channel 数据操作，最终都 Record 一条 window done，因此 stage C 的 Wait 不悬空。
- [确认] 并行路径结束时每个已启动 worker恰好向 main Record 一次；`WaitAllWorkers` 对 `1..workerCount` 各 Wait 一次。

## 一个 Distributed Tile 的信号时序

```mermaid
sequenceDiagram
    participant M as "AICPU main"
    participant RW as "root owner-worker"
    participant OC as "owner coordinator"
    participant OP as "owner peer-worker"
    participant PP as "peer worker"

    M->>RW: StartAllWorkers / WORKER_LAUNCH
    M->>OC: StartAllWorkers / WORKER_LAUNCH
    M->>OP: StartAllWorkers / WORKER_LAUNCH
    Note over RW,OC: ExecuteDistributedRoot + SubmitReceiveTileData
    OC-->>RW: Record SLOT_CONSUMED[window] 初始写许可
    RW->>RW: HcommLocalCopyOnThread(root user -> root slot)
    RW->>OC: HcommWriteOnThread(root slot -> owner slot)
    RW->>RW: HcommChannelFenceOnThread
    RW-->>OC: Record DATA_READY[window]
    OC->>OC: Wait DATA_READY; LocalCopy(owner slot -> owner user)
    Note over OC,OP: SubmitOwnerStripeStart
    OC-->>OP: Record WORKER_START[window]
    OP->>OP: Wait WORKER_START[window]
    Note over OP,PP: SubmitBidirectionalPeerExchange
    OP->>PP: Write own Tile; Fence; Record DATA_READY
    PP->>OP: Write peer Tile; Fence; Record DATA_READY
    OP->>OP: Wait DATA_READY; LocalCopy peer Tile
    PP->>PP: Wait DATA_READY; LocalCopy owner Tile
    OP-->>PP: Record SLOT_CONSUMED
    PP-->>OP: Record SLOT_CONSUMED
    OP->>OP: Wait own SLOT_CONSUMED
    OP-->>OC: Record WORKER_DONE[worker,window]
    Note over OC,RW: SubmitOwnerStripeFinish
    OC->>OC: Wait all WORKER_DONE[*,window]
    OC-->>RW: Record SLOT_CONSUMED[window] 最终归还
    RW->>RW: Wait SLOT_CONSUMED[window]
```

## Reset、Barrier、Fence 与完成判断

- [确认] 当前 Broadcast 主路径没有 Notify Reset 调用。
- [确认] 没有显式 Barrier 调用。
- [确认] 没有 `HcommChannelDrainOnThread` 调用。
- [确认] 每次远端 Write 后都调用 `HcommChannelFenceOnThread`，随后才 Record `DATA_READY`。它是 Channel 内写与通知的顺序点，不应写成全局 Barrier。
- [确认] worker 级结束由 `WaitAllWorkers` 判断；Tile/Slot 级结束由 `SLOT_CONSUMED` 和 window done 判断；Host 级 Notify表示 AICPU 任务编排完成。

## pipelineDepth > 1 的复用检查

- [确认] Channel DATA/consumed 以 window 区分；Thread start/done 也以 window 区分。
- [确认] 同 window 在 stripe `s` 与 `s+2` 之间复用。root 的第二轮 consumed Wait、owner 的 stage C、peer 发送末尾 consumed Wait共同建立复用前依赖。
- [确认] worker done 还包含 `workerIndex`，不同 peer 不共享 coordinator 上的同一完成槽。
- [确认] ChannelHandle 隔离 peer；方向由两端各自 Record/Wait 的 Channel 语义隔离。代码没有额外 direction 位。
- [待确认] HCOMM Notify 是计数、事件还是其他队列语义，以及同一 ID 连续两次 ACK/CONSUMED 的底层容量/消费规则，不在仓库中定义。
- [推测] 正常成功路径的 Post/Wait 数量和顺序是平衡的；如果某次调用在中途 API 错误返回，代码没有 Reset/Drain/清理剩余 Notify，后续复用同一 Engine Context 是否安全无法由仓库确认。
