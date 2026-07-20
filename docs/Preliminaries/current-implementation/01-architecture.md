# 架构、角色与调用链

## 入口与边界

| 层级 | 入口 | 位置 | 当前边界 |
|---|---|---|---|
| Host API | `HcclBroadcast` | 文件：`hccl_broadcast_problem_template/op_host/broadcast.cc`；行号：L199-L280 | 参数与资源控制面 |
| Kernel 启动 | `ops_hccl::LaunchAICPUKernel` | 文件：`hccl_broadcast_problem_template/op_host/launch_aicpu_kernel.cc`；行号：L47-L80 | 加载并下发 AICPU Kernel，连接用户 stream 与 AICPU 主线程 |
| AICPU Kernel | `HcclAICPUKernel` | 文件：`hccl_broadcast_problem_template/op_kernel_aicpu/aicpu_kernel.cc`；行号：L19-L60 | 反序列化、BatchMode、Host/AICPU 握手、调用算法编排 |
| AICPU 算法 | `ops_hccl::ExecOp` | 文件：`hccl_broadcast_problem_template/op_kernel_aicpu/exec_op.cc`；行号：L849-L869 | 动态计划与三档算法分派 |
| CCU | [待确认] 无仓库内入口 | 无 CCU 源文件或函数 | HCOMM 原语之后的执行主体不在当前仓库中 |

## 初始化/资源申请调用链

冷路径只在同一通信域、同一资源 Tag 第一次调用时走；Tag 是 `hccl_custom_broadcast_v3`，不包含 root、count 或用户 Buffer。

| 顺序 | 文件 | 函数 | 调用者 | 主要职责 | 下一级调用 |
|---:|---|---|---|---|---|
| 1 | `op_host/broadcast.cc` | `HcclBroadcast` | 用户程序/HCCL 测试程序 | 填充 `OpParam`，取得 `myRank/rankSize`，判定冷/热路径 | `ValidateBroadcastParam`、HCCL 资源 API |
| 2 | `op_host/broadcast.cc` | `BuildResourcePlan` | `HcclBroadcast` | 计算主线程、worker、Thread Notify、Channel Notify 数量 | 无 |
| 3 | `op_host/broadcast.cc` | `AcquireThreads` | `HcclBroadcast` | 从 `COMM_ENGINE_AICPU_TS` 申请 `threads`，固定 `threads[0]` 为主线程 | `HcclThreadAcquire` |
| 4 | `op_host/broadcast.cc` | `QueryBestLinkToPeer` | `AcquireChannels` | 从网络层 0..2 选择第一个非保留 Link | `HcclRankGraphGetLinks` |
| 5 | `op_host/broadcast.cc` | `AcquireChannels` | `HcclBroadcast` | 对每个 `remoteRank != myRank` 申请 Channel，取得远端 HCCL Buffer | `HcclChannelDescInit`、`HcclChannelAcquire`、`HcclChannelGetHcclBuffer` |
| 6 | `op_host/broadcast.cc` | `CreateAndStoreEngineContext` | `HcclBroadcast` | 在 AICPU_TS 保存完整 `AlgResourceCtx`，在 CPU_TS 保存 AICPU 主线程句柄 | `AlgResourceCtx::Serialize`、`HcclEngineCtxCreate/Copy` |
| 7 | `op_host/broadcast.cc` | `HcclBroadcast` 热路径块 | 后续调用 | 复用 AICPU Engine Context，并把保存的 AICPU Thread 导出回 CPU_TS | `HcclEngineCtxGet`、`HcclThreadExportToCommEngine` |

依据：文件：`op_host/broadcast.cc`；符号：`kResourceTag`、`ResourcePlan`、上述函数；关键变量：`resCtxHost`、`resource.channels`、`param.resCtx`；行号：L25-L195、L236-L275。

[确认] `count==0` 或 `rankSize==1` 在申请用户 stream 对应 CPU Thread 之前直接返回成功。其他调用即使最终选择串行 Direct，也会在冷路径申请全连接 Channel 和最多 15 个 worker。

## Kernel 下发调用链

| 顺序 | 文件 | 函数 | 调用者 | 主要职责 | 下一级调用 |
|---:|---|---|---|---|---|
| 1 | `op_host/launch_aicpu_kernel.cc` | `LoadAICPUKernel` | `LaunchAICPUKernel` | 从自定义 AICPU JSON 加载二进制；句柄是 `thread_local` 缓存 | `aclrtBinaryLoadFromFile` |
| 2 | 同上 | `ops_hccl::LaunchAICPUKernel` | `HcclBroadcast` | CPU Thread 向 AICPU Thread Record Notify 0 | `HcommThreadNotifyRecordOnThread` |
| 3 | 同上 | `ops_hccl::LaunchAICPUKernel` | 同上 | 把整个 `OpParam` 作为 Kernel 参数，在用户 stream 下发单 block Kernel | `aclrtLaunchKernelWithConfig` |
| 4 | 同上 | `ops_hccl::LaunchAICPUKernel` | 同上 | CPU Thread 等待 AICPU 回传 Notify 0 | `HcommThreadNotifyWaitOnThread` |
| 5 | `op_kernel_aicpu/aicpu_kernel.cc` | `HcclAICPUKernel` | ACL Runtime | 反序列化 `AlgResourceCtx`，开启 BatchMode，等待 Host 启动信号 | `AlgResourceCtx::DeSerialize`、`HcommBatchModeStart` |
| 6 | 同上 | `HcclAICPUKernel` | 同上 | 调用算法编排，随后向 Host Record 完成信号，结束 BatchMode | `ops_hccl::ExecOp`、`HcommBatchModeEnd` |

## 单个 Distributed Tile 的调用链

这里用普通有效 Tile 描述；`tile` 的 owner 不是 root。

| 顺序 | Rank/线程 | 文件 | 函数 | 主要职责 | 下一级调用 |
|---:|---|---|---|---|---|
| 1 | root / 对应 owner worker | `op_kernel_aicpu/exec_op.cc` | `ExecuteDistributedRoot` | 从 root 用户 Buffer 把 Tile 拷入 root 本地 `[window][owner]` Slot | `MakeTileDesc`、`GetSlotAddress`、`HcommLocalCopyOnThread` |
| 2 | owner / root-Channel coordinator | 同上 | `SubmitReceiveTileData` | 先向 root 声明 Slot 可写，等待 root 的 `DATA_READY`，再拷入 owner 用户 Buffer | Channel Notify、`HcommLocalCopyOnThread` |
| 3 | root / owner worker | 同上 | `ExecuteDistributedRoot` | 等 owner 初始许可，主动 Write 到 owner Slot，Fence 后发布 `DATA_READY` | `HcommWriteOnThread`、`HcommChannelFenceOnThread` |
| 4 | owner / coordinator | 同上 | `SubmitOwnerStripeStart` | 在本地复制排队后，以 Thread Notify 启动所有非 root peer worker | `HcommThreadNotifyRecordOnThread` |
| 5 | owner / peer worker | 同上 | `SubmitPeerWorkerStripe` | 等 coordinator 启动，按两端 Tile 是否有效选择双向、仅发、仅收或空操作 | `SubmitPeerExchange` |
| 6 | owner 与另一个 owner / 各自 peer worker | 同上 | `SubmitBidirectionalPeerExchange` | 双方先主动 Write 自己的 Tile，再接收对方 Tile并复制到各自用户 Buffer | Write、Fence、Channel Notify、LocalCopy |
| 7 | owner / coordinator | 同上 | `SubmitOwnerStripeFinish` | 等所有 peer worker 完成本 stripe；确认无人再读 own Tile 后向 root 返回最终 `SLOT_CONSUMED` | `HcommThreadNotifyWaitOnThread`、Channel Notify Record |
| 8 | root / owner worker | 同上 | `ExecuteDistributedRoot` 第二个 stripe 循环 | 等最终 `SLOT_CONSUMED`，允许该 window/owner Slot 在下一批复用 | Channel Notify Wait |

[确认] 发送方主动写：跨 rank 数据接口只有 `HcommWriteOnThread`，没有 `HcommReadOnThread`。

## 三种 rank 角色

### root

- [确认] Direct：逐块把数据复制到本地 HCCL Buffer，再按 Channel 顺序写所有非 root rank。
- [确认] Parallel Direct：主线程把整包复制一次；每条发送 Channel 由绑定 worker 执行 ACK/Write/Fence/DATA/ACK。
- [确认] Distributed：只执行 Scatter，把每个 block-cyclic Tile 发给其 owner；root 不参与 owner 之间的全互换。

### 中间 rank / owner

- [确认] Distributed 中每个非 root rank 都是一个 owner：接收 root 分配给自己的 Tile，先排队复制到本 rank 用户 Buffer，再把该 Tile 发给其余所有非 root rank；同时接收其他 owner 的 Tile。
- [确认] owner 的 root Channel worker 充当 coordinator；其他 Channel worker 负责与对应 peer 的 pairwise 交换。
- [确认] 代码不存在只有少数固定“中间 rank”的树形结构；所有非 root rank 都同时扮演 owner 和其他 owner Tile 的叶子接收方。

### 叶子 rank

- [确认] Direct/Parallel Direct 的所有非 root rank 都是叶子，只与 root 通信，并始终走 `ExecuteDirectPeer`。
- [确认] Distributed 没有纯叶子：每个非 root rank 对自己拥有的 Tile 是中间 owner，对其他 Tile 是最终接收方。

## coordinator 与 worker 的分支

- [确认] `threads[0]` 是 AICPU 主线程，负责 Host/AICPU 握手、启动全部 worker、等待最终 worker 完成；它在 Direct 中也直接执行数据任务。
- [确认] Distributed owner 上，与 root Channel 绑定的 worker 是 coordinator。`ExecuteDistributedOwner` 明确跳过该 Channel 的 peer-worker 分支。
- [确认] 其余 worker 按 `channel.workerIndex` 执行一个 peer Channel 的本 stripe 协议。16 rank 当前资源规划下是每 Channel 唯一 worker。
- [确认] Parallel Direct 只有 root 使用 worker；非 root 仍在 AICPU 主线程执行 `ExecuteDirectPeer`。
- [确认] 分支依据是 `param.myRank == param.root`、Channel 的 `remoteRank == param.root`、以及 `TileDesc::valid`，没有按物理 Server ID 分支。

## 通信结束与资源释放

| 顺序 | 文件 | 函数 | 调用者 | 主要职责 | 下一步 |
|---:|---|---|---|---|---|
| 1 | `op_kernel_aicpu/exec_op.cc` | `NotifyWorkerDone` | 并行/分布式执行函数 | 每个已启动 worker 向 AICPU 主线程的唯一索引 Record 最终完成 | `WaitAllWorkers` |
| 2 | 同上 | `WaitAllWorkers` | 并行/分布式执行函数 | 主线程等待 `1..workerCount` | 返回 `ExecOp` |
| 3 | `op_kernel_aicpu/aicpu_kernel.cc` | `HcclAICPUKernel` | Runtime | AICPU 主线程向 Host CPU Thread Record Notify 0，随后 `HcommBatchModeEnd` | Kernel 返回 |
| 4 | `op_host/launch_aicpu_kernel.cc` | `LaunchAICPUKernel` | `HcclBroadcast` | CPU Thread Wait Notify 0 后返回 | `HcclBroadcast` 返回 |

- [确认] Direct 没有 worker，靠每个 Channel 的最终 ACK 保证进入下一块前对端已排队完成本地复制。
- [确认] 当前仓库没有调用 Thread、Channel 或 Engine Context 的显式 Release/Destroy API；它们设计为同一通信域内复用。
- [待确认] 通信域销毁时由谁、何时释放这些资源，当前仓库没有代码依据。
- [确认] 错误分支中 `HcclAICPUKernel` 会在 Host Wait 失败或 `ExecOp` 失败时直接返回，未执行后面的 `HcommBatchModeEnd`；错误后的运行时清理语义不在仓库内。
