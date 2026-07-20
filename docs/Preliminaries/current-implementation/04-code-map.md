# 功能与代码块映射

## 总表

- [未使用] `ResourceLayoutVersion::VERSION_1` 和 `VERSION_2` 仍保留在枚举中，但当前 Host 固定序列化 `VERSION_3`，AICPU 的 `ValidateExecutionContext` 也只接受 `VERSION_3`。
- [未使用] `OpParam::reduceType` 属于通用参数结构，本 Broadcast 主路径不读取它；`opType` 被 Host 设为 `HCCL_CMD_BROADCAST`，但当前 `ExecOp` 不再以它分支。

| 功能 | 文件 | 函数/代码块 | 输入 | 输出 | 修改的状态 | 相关 Buffer/Notify |
|---|---|---|---|---|---|---|
| rank 信息与参数校验 | `include/common.h`、`op_host/broadcast.cc` | `OpParam`、`ValidateBroadcastParam`、`HcclBroadcast` | `count/dataType/root/comm` | `myRank/rankSize` 或错误码 | 填充单次 `OpParam` | 用户 Buffer 仅保存指针 |
| 上下游/owner 映射 | `op_kernel_aicpu/exec_op.cc` | `OwnerIndexToRank`、`RankToOwnerIndex`、`FindChannelByRemoteRank` | root、rank、Channel 表 | owner/rank/Channel | 无 | 选择 root Channel 或 peer Channel |
| Link 与 Channel 选择 | `op_host/broadcast.cc` | `QueryBestLinkToPeer`、`AcquireChannels` | local/remote rank、rank graph | 全连接 `ChannelInfo` | Engine Context 静态资源 | 远端 HCCL Buffer、4 Channel Notify |
| Tile 切分与尾块 | `op_kernel_aicpu/exec_op.cc` | `BuildExecutionPlan`、`MakeTileDesc` | totalBytes、Buffer 最小容量 | `tileBytes/stripeCount/TileDesc` | 单次执行计划 | `tile.bytes` 控制所有搬运长度 |
| Window/Slot 选择 | 同上 | `MakeTileDesc`、`GetSlotAddress` | stripe、owner、tileBytes | Slot 地址 | 无；只计算 | 本地/远端 HCCL Buffer `[window][owner]` |
| Direct 发送/接收 | 同上 | `ExecuteDirectRoot`、`ExecuteDirectPeer` | 用户 Buffer、root Channel | 所有非 root 用户 Buffer | Channel/Thread 任务流 | ACK 0、DATA 1；HCCL Buffer 基址 |
| Parallel Direct | 同上 | `ExecuteParallelDirectRoot` | 整包、所有 Channel worker | 并行 root 星形发送任务 | worker 与 main 任务流 | 共享 root HCCL Buffer；launch/final done；ACK/DATA |
| Distributed root scatter | 同上 | `ExecuteDistributedRoot` | root 用户 Tile、owner Channel | owner HCCL Slot | owner worker 任务流 | DATA[window]、CONSUMED[window] |
| owner 接收与本地复制 | 同上 | `SubmitReceiveTileData` | owner 本地 HCCL Slot | owner 用户 Buffer | coordinator 任务流 | 初始 consumed、DATA Wait |
| owner fanout/互换 | 同上 | `SubmitSendTile`、`SubmitReceiveTile`、`SubmitBidirectionalPeerExchange` | own/peer Tile | peer HCCL Slot与本地用户 Buffer | peer worker 任务流 | DATA/CONSUMED 双向配对 |
| Notify 申请 | `op_host/broadcast.cc` | `BuildResourcePlan`、`AcquireThreads`、`AcquireChannels` | rankSize、D=2 | Thread/Channel Notify 槽 | 静态资源 | Thread 33、Channel 4（16 rank） |
| Notify Wait/Post | `launch_aicpu_kernel.cc`、`aicpu_kernel.cc`、`exec_op.cc` | Host/AICPU 握手、`StartAllWorkers`、各 Submit/Execute 函数 | Thread/Channel Handle、索引 | 任务依赖边 | 各任务流 | 详见 `03-signal-path.md` |
| Thread 创建与分工 | `op_host/broadcast.cc`、`include/custom.h` | `BuildResourcePlan`、`AcquireThreads`、`ChannelInfo::workerIndex` | rankSize | 主线程 + worker | Engine Context | 每 Channel 绑定 worker |
| pipeline 推进 | `op_kernel_aicpu/exec_op.cc` | Distributed 中 `baseStripe/batchEnd` 三阶段循环 | stripeCount、D | 最多 D 个 window 的批次 | 任务提交顺序 | window start/done、Slot consumed |
| 完成判断 | 同上 | `SubmitOwnerStripeFinish`、`WaitAllWorkers` | worker/Channel 信号 | Slot 可复用、算法编排结束 | 消费 Notify | window done、final done |
| 资源复用/释放 | `op_host/broadcast.cc` | `HcclEngineCtxGet` 热路径 | resource Tag | 重用静态 Context | 不创建新资源 | [确认] 无显式释放代码 |

## 主路径关键函数

### 1. `HcclBroadcast`

- 位置：文件：`hccl_broadcast_problem_template/op_host/broadcast.cc`；行号：L199-L280。
- 调用者/下一级：用户调用；下调参数校验、资源申请/复用和 `ops_hccl::LaunchAICPUKernel`。
- 参数来源：`buf/count/dataType/root/comm/stream` 都来自公开 API。`inputPtr` 与 `outputPtr` 都指向 `buf`。
- 关键状态：`param.tag` 固定为 v3 资源布局；`param.resCtx/ctxSize` 指向序列化 Engine Context。
- Buffer：只保存用户 Buffer 地址；冷路径查询本地 HCCL Buffer。
- Notify：申请用户 stream 对应 CPU Thread，并导出 CPU/AICPU 两侧 Thread 句柄；实际 Record/Wait 在 launcher 中。
- 前置/保证：[确认] 校验空指针、rank、数据类型和乘法溢出；`count==0/rankSize==1` 直接成功。成功返回只保证任务已按 stream 下发，不是 CPU 同步完成。
- 尾块/pipeline：本函数不计算，交给 AICPU 动态计划。

### 2. `AcquireChannels`

- 位置：文件：`hccl_broadcast_problem_template/op_host/broadcast.cc`；行号：L130-L178。
- 调用者/下一级：冷路径 `HcclBroadcast` 调用；下调 `QueryBestLinkToPeer`、Channel 申请和远端 Buffer 查询。
- 参数物理含义：`param.myRank/rankSize` 定义本 rank 与所有 peer；`plan.channelNotifyNum=4`。
- 关键变量：`remoteRanks` 与描述符都按 rank 升序构造；`workerIndex = 1 + idx % workerCount`。
- Buffer：把每个对端 HCCL Buffer 的基址/大小写入 `ChannelInfo::remoteCclMem`。
- Notify：每 Channel 申请 4 个槽。
- 前置/保证：成功后 `resource.channels.size()==rankSize-1`，且顺序稳定；选择的是前三网络层中第一个非 RESERVED Link。
- 边界：[确认] `rankSize<=1` 无 Channel；超过 16 rank 时 Channel 可能共享最多 15 worker。

### 3. `ops_hccl::LaunchAICPUKernel` 与 `HcclAICPUKernel`

- 位置：文件：`op_host/launch_aicpu_kernel.cc`，L47-L80；文件：`op_kernel_aicpu/aicpu_kernel.cc`，L19-L60。
- 调用关系：`HcclBroadcast -> LaunchAICPUKernel -> ACL Runtime -> HcclAICPUKernel -> ops_hccl::ExecOp`。
- 参数：整个 `OpParam` 按值追加为 Kernel 参数；其中 Engine Context 是地址+长度。
- Buffer：不搬运用户数据，只传递地址。
- Notify：Host Record/AICPU Wait Thread Notify 0；AICPU Record/Host Wait 0。
- 前置/保证：AICPU 在 `HcommBatchModeStart` 内编排算法；成功路径最后 `HcommBatchModeEnd`。
- 边界：[确认] 任一前置 Wait 或 `ExecOp` 错误会提前返回，后续完成 Notify/BatchModeEnd 不执行。

### 4. `BuildExecutionPlan`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L260-L335。
- 调用者/下一级：`ops_hccl::ExecOp` 调用；内部使用资源校验与 Buffer 容量辅助函数，不下发 HCOMM 任务。
- 参数含义：本次 `OpParam` + 跨调用复用的 `AlgResourceCtx`。
- 关键变量：`totalBytes`、`commonBufferBytes`、`ownerCount`、`pipelineDepth`、`slotCapacity`、`distributedTileBytes`。
- Buffer：只读取地址非空与 size，不读写内容。
- Notify：只检查数量是否满足算法。
- 前置/保证：输出唯一的三档 `algorithm`、Tile/stripe 计划；资源不足退化为串行 Direct，不静默跳过通信。
- 边界：尾部由 `stripeCount=ceil(tileCount/ownerCount)` 保留；Tile 至少按元素大小对齐，Distributed 小于 4KiB Slot 则退化。

### 5. `MakeTileDesc` 与 `GetSlotAddress`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L338-L382。
- 调用者：root scatter、owner 接收、peer 交换各路径。
- 参数含义：`stripeIndex` 是流水批次内的 stripe；`ownerIndex` 是跳过 root 后的 owner 序号。
- 关键偏移：`globalTileIndex=stripe*ownerCount+owner`；用户偏移是 `globalTileIndex*tileBytes`；HCCL Slot 偏移是 `(window*ownerCount+owner)*tileBytes`。
- Buffer：`GetSlotAddress` 同时用于本地和远端 Buffer；在加地址前检查溢出与边界。
- Notify：不直接操作；返回的 `windowIndex` 决定 Notify ID。
- 保证：[确认] 最后 Tile 用剩余字节数；不存在的 Tile返回 `valid=false`。

### 6. `ExecuteDirectRoot` / `ExecuteDirectPeer`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L476-L530。
- 调用者：`ExecuteDirectFanout`；Parallel Direct 的非 root 也复用 `ExecuteDirectPeer`。
- 数据：root 用户块 -> root 本地 HCCL Buffer -> 非 root HCCL Buffer -> 非 root 用户块。
- Notify：每块、每 Channel 是 `ACK -> DATA -> ACK`。
- 前置/保证：root 每次覆盖共享本地槽前，上一块所有 peer 已返回最终 ACK；非 root 的 LocalCopy 在最终 ACK 前。
- 尾块：`bytes=min(tileBytes,totalBytes-offset)`；4B 输入只走一个完整块协议。

### 7. `ExecuteParallelDirectRoot`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L585-L622。
- 调用者：`ExecuteParallelDirectFanout`，仅 root 分支。
- 数据：主线程把整包从用户 Buffer 复制到共享 `localBuffer` 一次；所有 worker只读它并写各自 peer。
- Notify：先 `StartAllWorkers`；每 Channel仍用 Direct ACK/DATA；所有 worker 最后各 Record 一条 final done，main 全部 Wait。
- 前置/保证：只在整包能放入共同 Buffer 且 worker/Notify 资源检查通过时选择。主线程 LocalCopy 在所有 launch Record 之前。
- 边界：只有一个 stripe，无尾块分段；64KiB 与 1MiB 端点均包含在本算法请求范围。

### 8. `ExecuteDistributedRoot`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L636-L690。
- 调用者：`ExecuteDistributedScatterFanout` 的 root 分支。
- 数据：按 Channel/owner 把对应 block-cyclic Tile从 root 用户 Buffer复制到本地 Slot，再主动写远端 owner 的同 Slot。
- Notify：每 Tile先 Wait 初始 consumed，Write/Fence 后 Record DATA；本批所有 Tile发出后再 Wait 最终 consumed。
- 状态：外层按 Channel，内层按 `baseStripe += pipelineDepth`。每 Channel绑定的 worker流独立，可并发执行。
- 保证：最终 consumed 表示 owner 已完成该 Tile对所有 peer 的 fanout，随后 window 可复用。
- 尾块：无效 owner Tile跳过全部 Channel操作；有效的 4B Tile按真实 `tile.bytes` 搬运。

### 9. `ExecuteDistributedOwner`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L779-L834。
- 调用者：`ExecuteDistributedScatterFanout` 的非 root 分支。
- 参数/角色：`ownOwnerIndex` 由本 rank 与 root 映射；root Channel绑定的 worker是 coordinator。
- 三块职责：A `SubmitOwnerStripeStart` 接收 own Tile并启动 worker；B 为每 stripe/peer 调 `SubmitPeerWorkerStripe`；C `SubmitOwnerStripeFinish` 等所有 worker并归还 root Slot。
- Buffer：own Tile Slot既复制到本地用户 Buffer，也作为所有 peer Write 的只读源；收到的 peer Tile写入按其 owner 分开的 Slot。
- Notify：launch、window start/done、Channel DATA/consumed、final done全部在此汇合。
- 保证：下一 batch 前当前 batch 的 peer worker均完成；coordinator 和全部 peer worker最终都被 main 等待。

### 10. `SubmitBidirectionalPeerExchange`

- 位置：文件：`op_kernel_aicpu/exec_op.cc`；行号：L438-L473。
- 调用者：`SubmitPeerExchange`，条件是本端 own Tile与对端 peer Tile都有效。
- 参数含义：`ownTile` 是本 rank owner Tile；`peerTile` 是远端 rank owner Tile；两者在同一 stripe，window 相同。
- 数据：先把 own Slot主动写入对端对应 owner Slot；再等对端 DATA，复制 peer Slot 到用户最终偏移。
- Notify：双方都先 Record DATA 再 Wait DATA，避免双方以跨 rank Wait 开头；随后 Record peer consumed 并 Wait own consumed。
- 前置/保证：Fence 位于 Write 与 DATA Record 之间；函数返回前已排入 own 远端 Slot被消费的依赖。
- 尾块：若任一 Tile无效，不进入本函数，改走 `SubmitSendTile`/`SubmitReceiveTile`/空操作。
