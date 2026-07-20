# 正确性约束、风险与待确认问题

## 正确性约束表

| 约束 | 代码保证位置 | 违反后的结果 | 当前是否明确保证 |
|---|---|---|---|
| 参数中的 rank/root 合法 | `ValidateBroadcastParam`、`ValidateExecutionContext` | Channel查找越界或错误角色 | [确认] Host/AICPU 双重校验 |
| `count * elementSize` 不溢出 | 同上，先检查 `count > UINT64_MAX/elementSize` | 字节数回绕、越界搬运 | [确认] |
| Context 布局与 rankSize 匹配 | `ValidateExecutionContext` 检查 v3、rankSize | 错误反序列化或资源错配 | [确认] |
| Channel 数等于 `rankSize-1` | `ValidateExecutionContext` | peer缺失、Wait/Post不对称 | [确认] 数量；Distributed 还检查顺序/唯一 worker |
| Tile用户偏移和尾长度不越界 | `MakeTileDesc` 的溢出检查与 `min` | 读写用户 Buffer越界 | [确认] |
| Slot地址不越界 | `GetSlotAddress` 检查乘法、offset、bytes | HCCL Buffer越界 | [确认] |
| Slot覆盖前所有读取完成 | Direct最终 ACK；Distributed peer consumed；`SubmitOwnerStripeFinish` 后 root最终 consumed | 旧 Tile被覆盖，接收数据损坏 | [确认] 代码建立了 Notify依赖；底层完成语义见待确认项 |
| root本地 Direct源槽复用安全 | root收到每个 peer最终 ACK后才进入下一块 LocalCopy | Write仍读旧源时被覆盖 | [确认] 任务顺序明确 |
| Parallel Direct共享源只写一次 | `ExecuteParallelDirectRoot` 先 LocalCopy，再 `StartAllWorkers`；main最后等全部 worker | worker读写竞争 | [确认] worker只读共享 `localBuffer` |
| Notify复用前上一轮已消费 | Distributed按两 window分批，C阶段/最终 consumed结束后进入下一批 | 信号串轮次 | [确认] 正常成功路径 |
| Window/Slot/peer信号不混用 | window进入ID；peer由 ChannelHandle隔离；Slot含 owner维度 | 错 Tile唤醒 | [确认] peer/window；Channel Notify无独立 owner ID但每 Channel每 window同一时刻只处理一个对应 owner Tile |
| 不同 worker完成信号不混用 | `WorkerDoneNotifyIndex` 含 workerIndex、window | coordinator误判全部完成 | [确认] |
| 每个 Wait存在对应 Post | `03-signal-path.md` 配对；尾部由 `SubmitPeerExchange` 四分支对称选择 | 死锁/Checker卡住 | [确认] 正常路径静态配对 |
| 双向交换不会双方先 Wait | `SubmitBidirectionalPeerExchange` 双方先 Write/Fence/Record DATA | 循环等待 | [确认] |
| 无效尾 Tile不制造 Channel悬空 Wait | `SubmitPeerExchange`：双向/仅发/仅收/空操作 | 尾 stripe死锁 | [确认] |
| 无效尾 Tile仍产生 worker done | `SubmitOwnerStripeStart` 总是启动peer；`SubmitPeerWorkerStripe` 总是最终Record done | coordinator悬空 Wait | [确认] |
| worker退出前完成未决编排 | 每个 worker最后 `NotifyWorkerDone`，main `WaitAllWorkers` | 主线程过早通知 Host | [确认] 并行/Distributed；Direct无worker |
| Write先于DATA通知 | 每个 Write 后 `HcommChannelFenceOnThread`，再Record DATA | 接收方看到未完成数据 | [确认] 调用顺序；Fence硬件语义需外部确认 |
| 多线程不同时写同一HCCL Slot | Distributed Slot按owner分开；own Slot一个coordinator接收、多个worker只读 | 本地Buffer数据竞争 | [确认] 正常单次调用 |
| 多线程不同时写同一用户区域 | 每个 Tile偏移唯一；每个 rank对某 Tile只有一个接收worker/本地复制 | 用户 Buffer数据竞争 | [确认] 单次调用内 |
| 同一 Context不被多个并发调用混用 | 固定Tag、共享threads/channels/localBuffer，无调用ID | 不同调用Notify和Buffer互相污染 | [待确认] 代码未见锁或每调用命名空间 |
| 错误后资源/Notify可再次使用 | 错误路径多处直接返回，无Reset/Drain | 下一调用可能继承未消费信号/任务 | [待确认] |
| 通信域销毁时释放资源 | 当前仓库无Release/Destroy | 资源泄漏或由外部统一回收 | [待确认] |

## 已确认的实现行为

1. [确认] 算法分三档：`<64KiB` 串行 Direct；`64KiB..1MiB` 在资源满足时 Parallel Direct；`>1MiB` 在资源满足时 Distributed，否则分块 Direct。
2. [确认] 资源 Tag不包含 root/count/buf；同一通信域后续调用复用全连接 Channel、Thread 和 HCCL Buffer描述，root可变化。
3. [确认] 16 rank时每个本地 rank申请到其他15个 rank的 Channel、15个worker和1个AICPU主线程；代码没有按Server物理位置选不同算法分支。
4. [确认] Distributed 使用跳过 root 的15个owner，连续 Tile按 block-cyclic轮转分配；每个非 root rank先从root接收own Tile，再与其他非 root rank做pairwise fanout/互换。
5. [确认] 跨 rank只有发送方主动 `HcommWriteOnThread`；接收方等待 DATA后从本地 HCCL Slot复制到最终用户 Buffer。
6. [确认] `pipelineDepth=2` 对应两个Window；每个Window中每owner一个Slot；Channel DATA/consumed和Thread start/done都按Window隔离。
7. [确认] 尾 Tile长度按剩余字节数计算；在4MiB Tile条件下，400MiB+4B会生成独立4B Tile并执行完整root->owner->peer通信。
8. [确认] Parallel Direct只并行root发送侧；非 root继续使用原串行Direct接收协议。
9. [确认] Host/AICPU完成握手表示任务编排进入用户stream，不存在Host端 `aclrtSynchronizeStream`。
10. [确认] 仓库没有显式CCU实现、Notify Reset、全局Barrier、Channel Drain或资源释放代码。

## 有代码依据的潜在风险

以下只陈述风险，不给出修改方案。

### 1. 同一通信域并发调用共享全部协议状态

[推测] `kResourceTag` 固定，热路径复用相同 `AlgResourceCtx`；HCCL Buffer Slot、Thread和Channel Notify都没有调用序号维度。若同一 rank在多个用户stream上并发执行同一通信域的 Broadcast，两个调用可能同时使用相同Slot和Notify。当前仓库没有互斥、序列化证明或“上层保证同一通信域不并发”的接口契约。

依据：文件：`op_host/broadcast.cc`；符号：`kResourceTag`、`HcclBroadcast` 热路径；文件：`exec_op.cc`；符号：`GetSlotAddress`、各 Notify index函数。

### 2. 错误路径不清理 BatchMode 与未消费信号

[确认] `HcclAICPUKernel` 在 Host Wait或 `ExecOp` 失败时直接 `return 1`，不会执行后面的Device->Host完成Record和 `HcommBatchModeEnd`。各执行函数在任一HCOMM调用失败时也立即返回，没有Notify Reset/Drain。

[推测] 如果运行时仍保留已排队的部分任务，复用相同Context的后续调用可能遇到旧信号或旧Buffer任务；具体后果取决于HCOMM错误恢复语义。

### 3. Link选择是“首个可用”，不是显式带宽最优

[确认] `QueryBestLinkToPeer` 按layer 0..2返回首个非 `COMM_PROTOCOL_RESERVED` Link。代码不比较带宽、协议优先级或Server边界。

[推测] 在题目的Full-Mesh + Clos拓扑上，它能选择真实Link，但是否总是最优Link无法由代码确认。

### 4. 资源创建中途失败的释放路径不可见

[确认] 冷路径依次申请CPU Thread、AICPU Threads、Channels和两个Engine Context；宏在失败时直接返回。仓库没有对应Release调用。

[待确认] HCCL API是否对失败的资源申请自动回滚，或者通信域销毁时统一清理。

### 5. 底层完成语义是关键外部假设

[确认] 代码使用 `HcommChannelFenceOnThread` 把Write与DATA Record排序，并用接收方LocalCopy后的consumed允许复用。

[待确认] 当前仓库没有HCOMM实现，无法仅凭调用点证明Fence何时保证远端可见、LocalCopy任务何时释放源Slot、Notify是否为计数语义。若这些语义不同于代码假设，Slot安全性会受影响。

### 6. 超过16 rank的行为会退化

[确认] worker最多15个。`rankSize>16` 时Channel共享worker，Distributed的“每peer唯一worker”检查失败并回退Direct；Parallel Direct允许Channel共享worker。正确性保底仍存在，但当前题目外规模的性能和Notify压力没有测试依据。

## 当前无法确认的问题

| 问题 | 为什么当前无法确认 | 需要的额外信息 |
|---|---|---|
| HCOMM Write/Fence/LocalCopy的精确完成点是什么？ | 仓库只有头文件调用点，没有实现和正式语义说明 | CANN 9.1 HCOMM API文档、头文件注释或运行时任务图 |
| 同一Notify ID连续两次 ACK/consumed是否严格计数、不合并？ | 当前代码依赖一轮内同ID两次Post/Wait | Notify底层语义文档或Checker/硬件trace |
| ChannelHandle双向同时Write的保证是什么？ | owner-peer双方在同一逻辑Channel上对称Write | HCOMM Channel双向并发契约 |
| CCU实际执行哪些任务？ | 仓库无CCU入口，只能看到AICPU提交HCOMM原语 | CANN运行时架构文档或CCU任务dump |
| `HcclBroadcast`返回时数据是否已完成还是仅已入stream？ | 无Host同步；Thread API可能是任务编排接口 | Hccl自定义算子Host API的异步契约、实际stream trace |
| Engine Context/Thread/Channel由谁释放？ | 仓库无销毁入口 | HcclComm销毁流程或框架生命周期代码 |
| 同一comm并发Broadcast是否被上层禁止？ | 本实现没有调用级隔离 | HCCL并发调用契约和测试用例 |
| 真实HCCL Buffer大小与最终Tile大小是多少？ | size由运行时 `HcclGetHcclBuffer` 返回 | 16-rank运行日志中的 `localBuffer` 和plan日志 |
| 2 Server拓扑上每对rank最终选择的Link是什么？ | rank graph由运行时提供 | `HcclRankGraphGetLinks` dump或Channel诊断日志 |
| 当前三档阈值和4MiB Tile是否性能最优？ | 代码给出参数但无本次工作树对应的完整性能结果 | 官方/仿真环境多轮带宽数据；这不影响本文确认的功能行为 |

## 一致性复核结论

- [确认] `02-resource-and-data-path.md`、`03-signal-path.md` 和 `05-execution-trace.md` 使用同一组公式：Window=`stripe%D`，Slot=`window*ownerCount+owner`，DATA=`window`，consumed=`D+window`。
- [确认] 所有搬运长度统一为Byte；`count` 只在计算 `totalBytes` 前是元素数。
- [确认] 正常路径的每个显式Channel/Thread Wait都已找到协议上唯一对应的Record角色。
- [确认] 文档没有把运行时Buffer大小、CCU职责、HCOMM底层完成语义或并发调用安全性写成已确认事实。
