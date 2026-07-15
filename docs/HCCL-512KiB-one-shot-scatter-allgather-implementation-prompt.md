# HCCL 512 KiB One-Shot Scatter + AllGather 实现 Prompt

你刚接手这个项目。请先完整阅读当前代码和已有实现分析，再实现一个**仅服务于 16 rank、精确 512 KiB Broadcast** 的独立快速路径。

本任务的第一原则是：**不得改动或参数化现有大包 `DISTRIBUTED_SCATTER_FANOUT` 的核心执行逻辑。** 新算法即使与大包算法存在少量重复，也必须保持独立，避免影响已经验证且大包性能较好的路径。

## 一、开始前必须阅读

至少阅读以下文件：

```text
hccl_broadcast_problem_template/op_kernel_aicpu/exec_op.cc
hccl_broadcast_problem_template/op_host/broadcast.cc
hccl_broadcast_problem_template/include/custom.h
docs/current-implementation/README.md
docs/current-implementation/02-resource-and-data-path.md
docs/current-implementation/03-signal-path.md
docs/current-implementation/05-execution-trace.md
docs/current-implementation/06-invariants-and-questions.md
```

先确认当前分支、Commit ID 和工作区状态。已有未提交文件属于用户，不得覆盖、删除或顺手格式化。

## 二、修改边界

只允许修改：

```text
hccl_broadcast_problem_template/op_kernel_aicpu/exec_op.cc
```

不要修改：

```text
hccl_broadcast_problem_template/op_host/broadcast.cc
hccl_broadcast_problem_template/include/custom.h
```

原因是现有 Host 资源已经足够：

- 16 rank 时已经申请 15 个 Channel worker；
- 每个远端 rank 已经有一条 Channel，并绑定唯一 worker；
- 每条 Channel 已有 4 个 Notify；
- 每个 ThreadHandle 已按静态 `pipelineDepth=2` 资源布局申请 33 个 Notify；
- 本地和远端 HCCL Buffer 地址已经保存在 `AlgResourceCtx`/`ChannelInfo` 中。

不得修改资源 Tag、`ResourceLayoutVersion`、序列化字段、Channel 创建方式、Thread 数量、Notify 数量或 Host 冷/热路径。

如果实现过程中发现现有资源确实不足，不要自行扩大修改范围；停止实现并用代码证据说明阻塞点。

## 三、必须保持完全不变的大包核心

不得修改以下函数的函数体，也不得为了新算法给它们增加布尔参数或模式分支：

```text
MakeTileDesc
GetSlotAddress
SubmitSendTile
SubmitReceiveTileData
SubmitReceiveTile
SubmitBidirectionalPeerExchange
ExecuteDistributedRoot
SubmitPeerExchange
SubmitOwnerStripeStart
SubmitOwnerStripeFinish
SubmitPeerWorkerStripe
ExecuteDistributedOwner
ExecuteDistributedScatterFanout
```

不得修改以下大包配置：

```text
AlgorithmConfig::kPreferredTileBytes
AlgorithmConfig::kMinimumTileBytes
AlgorithmConfig::kPreferredPipelineDepth
```

不得改变现有算法的阈值和选择结果，唯一例外是把“16 rank 且精确 512 KiB”从当前 `PARALLEL_DIRECT_FANOUT` 中截取出来交给新算法。

以下调用在新算法之外必须保持原行为：

- 小于 512 KiB；
- 大于 512 KiB；
- 512 KiB 但不是 16 rank；
- 新算法资源检查失败；
- 所有大于 1 MiB 的大包。

## 四、新算法的精确启用条件

新增独立算法枚举，建议命名：

```cpp
AlgorithmKind::ONE_SHOT_SCATTER_ALLGATHER_512K
```

只有同时满足下列条件才能选择它：

```text
totalBytes == 512 * 1024
rankSize == 16
ownerCount == 15
本 rank 有 15 个 Channel 和 15 个唯一 worker
threads.size() == 16，threads[0] 是 aicpuThread
local HCCL Buffer >= totalBytes
每个 remote HCCL Buffer >= totalBytes
Thread/Channel Notify 数量满足本 Prompt 的固定布局
```

资源检查必须放在新的独立函数中，例如：

```cpp
HasOneShot512KResources(...)
```

资源检查失败时打印一条 `HCCL_WARNING`，然后**继续执行原有选择逻辑**，使 512 KiB 安全回退到当前 `PARALLEL_DIRECT_FANOUT`。不得因为快速路径资源不足直接报错。

不要把新路径扩展到一个范围，例如 `64 KiB ~ 1 MiB`。第一版只优化精确的 524288 字节，便于隔离正确性和性能变化。

## 五、算法目标

新算法固定为：

```text
algorithm pipelineDepth = 1
stripeCount = 1
ownerCount = rankSize - 1 = 15
root 不作为 owner
不创建 root 专属 slice
HCCL Buffer 按最终数据偏移直接组装
一次调用内每个 slice 位置只写一次，不复用 Window/Slot
非 root 收齐数据后只执行一次整包 LocalCopy 到用户 Buffer
```

不要把这里的 `pipelineDepth=1` 用来重新解释 Host 静态资源布局。它只是新算法的语义：单 stripe、无流水、无窗口复用。Host 资源仍然是按 `resource.pipelineDepth == 2` 申请的。

### Slice 计算

owner 序号继续跳过 root：

```text
ownerIndex 0..14
ownerRank = OwnerIndexToRank(ownerIndex, root)
```

按元素大小对齐计算 slice stride：

```text
rawSliceBytes = ceil(totalBytes / ownerCount)
sliceBytes = AlignUp(rawSliceBytes, elementSize)
sliceOffset(owner) = ownerIndex * sliceBytes
actualSliceBytes(owner) = min(sliceBytes, totalBytes - sliceOffset)
```

所有乘法、加法和向上对齐都要检查溢出。`sliceOffset >= totalBytes` 时 slice 无效；但在 16 rank、512 KiB 的目标条件下 15 个 owner 都应有有效 slice。

以 FP32 为例：

```text
totalBytes = 524288
ownerCount = 15
sliceBytes = 34956
owner 0..13 各 34956 字节
owner 14 为 34904 字节
```

为新算法定义独立的纯描述结构，例如：

```cpp
struct OneShotSliceDesc {
    uint32_t ownerIndex;
    uint32_t ownerRank;
    uint64_t offset;
    uint64_t bytes;
    bool valid;
};
```

新增独立的 `MakeOneShotSliceDesc` 和边界检查函数。不要复用或修改大包的 `TileDesc`、`MakeTileDesc`、`GetSlotAddress`，因为大包地址是 `[window][owner][tile]` 槽位语义，而新路径地址必须是最终整包偏移：

```text
local address  = resource.localBuffer.addr + slice.offset
remote address = channel.remoteCclMem.addr + slice.offset
```

## 六、数据路径

### Root rank

1. 在 AICPU main thread 上执行一次整包：

```text
inputPtr[0, totalBytes) -> local HCCL Buffer[0, totalBytes)
```

2. 显式启动 15 个 worker。
3. 每个 worker 只负责与其 Channel 对应的一个非 root owner：

```text
root HCCL Buffer[slice.offset, slice.bytes)
    -> owner remote HCCL Buffer[slice.offset, slice.bytes)
    -> DATA_READY
```

4. root 的该 worker 随后等待同一个 owner 在完成 AllGather 和最终整包 LocalCopy 后返回 `FINAL_ACK`。
5. 收到 `FINAL_ACK` 后，该 worker 向 root main thread 发送一次最终 worker-done。
6. root main thread 最后统一等待 15 个 worker-done，然后返回。

root 不发送完整的 512 KiB 给每个 rank；root 总发送量必须正好等于 512 KiB。

遵循当前 Broadcast 的 root Buffer 语义。不要擅自给 root 增加 `inputPtr -> outputPtr` 拷贝；非 root 才需要最后的整包 HCCL Buffer -> `outputPtr`。

### 每个非 root owner rank

每个非 root rank 同时是：

- 自己 slice 的 owner；
- 其他 14 个 owner slice 的接收方。

使用连接 root 的 Channel worker 作为 coordinator；其余 14 个非 root Channel worker 作为 peer worker。

执行顺序：

1. AICPU main thread 显式启动全部 15 个 worker。
2. coordinator 在 root Channel 上等待自己 slice 的 `DATA_READY`。
3. 收到自己的 slice 后，coordinator 向 14 个 peer worker 各发送一次启动 Notify。
4. 每个 peer worker 在独占的 peer Channel 上严格按以下顺序执行：

```text
等待 coordinator START
把本 rank 拥有的 slice 写到 peer 的相同 slice.offset
发布 DATA_READY
等待对端 owner 写来的 DATA_READY
向 coordinator 发布 PEER_DONE
```

必须先 `Write + DATA_READY`，再等待对端 `DATA_READY`，避免两个 rank 都以跨 rank Wait 开头形成环。

不同 sender 只写自己的 `slice.offset`，因此 15 个 slice 不重叠，不需要锁，也不需要 per-slice LocalCopy。

5. coordinator 等待 14 个 `PEER_DONE`。此时本地 HCCL Buffer 已经包含完整 512 KiB。
6. coordinator 只执行一次：

```text
local HCCL Buffer[0, totalBytes) -> outputPtr[0, totalBytes)
```

7. 整包 LocalCopy 排队完成后，coordinator 在 root Channel 上发布 `FINAL_ACK`。
8. coordinator 向本 rank AICPU main thread 发布一次最终完成 Notify。
9. 非 root main thread 只等待 coordinator 的这个最终完成 Notify，然后返回。因为 coordinator 已经等待全部 14 个 peer worker，完成关系已经传递到 coordinator。

不要让非 root main thread 再等待全部 15 个 worker，否则必须额外产生 14 个冗余 worker-to-main Notify。新路径应有自己的最终等待函数，不要强行复用 `WaitAllWorkers`。

### 非 root 的任务提交顺序

`Hcomm*OnThread` 在这里是在不同 ThreadHandle 上编排任务，不要把伪代码误写成 CPU 阻塞式阶段。为了避免真实硬件在 BatchMode 结束前提前下发 SQE 后产生反压死锁，非 root 必须按以下顺序向各线程提交任务：

```text
A. 向 coordinator 提交：等待 root DATA_READY，并向 14 个 peer worker Record START
B. 向 14 个 peer worker 提交：Wait START -> Write/Signal -> Wait peer DATA -> Record PEER_DONE
C. 再向 coordinator 提交：等待 14 个 PEER_DONE -> 整包 LocalCopy -> FINAL_ACK -> coordinator done
D. 最后在 main thread 提交一次 coordinator done Wait
```

不得先向 coordinator 连续提交全部 14 个 `PEER_DONE` Wait，再晚很久才提交能够释放这些 Wait 的 peer worker Record。虽然新路径只有一个 stripe，仍要保持生产者任务及时进入对应 worker 流。

## 七、同步和 Notify 布局

新算法可以复用现有静态 Notify 资源，但必须定义独立、带 `OneShot`/`Medium` 前缀的语义函数或常量，不能调用依赖大包 `plan.pipelineDepth` 的 Notify 公式。

建议使用：

### Channel Notify

```text
Notify 0: ONE_SHOT_DATA_READY
Notify 2: ONE_SHOT_FINAL_ACK，仅 root <-> owner Channel 使用
Notify 1、3: 新路径不使用
```

同一 peer Channel 的两个方向可以像当前 `SubmitBidirectionalPeerExchange` 一样使用相同 DATA index；Channel Notify 的方向由通信端点区分。

由于本次调用中每个 slice offset 只写一次，因此：

- 不发送初始 SLOT_CONSUMED；
- 不等待 SLOT_CONSUMED 才允许首次 Write；
- 不发送 per-slice consumed ACK；
- 不执行 Window 复用协议。

`FINAL_ACK` 不是 Window consumed。它表示该 owner 已经收齐全部 slice、完成整包 LocalCopy，root 可以结束本次调用并允许后续调用复用资源。

### Thread Notify

沿用静态资源布局的安全索引：

```text
main -> 所有 worker 的一次性 launch:
    WorkerLaunchNotifyIndex(resource.workerCount, resource.pipelineDepth)
    16 rank 时为 2 * (15 + 1) = 32

coordinator -> peer worker START:
    index 0，目标 ThreadHandle 各不相同

peer worker -> coordinator PEER_DONE:
    使用与 workerIndex 一一对应的 index 1..15

worker/coordinator -> main 最终完成:
    使用发送 worker 的 workerIndex 1..15
```

注意：Notify 资源属于目标 ThreadHandle，因此相同数值出现在不同目标线程上不是同一个 Notify。

必须新增新路径专用的启动函数，例如：

```cpp
StartOneShotWorkers(...)
```

它必须用 `resource.pipelineDepth` 计算静态 launch index 32。不要调用现有 `StartAllWorkers(resource, plan)` 并传入算法 `pipelineDepth=1`，否则会计算出 index 16，混淆静态资源布局和算法流水深度。

每一个 Wait 都必须能找到唯一的 Record，每一个 Record 都必须被消费。重复调用 Broadcast 时不能残留上一次的 Notify。

“阶段结束只做一次统一同步”是指：

- 不增加 Scatter 全局 Barrier；owner 收到自己的 seed 后立即开始 fanout；
- 不做每个 slice 的 consumed/reuse 同步；
- 每个 rank 的 main thread 最终只进行一次完成汇聚。

这不等于删除必要的 `DATA_READY` Wait 或 coordinator 对 14 个 `PEER_DONE` 的等待。

## 八、尽量融合 Write + DATA Notify

先检查当前实际构建环境的 CANN/HCOMM 头文件，确认是否存在“Write 完成后直接发布远端 Notify”的 OnThread API，并核对：

- 精确函数签名；
- Notify 是本地还是远端语义；
- 是否保证 DATA Notify 不会早于数据写完成；
- 是否受当前 Channel 类型支持；
- HCCL-VM CheckerV3 是否识别该 API。

不得根据名字猜测或发明 API。

为新路径封装独立函数，例如：

```cpp
SubmitOneShotWriteAndSignal(...)
```

如果确认存在且 Checker/运行环境支持融合 API，则只在新路径中使用。否则安全回退为：

```text
HcommWriteOnThread
HcommChannelFenceOnThread
HcommChannelNotifyRecordOnThread(DATA_READY)
```

在没有官方接口语义证据和验证结果时，不得为了性能删除 Fence。不要修改大包当前的 `Write + Fence + DATA_READY` 序列。

## 九、建议新增的独立代码结构

命名可以调整，但职责必须独立：

```text
HasOneShot512KResources
MakeOneShotSliceDesc
GetOneShotSliceAddress
StartOneShotWorkers
SubmitOneShotWriteAndSignal
NotifyOneShotPeerDone
WaitOneShotPeerWorkers
ExecuteOneShot512KRoot
ExecuteOneShot512KOwner
ExecuteOneShot512KScatterAllGather
```

允许安全复用的现有纯工具：

```text
CheckHcommRet
OffsetPtr
DivideRoundUp
OwnerIndexToRank
RankToOwnerIndex
FindChannelByRemoteRank
ValidateExecutionContext
NotifyWorkerDone（语义匹配时）
```

不要把新路径塞进 `ExecuteDistributedRoot/Owner`，不要给大包函数添加以下一类参数：

```text
oneShot
skipConsumed
singleStripe
finalWholeCopy
fuseNotify
```

这种参数化会把大包已经验证的 Window/Slot 协议变成多种分支组合，属于本任务明确禁止的改动。

## 十、算法选择的接入方式

只允许对公共调度位置做最小接入：

1. `AlgorithmKind` 新增枚举；
2. `AlgorithmName` 新增名称；
3. `BuildExecutionPlan` 在当前 `PARALLEL_DIRECT_FANOUT` 判断之前加入精确 512 KiB/16-rank 检查；
4. `ExecOp` 的 switch 新增新算法 case；
5. 新增完全独立的新路径函数。

新路径计划应设置：

```text
ownerCount = 15
pipelineDepth = 1
stripeCount = 1
tileBytes 或独立字段 = 对齐后的 slice stride
```

但是任何 Thread Notify 静态索引仍然必须根据 `resource.pipelineDepth` 计算。

在代码评审时，除上述公共接入点外，大包相关 diff 应为零。

## 十一、必须保持的正确性约束

1. root 可以是任意 rank，至少验证 `root=0` 和 `root=7`。
2. owner 映射必须跳过 root，不能假设 root 恒为 0。
3. 最后一个 slice 必须使用真实尾长，不能越过 `totalBytes`。
4. 每个远端写地址必须检查 `offset + bytes <= remoteCclMem.size`。
5. 非 root 必须收到 14 个 peer slice 加 root seed 后，才能整包 LocalCopy。
6. root 必须等 15 个 owner 的 `FINAL_ACK`，不能提前返回并复用 HCCL Buffer。
7. worker 所有通信任务必须通过显式 `main -> worker` launch 边从 AICPU main stream 可达。
8. peer 双方必须先发送再等待，不能构造跨 rank Wait 环。
9. 本次调用的所有 Notify 必须配对消费，支持同一通信域连续多次 Broadcast。
10. 错误路径不得留下会导致下一次调用误触发的半套协议；若任务提交中途失败，记录准确 API 和 rank/peer 信息。
11. 不要在热路径加入 `printf`、`snprintf` 或逐 slice 日志。保留一条计划级 `HCCL_INFO` 即可。

## 十二、验证要求

先构建，再做 Checker 和运行验证；任何一步失败都要先定位原因，不能通过扩大超时掩盖同步错误。

### A. 静态检查

```text
git diff --check
```

确认：

- 只修改 `op_kernel_aicpu/exec_op.cc`；
- 上文列出的大包核心函数体没有 diff；
- Host、`custom.h`、资源 Tag、布局版本没有变化；
- 每个 Wait/Record 可以逐项配对；
- 没有地址溢出和尾 slice 越界。

### B. 512 KiB 目标正确性

在 16 rank 竞赛拓扑至少验证：

```text
512 KiB, root=0
512 KiB, root=7
```

CheckerV3 必须成功，所有 rank 输出必须与 root 输入完全一致。

再在同一个通信域连续执行多次，至少覆盖：

```text
root=0 -> root=7 -> root=0
```

用于发现残留 Notify、固定 root 假设和资源复用问题。

### C. 精确门控和回退

至少验证：

```text
512 KiB - 4B：仍走原路径并正确
512 KiB + 4B：仍走原路径并正确
512 KiB、非 16 rank：仍走原路径并正确
```

日志中只有 16 rank、精确 524288 字节选择新算法。

### D. 大包零回归

至少运行：

```text
128 MiB, root=0
128 MiB, root=7
一个非整齐尾块的大包，例如 128 MiB + 4B
```

必须继续选择 `DISTRIBUTED_SCATTER_FANOUT`，Checker 和数据正确性通过。不要默认反复运行 16-rank 512 MiB 本地测试，避免环境 OOM；只有机器资源明确足够时再执行。

### E. 性能 A/B

使用热资源重复测量，排除首次 Engine Context 创建开销。记录当前基线和新路径的多次结果，至少报告：

```text
最小值
中位数
是否稳定选择新算法
是否存在偶发超时或长尾
```

不要只报告一次最好成绩，也不要预设一定能达到 35 us。如果新路径没有稳定优于当前约 69 us 的 512 KiB 基线，保留代码但不要擅自扩大启用范围，并明确说明瓶颈更可能来自控制延迟或第二跳开销。

## 十三、完成后的回复格式

完成后说明：

1. 实际修改了哪些文件；
2. 新算法的函数结构和精确启用条件；
3. root、owner、peer worker 的真实数据/信号路径；
4. 是否找到了可用的融合 Write + Notify API；如果没有，使用了什么安全序列；
5. Checker 和正确性测试结果；
6. 512 KiB 性能 A/B 结果；
7. 大包回归结果；
8. 用 `git diff` 明确确认大包核心函数体未修改；
9. 尚未确认的风险或环境限制。

不要自动提交代码，除非用户明确要求 commit。
