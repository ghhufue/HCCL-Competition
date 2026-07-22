# HCCL Broadcast：跨 Batch Write Window 修改 Prompt

> 将下面内容完整交给代码 AI。请让 AI 直接检查并修改当前仓库，不要只给概念方案。

---

## Prompt 正文

你现在需要修改我的 HCCL Broadcast 竞赛实现，解决 Owner Write 阶段“每个 Batch 都被本 Die 最慢 peer 锁步”的问题。

仓库与目标分支：

```text
仓库：https://github.com/ghhufue/HCCL-Competition
分支：codex/hccl-broadcast-final
目标目录：Hccl_Broadcast_Final
```

请先阅读当前分支的实际代码、参数传递链、CCU Event/Channel 使用方式和已有测试脚本，再实施修改。不要根据下面的伪代码猜测真实 API、类型或文件位置；若仓库当前结构与描述不同，以实际源码为准，并在修改说明中指出差异。

## 一、问题背景

当前 Owner Write 的流程大致为：

```text
提交 Batch 0 到本 Die 的所有 peer
EventWait：等待 Batch 0 的所有 peer 完成

提交 Batch 1 到本 Die 的所有 peer
EventWait：等待 Batch 1 的所有 peer 完成
```

相关逻辑重点检查：

```text
Hccl_Broadcast_Final/op_kernel_ccu/ccu_kernel.cc
SubmitOwnerWrites(...)
RunPushFirst(...)
RunPushLoop(...)
RunPushTail(...)
```

此前观察到 `SubmitOwnerWrites()` 在向本 Die 的所有 peer 提交 Write 后，立即使用组合 completion mask 调用 `EventWait`。因此当前等价于：

```text
pushWindowDepth = 1
```

假设一个 Die 负责多个 peer，其中某个 peer 的 Batch 0 较慢，其他 peer 即使已经完成，也不能提前排入 Batch 1。这造成：

- peer 之间被强制锁步；
- 快 Channel 空闲等待慢 Channel；
- 12/16 rank 时 head-of-line blocking 更明显；
- CCU 队列和 Channel 的在途深度没有充分利用；
- 后续 Tile 即使已经 ready，也无法及时提交 Push。

## 二、本次修改目标

把 Owner Write 从“每批提交后立即等待”改为“有限深度的跨 Batch Write Window”：

```text
Depth = 1：
Submit B0 → Wait B0
Submit B1 → Wait B1

Depth = 2：
Submit B0
Submit B1
Wait/回收 B0
Wait/回收 B1
继续提交 B2、B3

Depth = 4：
最多允许4个Batch同时在途
窗口满后再等待并回收相应完成事件
```

目标不是无限提交，也不是删除完成等待，而是：

1. 将 Write 提交和完成等待拆开；
2. 最多保留 `pushWindowDepth` 个在途 Batch；
3. 窗口满时安全等待和回收；
4. Push 结束前 drain 本 Die 提交的所有 Write；
5. 保留 `depth=1` 作为完全等价的回退和对照组；
6. 首版默认使用 `depth=2`，稳定后才测试 `depth=4`。

## 三、必须保持不变的算法约束

本次只修改 Owner Push 的跨 Batch 提交/等待方式，不要破坏已经确定的数据通路：

1. Phase 0～2 的 Seed 阶段仍然是 receiver pull。
2. 从 root rank 到 owner rank 的数据传输始终由 owner 主动 Pull，不能改回 root Push。
3. 总数据按 rank 数量划分为 N 个连续 owner block，不能改回 block-cyclic。
4. 每个 owner 只负责自己连续 block 的 Seed Pull 和后续扩散。
5. AllGather/扩散阶段仍使用 owner write，由 owner 主动写其他 rank。
6. receiver 不发送逐 Tile READY，也不返回逐 Tile ACK。
7. 两个 Die 独立向各自负责的 peer 集合推进，不增加逐 Batch 跨 Die barrier。
8. 最终完成协议仍是：

```text
每个Die drain自己提交的全部Write
→ 两个Die汇合
→ OWNER_DONE
→ root汇总所有owner
→ GLOBAL_DONE
```

9. 不要改变数据分块、尾块吸收、root owner 特殊处理以及当前正确的数据地址计算。
10. 不要为了实现 Window 而恢复远端 READY/ACK 信号。

## 四、修改前必须确认的技术前提

在写代码前，必须从当前 HCOMM/CCU 头文件、仓库内其他 Kernel 或官方参考实现中确认以下语义，并把依据写入修改报告：

### 1. 同一 Channel 是否保证 Write 顺序

需要确认：同一个 peer 的 Batch 0、Batch 1 连续提交到同一 Channel 时，硬件/API 是否保证按提交顺序完成或至少按顺序对目标地址生效。

如果不保序，不能仅依靠最终一次完成事件代表之前的 Write。

### 2. Event bit 是否具有计数语义

当前方案可能对同一个 peer event bit 连续产生多个 Write 完成事件，然后连续执行多次 `EventWait(mask)`。

必须确认：

```text
同一个event bit连续Record/完成两次
在Wait之前是否会累计为2次
还是只能保存一个置位状态
```

若 Event 只有布尔置位语义，Batch 0 和 Batch 1 复用同一组 bit 会丢事件，必须给每个 Window Slot 使用独立 Event 或独立 event range。

### 3. EventWait(mask) 的消费语义

确认一次 `EventWait(mask)` 是否会对 mask 中每个 bit 各消费一次完成计数，以及后续能否继续等待下一轮相同 mask。

### 4. Event/Mask 的实际位宽与映射

不要默认 `uint16_t`、`1 << peerRank` 或 rank 编号就是正确映射。检查当前代码如何构造 completion mask，是否按 channel index、local peer index、rank id 或固定 event offset 映射。

### 5. CCU 队列允许的在途 Write 数

确认 depth 2/4 是否可能超过：

- 单 Channel 在途 Write 上限；
- 单 Kernel/LoopGroup 指令或任务上限；
- Event 资源数量；
- completion mask 能表达的事件数量；
- 当前硬件或模拟器的队列容量。

如果无法从源码或文档确认，先实现最保守的 `depth=2`，并保留 `depth=1`；不要假设 `depth=4` 一定安全。

## 五、推荐的代码重构

### 1. 拆分“提交”和“等待”

将当前类似下面的单体逻辑：

```cpp
SubmitOwnerWrites(...)
{
    for (peer : localDiePeers) {
        Write(...);
        completionMask |= EventBit(peer);
    }

    EventWait(event, completionMask);
}
```

拆成职责清晰的两个操作，实际函数名按项目风格确定：

```cpp
SubmitOwnerWritesNoWait(batchDesc, eventSlot)
    -> 返回或记录该Batch实际使用的completion信息

WaitOwnerWriteBatch(inflightDesc)
    -> 等待并回收该Batch对应的完成事件
```

要求：

- `SubmitOwnerWritesNoWait` 只负责计算源/目标地址、向目标 peer 提交 Write、记录完成信息；
- 函数内不能再无条件执行全 peer `EventWait`；
- `WaitOwnerWriteBatch` 只等待本 Batch 实际提交的 peer；
- 非 root owner 跳过写回 root 时，不要错误等待一个永远不会产生的 Write 完成事件；
- 如果当前实现通过 `EventRecord` 补齐被跳过的 root peer，先判断这种做法在 Window 化后是否仍必要、是否会导致计数错位；
- completion mask/slot/event 必须与实际 Batch 一一对应，不能让 Batch 之间错误覆盖。

### 2. 增加 Push Window 配置

增加一个配置字段，建议名称：

```cpp
uint32_t pushWindowDepth;
```

要求：

- 支持值至少包括 `1`、`2`，在资源确认安全后支持 `4`；
- `1` 必须与修改前行为等价；
- 默认值先设为 `2`；
- 可通过 Host 配置或环境变量切换，例如：

```text
HCCL_BROADCAST_PUSH_WINDOW_DEPTH=1
HCCL_BROADCAST_PUSH_WINDOW_DEPTH=2
HCCL_BROADCAST_PUSH_WINDOW_DEPTH=4
```

- 非法值必须安全回退或明确报错，不能静默生成错误参数；
- 如果当前参数通过 Host taskArgs 传入 CCU，必须完整打通：

```text
Host配置
→ 算法/Chunk描述
→ taskArgs打包
→ CCU LoadArg
→ BroadcastContext
→ Push调度逻辑
```

- Host 端参数顺序、类型、宽度必须与 CCU 端严格一致；
- 补充必要的日志，至少能确认实际生效的 window depth。

### 3. 建立明确的 Inflight Window 状态

不要只用一个模糊的计数器。为每个在途 Batch/Window Slot 保存足够的信息，例如：

```cpp
struct InflightPush {
    bool valid;
    uint32_t slotId;
    uint64_t offsetBytes;
    uint64_t bytes;
    CompletionMask completionMask;
    EventHandleOrOffset event;
};
```

实际类型根据 CCU DSL 的限制调整。如果设备端不能使用普通 C++ 容器或动态结构，就使用固定深度的静态 Slot、展开代码或 LoopGroup 支持的 event offset。

必须区分：

```text
ready：源数据已经可读
submitted：Write已经提交，但可能未完成
completed/reclaimed：完成事件已经等待并回收
```

不能把 `submittedBytes` 当成 `completedBytes`，也不能因为最后一个 Batch 已提交就发送 `OWNER_DONE`。

### 4. Window 调度逻辑

期望语义：

```cpp
while (还有待Push的Batch || 还有inflight Batch) {
    while (还有待Push的Batch && inflightCount < pushWindowDepth) {
        等待当前Batch对应的Seed数据ready;
        构造Batch的源地址、目标地址和bytes;
        SubmitOwnerWritesNoWait(...);
        将Batch登记到inflight window;
    }

    if (inflightCount == pushWindowDepth || 没有更多待提交Batch) {
        等待并回收最老的inflight Batch，或按硬件API允许的方式回收;
    }
}

最终确认inflightCount == 0;
```

如果 CCU DSL 不适合动态 while/ring buffer，可以按固定深度展开：

```text
Depth 1路径：Submit 1次 → Wait 1次
Depth 2路径：Submit Slot0 → Submit Slot1 → Wait Slot0 → Wait Slot1
Depth 4路径：Submit 4次 → 对4个Slot安全回收
```

完整 Window 和不足一个 Window 的尾部必须分别正确处理：

```cpp
fullWindowCount = batchCount / depth;
windowRemainder = batchCount % depth;
```

尾部只等待实际提交的 Batch，不能等待空 Slot。

### 5. First、Loop、Tail 的实施顺序

为了降低风险，分两步实施：

第一步：

```text
RunPushFirst：保留原先提交后等待
RunPushLoop：实现Depth 1/2，确认窗口机制正确
RunPushTail：保留原先提交后等待
```

第二步：

将 First、Loop、Tail 统一抽象成一条 Batch 流，让第一批、常规批和尾批都进入同一个 Window 调度器，消除边界处不必要的等待。

如果当前代码结构已经可以安全地一次统一，不必机械地保留旧函数，但必须先证明 `depth=1` 行为与原版本等价。

## 六、Event 设计的两种可接受方案

根据实际 API 语义二选一，不要混用。

### 方案 A：复用 Event，依赖计数累计

仅当源码/官方示例明确证明以下条件成立时使用：

1. 同一个 event bit 可以累计多次完成；
2. 每次 `EventWait(mask)` 对每个 bit 消费一次；
3. 连续提交不会覆盖先前未消费的完成；
4. 相同 Channel 的完成顺序可用于对应 Batch 顺序。

这时 Depth 2 可以是：

```text
Submit B0，使用peer completion mask
Submit B1，复用同一mask并累计第二次完成
EventWait(mask)  // 消费B0对应的每peer一次完成
EventWait(mask)  // 消费B1对应的每peer一次完成
```

### 方案 B：每个 Window Slot 使用独立 Event

如果 Event 不具备可靠的计数累计语义，必须：

```text
Slot 0 → event/eventOffset 0
Slot 1 → event/eventOffset 1
Slot 2 → event/eventOffset 2
Slot 3 → event/eventOffset 3
```

每次 Slot 回收后才能复用其 Event。需要同步调整 Event 资源申请、编号、参数和 Kernel 上下文。

优先选择有源码或官方实现证据支撑、改动最小且正确性可证明的方案。不能因为方案 A 代码更少就默认 Event 能累计。

## 七、与 Batch Merge 的关系

当前还存在另一个潜在问题：如果代码在开启合并后直接令：

```cpp
readyTiles = pushMergeFactor;
WaitSeedReadyCount(readyTiles);
```

那么它实际是在主动等待凑满 8/16 MiB，而不是“只合并自然积压、没有积压就立即发送”。

但本次修改不要同时重写动态合并算法。原因是需要单独测量 Write Window 的收益和正确性。

本次测试时优先关闭合并：

```text
enablePushBatchMerge = false
```

先完成以下对照：

```text
merge=0, depth=1
merge=0, depth=2
merge=0, depth=4（确认资源安全后）
```

Window 稳定后，再测试：

```text
merge=1, depth=1
merge=1, depth=2
merge=1, depth=4
```

如果检查时发现当前合并语义与设计目标不一致，请在报告中单独记录，并给出后续修改建议；除非它直接阻塞 Window 的正确实现，否则不要在本次补丁中顺手大改。

## 八、完成与安全要求

以下条件必须同时满足：

1. 每个已提交 Write 都有可追踪的完成事件。
2. 每个完成事件只被正确消费一次，不丢失、不串 Batch、不重复等待。
3. Window Slot 只有在对应 Batch 完成并回收后才能复用。
4. Push Kernel 返回前，所有在途 Batch 必须完成并 drain。
5. 两个 Die 各自完成本地 drain 后，才能进入原有 Die 汇合。
6. `OWNER_DONE` 只能在 owner 的两个 Die 都完成所有 Write 后发送。
7. `GLOBAL_DONE` 只能在 root 收齐所有 owner 完成后发送。
8. 不允许通过额外的全局 stream synchronize 或粗粒度 host barrier 掩盖 Event 错误。
9. 不允许无限增加在途请求；必须严格受 `pushWindowDepth` 限制。
10. 不允许为每个 Batch 新建 Host 线程或重新启动独立 Kernel；应在已有长期运行的 Push worker/kernel 内完成调度。

## 九、分阶段实现和验证

### 阶段 1：等价重构

先拆分 Submit/Wait，但设置：

```text
pushWindowDepth = 1
enablePushBatchMerge = false
```

要求：

- checker 结果与原版本一致；
- 所有 case 无超时、无数据错误；
- Write 次数、Batch 次数、等待次数与原版本一致；
- 性能不能出现无法解释的明显回退。

### 阶段 2：Depth 2

开启：

```text
pushWindowDepth = 2
enablePushBatchMerge = false
```

重点验证：

- 两个 Batch 确实先后提交，再执行对应回收；
- 没有 Event 覆盖、丢失或多消费；
- 非 root owner 跳过 root peer 时 mask 不错位；
- 奇数 Batch 数时 remainder 正确；
- 尾块不足 TileSize 时地址和字节数正确；
- 连续多次 Broadcast 不出现第二次/后续调用超时。

### 阶段 3：Depth 4

只有在确认 Event 与队列资源允许后再开启。若出现不稳定，不要用同步兜底掩盖，而应保留默认 depth 2，并报告 depth 4 的资源限制。

### 阶段 4：统一 First/Loop/Tail

在 Depth 2 稳定后，将所有 Batch 统一进入 Window，重新跑完整验证矩阵。

## 十、测试矩阵

至少覆盖：

### Rank 数

```text
2 / 4 / 8 / 12 / 16 rank
```

### Root

```text
root = 0
root = 中间rank
root = 跨Die代表性rank，例如7
root = 最后一个rank
```

### 数据量

```text
4 B
小于1个Tile
恰好1个Tile
1个Tile + 4 B
多个Tile
奇数个Batch
偶数个Batch
大包，例如64/256/400/512 MiB
大包 + 4 B
```

### 配置组合

```text
merge=0, depth=1
merge=0, depth=2
merge=0, depth=4（若安全）
merge=1, depth=1
merge=1, depth=2
merge=1, depth=4（若安全）
```

### 稳定性

对关键 case 连续运行至少 20 次，重点观察：

- 第二次及后续 Broadcast 是否超时；
- Event 是否跨轮次残留；
- Window Slot 是否提前复用；
- 尾批是否等待不存在的事件；
- 双 Die 是否有一侧提前发送完成信号；
- checker 是否全程通过。

## 十一、性能采集要求

至少记录以下指标：

```text
端到端Broadcast时延
首个Push提交时间
Seed全部完成时间
最后一个Push提交时间
最后一个Push完成时间
Batch数量
Write数量
EventWait数量
最大inflight Batch数量
各Die负责的peer数量
```

重点比较：

```text
depth=1 与 depth=2 的 12/16 rank 大包时延
depth=2 与 depth=4 是否仍有收益
窗口化是否拖慢 Seed Pull
是否减少快 Channel 的空闲时间
```

不要只给最终总耗时。需要用日志、trace 或代码级计数证明多个 Batch 曾同时在途，否则无法确认 Window 真正生效。

## 十二、禁止事项

1. 禁止只删除 `EventWait` 后无限提交。
2. 禁止在没有证据时假设 Event 是计数器。
3. 禁止不同在途 Batch 错误复用只能保存单次状态的 Event bit。
4. 禁止等待未实际提交 Write 的 peer event。
5. 禁止把“最后一个 Batch 已提交”当作“所有 Write 已完成”。
6. 禁止更改 Phase 0～2 的 owner pull 数据方向。
7. 禁止改回 block-cyclic。
8. 禁止恢复逐 Tile 远端 READY/ACK。
9. 禁止为了过测试增加全局同步、Stream 同步或逐 Batch 双 Die barrier。
10. 禁止同时大改自然积压合并，导致 Window 收益无法单独验证。
11. 禁止只给伪代码而不修改实际工程。
12. 禁止忽略 Host/CCU 参数打包顺序和宽度一致性。

## 十三、最终交付要求

完成修改后，请输出：

1. 问题根因：指出原代码在哪个循环/函数形成逐 Batch 全 peer barrier。
2. Event 语义核查结论：给出同 Channel 保序、Event 累计和 Wait 消费语义的源码/官方实现依据。
3. 修改文件清单：逐文件说明修改目的。
4. 参数链说明：`pushWindowDepth` 如何从 Host 传入 CCU。
5. 新调度流程：说明 Submit、Inflight、Wait/Reclaim、Final Drain 的状态变化。
6. Event 设计：明确使用“计数复用”还是“每 Slot 独立 Event”，以及选择理由。
7. 正确性证明：说明为何不会丢事件、串 Batch、提前复用 Slot 或提前发送完成信号。
8. 测试结果：列出命令、rank/root/size/config、checker 结果与稳定性次数。
9. 性能对比：至少给出 depth 1/2，以及安全时 depth 4 的数据。
10. 剩余风险：列出未确认的硬件队列、Event 资源或模拟器差异。
11. 提交一个最小、可回退、无无关改动的代码补丁。

如果受限于当前环境无法运行 HCCL VM 或完整性能测试，仍应完成静态核查和安全实现，但必须明确区分：

```text
已经从源码确认的结论
已经实际运行验证的结论
尚需在真实环境验证的假设
```

最终验收标准是：

> `depth=1` 保持原有正确性和行为；`depth=2` 允许至少两个 Owner Push Batch 同时在途，不再每个 Batch 都进行全 peer 锁步，并且在最终 `OWNER_DONE` 前可靠 drain 全部 Write；整个过程不增加远端逐 Tile READY/ACK，也不改变 owner pull、连续分块和 owner write 的既定数据通路。

