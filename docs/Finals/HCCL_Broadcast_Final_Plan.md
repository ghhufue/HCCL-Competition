# HCCL Broadcast 决赛详细解决方案

> 基于仓库 `ghhufue/HCCL-Competition` 2026-07-20 最新 `main`，结合初赛实现、决赛题面、三种拓扑和 HCOMM CCU 官方开发资料整理。

## 1. 最终结论

决赛建议不要把初赛的 AICPU 多线程、Window/Slot、`DATA_READY/SLOT_CONSUMED` 协议逐行迁移到 CCU。应保留初赛最有价值的“大包分布式复制”思想，但用 CCU 重新表达为：

1. 每个 rank 到其余每个 rank 只申请一条 Channel；
2. 整个算子只使用一个与用户 stream 绑定的 CCU Thread；
3. 注册两个 CCU Kernel：
   - `DIRECT_FANOUT`：root 并发把整块 Chunk 写给所有非 root，用于 512 KiB 的性能 A/B；
   - `PULL_SCATTER_ALLGATHER`：将 Chunk 按 rank 切成 `N` 片，每个 rank（包括 root）拥有一片；非 root 主动从 root 读取种子片，再主动从其他 owner 读取缺失片，作为三个拓扑的大包主算法，并建议先作为 512 KiB 默认算法；
4. 直接使用用户 `buf` 和 `HcommCcuGetMemToken()` 做 Mem2Mem，不在 HCCL Buffer 中建立初赛那套 Window/Slot；
5. `512 MiB` 切成 `256 MiB + 256 MiB` 两次 Launch；`400 MiB + 4B` 切成 `256 MiB + (144 MiB + 4B)` 两次 Launch；
6. root 作为 TaskArg 动态传入。一个通信域只创建一组 Channel 和两个 Kernel，不能因为 root 不同重复申请同一对端的 Channel；
7. 用 CCU `Event` 汇聚本 rank 的并发 Read；算法阶段由 root 组织两次全局栅栏，只在 root—peer Channel 上顺序复用一个 Phase Notify，不使用非 root 之间逐对的完成信号。

推荐的初始算法矩阵如下，随后只根据真实 A/B 数据调整 512 KiB：

| 拓扑 | rank 数 | 512 KiB | 512 MiB | 400 MiB + 4B |
| --- | ---: | --- | --- | --- |
| `2×8` | 16 | `PULL_SCATTER_ALLGATHER`，与 `DIRECT_FANOUT` A/B | 分 Chunk 的 `PULL_SCATTER_ALLGATHER` | 分 Chunk 的 `PULL_SCATTER_ALLGATHER` |
| `4×1` | 4 | `PULL_SCATTER_ALLGATHER`，与 `DIRECT_FANOUT` A/B | 分 Chunk 的 `PULL_SCATTER_ALLGATHER` | 分 Chunk 的 `PULL_SCATTER_ALLGATHER` |
| `8+4` | 12 | `PULL_SCATTER_ALLGATHER`，与 `DIRECT_FANOUT` A/B | 分 Chunk 的 `PULL_SCATTER_ALLGATHER` | 分 Chunk 的 `PULL_SCATTER_ALLGATHER` |

这套方案的核心优点是：算法天然支持 4、12、16 rank，root 可变，尾块可变；CCU 将初赛大量 Host/AICPU 任务提交开销压缩成少量预注册 Kernel Launch。

---

## 2. 决赛与初赛的本质差异

### 2.1 决赛约束

仓库中的决赛题面确认了以下变化：

- 集群扩大为 `4 Server × 8 NPU`，实际评测使用 `2×8`、`4×1`、`8+4` 三种子拓扑；
- 数据仍为 FP32，只有 `512 KiB`、`512 MiB`、`400 MiB + 4B`；
- 使用 CCU 通信引擎，而不是初赛的 `AICPU + TS`；
- 每个对端只能申请一条 Channel；
- `MAX_DATA_SIZE = 256 MiB`，大包必须分次下发；
- 三种拓扑 rank 数不同，不能硬编码 16 rank。

### 2.2 初赛哪些设计值得保留

| 初赛机制 | 决赛处理 |
| --- | --- |
| `OwnerIndexToRank` / `RankToOwnerIndex` 跳过 root 的映射 | N 分片基线不再需要；`ownerIndex == ownerRank`，仅为以后切换 N-1 分片保留设计空间 |
| root Scatter、非 root owner 扩散 | 保留分布式复制思想；基线改成 N 片和 receiver-initiated Read |
| 每个 peer 一条 Channel | 保留，并严格满足决赛“一对端一 Channel”要求 |
| 集中式 `BuildExecutionPlan`、容量和尾块校验 | 保留到 Host `ExecOp` |
| 尾块只传真实长度 | 保留，所有 offset/bytes 都按真实 Chunk 计算 |
| Wait/Record 必须逐项配对、连续调用不能残留信号 | 保留，是 CCU Kernel 正确性的核心 |

### 2.3 哪些机制必须删除重做

| 初赛机制 | 为什么决赛不应继续使用 |
| --- | --- |
| 1 个 AICPU 主线程 + 15 个 worker | CCU 内的 Event 和并发通信引擎可以表达并发；大量 Thread 反而增加资源和控制复杂度 |
| 每线程 33 个 Notify | 决赛只需一个 Thread，Thread Notify 可为 0；跨 rank 同步放在 Channel Notify 中 |
| HCCL Buffer 的双 Window/Slot | CCU Mem2Mem 可直接访问带 token 的用户 Buffer，不需要先搬入 400 MiB CCL Buffer |
| `SLOT_CONSUMED` 覆盖协议 | 每个 Chunk 的每个目标区间只写一次；Kernel 结束前完成同步，无槽位循环复用 |
| `HcommBatchModeStart/End`、worker coordinator、Fence 编排 | 这是 AICPU/TS 表达方式，决赛应改用预注册 CCU 指令 |
| 4/9/12 MiB Tile 选择器 | CCU 主算法按 owner 对 Chunk 一次切片，`256 MiB` 上限由 Host Chunk 负责 |
| 精确 16-rank、512 KiB One-Shot 门控 | 决赛存在 4/12/16 rank，应统一参数化 |

---

## 3. N 分片 Pull Scatter + AllGather

设本次 Chunk 大小为 `M`，rank 数为 `N`。基线固定切成 `N` 片，每个 rank（包括 root）都是一片的逻辑 owner：

```text
ownerCount  = N
ownerIndex  = ownerRank
sliceStride = AlignUp(CeilDiv(M, N), 4)
sliceOffset = ownerRank * sliceStride
sliceBytes  = sliceOffset >= M ? 0 : min(sliceStride, M - sliceOffset)
```

root 可变只影响“从哪个 Channel 读取”，不影响分片编号。因此基线不需要跳过 root 的 `OwnerIndexToRank/RankToOwnerIndex` 映射。

算法分为两个数据阶段和两个全局栅栏：

1. **Seed Pull**：每个非 root rank 从 root 主动读取自己的 owner slice；root 不提交这些 Mem2Mem 任务。
2. **Seed Barrier**：非 root 完成本地 Read 后通知 root；root 收齐后通知全部非 root 进入下一阶段。此时所有非 root owner 的源 slice 均已就绪。
3. **Pull AllGather**：每个非 root rank 从其余所有 owner 主动读取剩余 `N-1` 片，其中包括 root 的 owner slice。
4. **Completion Barrier**：非 root 完成本地全部 Read 后通知 root；root 收齐后释放所有 rank，防止某个 owner 提前退出而其源 slice 仍被远端读取。

单个非 root rank 一共主动读取 `N` 片：Seed 阶段读取自己的 1 片，AllGather 阶段读取剩余 `N-1` 片。root 本来就持有完整输入，不需要接收任何片。

数据量与压力：

- root 作为远端源被读取约 `2(N-1)M/N`：非 root owner 的种子片各一次，加上 root 自有片被每个非 root 各读一次；
- 每个非 root owner 的源片在第二阶段被另外 `N-2` 个非 root 读取，约为 `(N-2)M/N`；
- 总网络数据量仍约为 `(N-1)M`；
- 相比 Direct Fanout 的 root 侧 `(N-1)M`，root 的物理源流量降到约 `2M`，而且所有 Read 指令由接收 rank 发起，root CCU Thread 不需要集中提交 `N-1` 个 Scatter Write。

`N-1` 分片仍作为后续优化候选：它能把 root 源流量进一步降到约 `M`，但需要跳过 root 的 owner 映射。第一版先用 N 分片建立更简单、对称的正确性基线。

| 对比项 | N 分片基线 | N-1 分片候选 |
| --- | --- | --- |
| owner 数 | `N`，包含 root | `N-1`，跳过 root |
| owner 映射 | `ownerIndex == ownerRank` | 需要动态跳过 root |
| root 自有片 | 有，需分发给所有非 root | 无 |
| root 源流量 | 约 `2(N-1)M/N` | 约 `M` |
| 总网络数据量 | 约 `(N-1)M` | 约 `(N-1)M` |
| 当前定位 | 第一版正确性与性能基线 | root 源端成为瓶颈后的 A/B 优化 |

### 3.1 `2×8`，16 rank

- 每片约为 `M/16`；
- 15 个非 root rank 并发从 root 读取各自的种子片；
- 第二阶段每个非 root rank 并发读取剩余 15 片，即 root 片和另外 14 个非 root owner 的片；
- Server 内的每条 Mesh 直连只承载一片量级；
- 跨 Server 的单 NPU Clos 带宽约为单条 Mesh 的 8 倍，多个 receiver 对不同 owner 的 Read 可以分散到多条 Channel。

因此不必先固定成“跨机一份、机内再广播”的串行分层树。全 owner 的 Pull Scatter + AllGather 更容易同时占满 Mesh 与 Clos，并且正是初赛大包高性能方案最值得继承的部分。

### 3.2 `4×1`，4 rank

四个 rank 分属四台 Server，主要走 Clos：

- 每片为 `M/4`；
- root 作为源被读取约 `1.5M`，但 Read 由三个非 root rank 发起；
- 每个非 root owner 在第二阶段作为源被读取约 `M/2`；
- 第二阶段每个非 root rank 读取剩余 3 片；
- Direct Fanout 则要求 root 发送 `3M`。

该拓扑中 Pull Scatter + AllGather 对 root Clos 端口的压力明显更小。

### 3.3 `8+4`，12 rank

- 每片约为 `M/12`；
- 11 个非 root rank 从 root 并发读取种子片；
- 第二阶段每个非 root rank 读取剩余 11 片，即 root 片和另外 10 个非 root owner 的片；
- 本地 Mesh 链路的粒度为一片，跨机聚合流量仍受益于约 8 倍的 Clos 带宽。

算法只依赖 `rankSize=12` 和真实 Channel 图，不依赖第二台 Server 具体选中了哪四个物理 NPU。

---

## 4. CCU 数据通路

### 4.1 直接访问用户 Buffer

每次 `HcclBroadcast` 在 Host 上为完整用户 Buffer 生成 token：

```cpp
uint64_t token = 0;
uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
HcommCcuGetMemToken(baseAddr, totalBytes, &token);
```

`buf` 是原地输入/输出：

- root 的本地 `LocalAddr = {baseAddr + chunkOffset, token}` 是第一阶段远端 Read 的源；
- 非 root 的同一地址区间既是目的 Buffer，也在种子片到达后成为第二阶段的 owner 源；
- 前同步交换各 rank 的 `baseAddr` 和 token；
- 非 root 从 root 读取种子片，再从其他 owner 读取剩余片。

这样取消了：

```text
用户 Buffer -> 本地 HCCL Buffer -> 远端 HCCL Buffer -> 用户 Buffer
```

只保留：

```text
root 用户 Buffer --Read--> owner 用户 Buffer --Read by peers--> 其他非 root 用户 Buffer
```

注意：token 是安全信息，绝对不能写入日志；只能通过 CCU 的 Channel Variable 同步机制传递。

### 4.2 大包 Chunk

Host 统一按 `MAX_DATA_SIZE=256 MiB` 切分：

```text
512 MiB:
  Launch 0: offset=0,       bytes=256 MiB
  Launch 1: offset=256 MiB, bytes=256 MiB

400 MiB + 4B:
  Launch 0: offset=0,       bytes=256 MiB
  Launch 1: offset=256 MiB, bytes=144 MiB + 4B
```

每个 Launch 重新执行前同步和完成同步。先以正确、稳定为目标，不在第一版中让两个 Kernel 在同一 Channel 上并发。

### 4.3 512 KiB

第一版直接复用一次 Pull Scatter + AllGather Launch：

- 16 rank：每片 `32 KiB`；
- 12 rank：`sliceStride=43,692 B`，约 `42.67 KiB`；
- 4 rank：每片 `128 KiB`。

同时保留 `DIRECT_FANOUT` Kernel 做 A/B。不要凭初赛 60 μs/35 μs 目标直接认定哪条路径更快：CCU 把调度开销显著缩小后，两阶段的数据量优势可能超过 Direct 的单阶段优势，也可能仍受小 Read 数量影响，必须按三种拓扑分别实测。

---

## 5. CCU 信号通路

建议每条 Channel 申请 2 个 Notify 寄存器：

```text
CKE 0：前同步
  bit 1：BUFFER_ADDR_READY
  bit 2：TOKEN_READY

CKE 1：算法阶段同步
  bit 0：PHASE_SIGNAL
```

`PHASE_SIGNAL` 在 root—peer Channel 上按严格顺序复用四次：

```text
peer -> root：SEED_DONE
root -> peer：PHASE2_START
peer -> root：READ_DONE
root -> peer：GLOBAL_DONE
```

同一方向的下一次 Record 只会在上一次 Wait 已经消费后发生。非 root—非 root Channel 在算法阶段不使用 Notify，只承载 Read。若最终 API 明确禁止在同一 Kernel 内复用同一 mask，再把四个阶段拆成独立 bit；基线优先采用单 bit 方案。

### 5.1 前同步

所有 rank：

1. `LoadArg` 加载本地 `bufAddr` 和 token；
2. 对全部 Channel 调用 `WriteVariableWithNotify`，发布本地地址和 token；
3. 对全部 Channel 等待 `BUFFER_ADDR_READY | TOKEN_READY`；
4. 之后每个 rank 都拥有所有 peer 的远端地址和 token。

前同步中来自 root 的地址/token Notify 同时证明 root Kernel 已经进入本次 Launch；结合用户 stream 顺序，它可以作为 root 源 Buffer 可读的起始依赖，不再额外发送 `ROOT_READY`。

### 5.2 Seed Pull 与第一次全局栅栏

每个非 root rank：

1. 在 root Channel 上发起一个 Read，读取 `myRank` 对应的 owner slice；
2. 用本地 Event 等待该 Read 完成；无效的零长度 slice 直接 `EventRecord` 补齐 bit；
3. 向 root Record 一次 `PHASE_SIGNAL`，语义为 `SEED_DONE`；
4. Wait root 返回的 `PHASE_SIGNAL`，语义为 `PHASE2_START`。

root：

1. 不提交 Scatter Write，也不读取数据；
2. Wait 所有非 root 的 `SEED_DONE`；
3. 收齐后向全部非 root Record `PHASE2_START`。

第一次栅栏结束时，每个非 root owner 的 slice 都已经完整写入其用户 Buffer，后续任何 rank 都可以安全读取。

### 5.3 Pull AllGather 与完成栅栏

每个非 root rank：

1. 对所有满足 `ownerRank != myRank` 的 owner 发起 Read，共 `N-1` 次；其中 `ownerRank == root` 时直接从 root 读取 root slice；
2. 每次 Read 把 `ownerRank` 的 slice 直接写入本 rank 用户 Buffer 的对应 offset；
3. 用本地 Event 汇聚所有 Read；
4. Read 全部完成后向 root Record `PHASE_SIGNAL`，语义为 `READ_DONE`；
5. Wait root 返回的 `PHASE_SIGNAL`，语义为 `GLOBAL_DONE`，之后才允许 Kernel 结束。

root：

1. Wait 所有非 root 的 `READ_DONE`；
2. 此时全部远端 Read 已完成，所有 owner 源 slice 都不再被使用；
3. 向全部非 root Record `GLOBAL_DONE`，然后结束 Kernel。

### 5.4 为什么选择 receiver pull

Push 与 Pull 的数据量相同，但信号图不同：

- owner Push 需要每个 owner-target 对至少一个 `SLICE_READY`，N 分片下约为 `(N-1)^2` 个逐对完成通知，还需要 root 汇聚全局完成；
- receiver Pull 用第一次全局栅栏统一证明所有源片可读，用第二次全局栅栏统一保护源片生命周期；算法阶段共 `4(N-1)` 次 Record/Wait 配对；
- 16 rank 时约为 `60` 次阶段信号，而逐对 Push 仅 `SLICE_READY` 就需要 `225` 次；
- 所有 Mem2Mem 都由接收 rank 发起，第一阶段不会把任务提交压力集中在 root。

因此基线选择 Pull。后续只有在实测证明 CCU Write 的数据面明显更快时，才增加 Push 变体进行 A/B。

### 5.5 为什么不会死锁

- 非 root 的 Seed Read 只依赖已经完成的前同步，不等待其他非 root；
- root 等完全部 `SEED_DONE` 才发布 `PHASE2_START`，所以第二阶段不会读取尚未就绪的 owner slice；
- 第二阶段每个 rank 只提交 Read 并等待自己的 Event，不存在 peer 之间互相先 Wait 的环；
- root 收齐全部 `READ_DONE` 后才发布 `GLOBAL_DONE`，保证 owner 不会在远端仍读取其源片时提前退出；
- 同一个 Phase bit 的四次使用方向交替、顺序固定，每次 Record 都有唯一 Wait。

---

## 6. 两个 Kernel 的详细设计

### 6.1 静态 KernelArg

建议在 `ccu_kernel.h` 或 `custom.h` 中定义：

```cpp
struct BroadcastKernelArg : CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t remoteRanks[MAX_RANK_SIZE]; // 与 channels[] 一一对应
};
```

KernelArg 只保存注册时不会变化的内容：`rankSize`、`rankId`、Channel 和 Channel 对应的 remote rank。

以下内容不能放入 KernelArg：

- root；
- 用户 Buffer 地址；
- token；
- totalBytes/chunkOffset/chunkBytes；
- sliceStride。

它们必须是每次 Launch 的 TaskArg，保证同一 Engine Context 支持 root 和数据量变化。

### 6.2 TaskArg 顺序

建议两个 Kernel 使用同一套前六项：

```text
taskArgs[0] = baseAddr
taskArgs[1] = token
taskArgs[2] = root
taskArgs[3] = chunkOffset
taskArgs[4] = chunkBytes
taskArgs[5] = sliceStride       // Direct 可传 chunkBytes
```

`LoadArg` 的索引必须从 0 连续，Host 的 `argNum` 必须严格等于 Kernel 中不同 `LoadArg` 的数量。

### 6.3 动态 root 的实现

不要为不同 root 重复申请 Channel，也不建议一开始注册 16 个 root 专用 Kernel。

Kernel 中 `rankId` 和 `remoteRanks[i]` 是静态值，root 是 CCU Variable：

```cpp
CCU_IF(ctx.root == arg->rankId) {
    // root 分支
}
CCU_ELSE {
    // owner 分支
}
```

查找 root Channel 时对静态 Channel 数组展开：

```cpp
for (uint32_t i = 0; i < arg->channelCount; ++i) {
    CCU_IF(ctx.root == arg->remoteRanks[i]) {
        ccu::NotifyWait(arg->channels[i], CKE_PHASE, MASK_PHASE_SIGNAL);
    }
}
```

N 分片下 owner 与 rank 直接对应：

```text
ownerIndex = ownerRank
ownerRank  = ownerIndex
```

动态 root 只用于选择 root Channel和跳过第二阶段的 root/self，不参与 owner 编号计算。

### 6.4 `PULL_SCATTER_ALLGATHER` 伪代码

```text
InitResource()
LoadArgs(baseAddr, token, root, chunkOffset, chunkBytes, sliceStride)
PreSyncAllPeerAddressesAndTokens()

Slice(ownerRank):
    off = ownerRank * sliceStride
    len = off >= chunkBytes ? 0 : min(sliceStride, chunkBytes - off)

if myRank == root:
    // Barrier 1：等所有非 root 拉到自己的 owner slice
    Wait PHASE_SIGNAL(SEED_DONE) from every non-root
    Record PHASE_SIGNAL(PHASE2_START) to every non-root

    // Barrier 2：等所有非 root 拉完剩余 owner slice
    Wait PHASE_SIGNAL(READ_DONE) from every non-root
    Record PHASE_SIGNAL(GLOBAL_DONE) to every non-root
else:
    // Seed Pull：当前 receiver 只读取自己的 owner slice
    Read(local.buf + chunkOffset + Slice(myRank).off,
         root.buf  + chunkOffset + Slice(myRank).off,
         Slice(myRank).len, seedEvent, bit(myRank))
    EventWait(seedEvent, validSeedMask)

    Record PHASE_SIGNAL(SEED_DONE) to root
    Wait PHASE_SIGNAL(PHASE2_START) from root

    // Pull AllGather：读取其余 N-1 个 owner slice，包括 root slice
    for ownerRank in [0, rankSize):
        if ownerRank != myRank:
            Read(local.buf + chunkOffset + Slice(ownerRank).off,
                 owner[ownerRank].buf + chunkOffset + Slice(ownerRank).off,
                 Slice(ownerRank).len,
                 gatherEvent, bit(ownerRank))
        else:
            EventRecord(gatherEvent, bit(ownerRank))
    EventWait(gatherEvent, allRankMask)

    Record PHASE_SIGNAL(READ_DONE) to root
    Wait PHASE_SIGNAL(GLOBAL_DONE) from root
```

实际代码必须保护 `len=0` 和 `off>=chunkBytes`。若为了使用固定 Event mask 跳过某次 Read，应对相应 bit 调用 `EventRecord`，否则 `EventWait` 会永久等待。

### 6.5 `DIRECT_FANOUT` 伪代码

```text
PreSyncAllPeerAddressesAndTokens()

if myRank == root:
    Write full Chunk to every non-root peer with different Event bits
    EventWait(allNonRootMask)
    Record PHASE_SIGNAL(DIRECT_READY) to every peer
    Wait PHASE_SIGNAL(DIRECT_DONE) from every peer
else:
    Wait PHASE_SIGNAL(DIRECT_READY) from root
    Record PHASE_SIGNAL(DIRECT_DONE) to root
```

这条路径任务更少，但 root 数据发送量为 `(N-1)M`。只用于 512 KiB A/B，不用于大包。

---

## 7. 六个允许修改文件的落地设计

### 7.1 `include/custom.h`

新增或调整：

- `ResourceLayoutVersion`；
- `BroadcastKernelArg`；
- `KernelKind { DIRECT, PULL_SCATTER_ALLGATHER }`；
- `AlgResourceCtx` 中保存：
  - `rankSize`；
  - Channel 相关静态映射；
  - 两个 `CcuKernelHandle`；
  - 可保留 `localBuffer` 作为环境信息，但主路径不使用；
- Serialize/DeSerialize 的字段顺序和版本严格一致。

不要把 `root/count/buf/token` 写入 Engine Context。

### 7.2 `op_host/broadcast.cc`

实现以下函数并保持职责分离：

```text
ValidateBroadcastParam
QueryBestCcuLinkToPeer
AcquireAllPeerChannels
BuildKernelArg
RegisterBroadcastKernels
CreateAndStoreEngineContext
```

资源计划：

```text
Thread：每次调用 HcclThreadAcquireWithStream(... COMM_ENGINE_CCU ..., notify=0)
额外 Thread：0
Thread Notify：0
Channel：rankSize - 1
每 Channel Notify：2
Kernel：2
```

Channel 获取时：

1. 扫描 RankGraph 可用网络层；
2. 优先选择 CCU 支持的 `COMM_PROTOCOL_UBC_CTP`；
3. 跳过 `COMM_PROTOCOL_RESERVED`；
4. 每个 remoteRank 只创建一个 Channel；
5. Channel 数组按 remoteRank 升序，并同步填充 `remoteRanks[]`。

Kernel 注册必须放在一次完整流程中：

```text
HcclCommQueryCcuIns
HcommCcuKernelRegisterStart
HcommCcuKernelRegister(DIRECT)
HcommCcuKernelRegister(PULL_SCATTER_ALLGATHER)
HcommCcuKernelRegisterEnd
```

热路径上使用本次 `HcclThreadAcquireWithStream` 得到的 `param.cpuThread` 下发 Kernel，不要错误复用第一次调用序列化进去的 stream-bound Thread。

### 7.3 `op_host/exec_op.h`

声明：

```text
BuildExecutionPlan
LaunchDirectChunk
LaunchPullScatterAllGatherChunk
ExecOp
```

必要的纯 Host 结构可放这里：`ExecutionPlan`、`ChunkDesc`。

### 7.4 `op_host/exec_op.cc`

负责：

1. 反序列化和版本校验；
2. 计算 `totalBytes = count * 4` 并检查溢出；
3. 为完整用户 Buffer 获取一次 token；
4. 循环构造不超过 256 MiB 的 Chunk；
5. 计算 4B 对齐的 `sliceStride`；
6. 构造严格连续的 TaskArg；
7. 使用当前调用的 `param.cpuThread` 执行 `HcommCcuKernelLaunch`；
8. 512 KiB 根据 A/B 选择 Kernel，大包固定 Pull Scatter + AllGather。

第一版选择器：

```cpp
if (totalBytes == 512_KiB) {
    algorithm = PULL_SCATTER_ALLGATHER; // 先作为默认
} else {
    algorithm = PULL_SCATTER_ALLGATHER;
}
```

保留一个编译期常量或非常小的选择表切换 512 KiB Direct，便于测量，不要在热路径读环境变量或打印逐 Chunk 日志。

### 7.5 `op_kernel_ccu/ccu_kernel.h`

定义：

- CKE index 和 bit mask；
- `BroadcastContext`；
- `InitResource`、`LoadArgs`、`PreSync` 等声明；
- 两个 Kernel 入口。

使用固定数组，不在 Kernel 中根据 rankSize 动态分配大量 STL 对象。

### 7.6 `op_kernel_ccu/ccu_kernel.cc`

建议函数结构：

```text
InitBroadcastResource
LoadBroadcastArgs
PreSyncBufferInfo
GetRootChannelAndSliceDesc
RunDirectRoot / RunDirectReceiver
RunPullSeed / RunPullAllGather
CcuBroadcastDirectKernel
CcuBroadcastPullScatterAllGatherKernel
```

每个 CCU API 都用 `CCU_CHK_RET` 检查返回值。禁止在 Kernel 热路径输出地址、token 或逐 peer 日志。

---

## 8. 正确性不变量

实现完成前逐条核对：

1. 一个通信域中，每个 remote rank 只对应一个 Channel；
2. `channels[i]` 与 `remoteRanks[i]` 始终一一对应；
3. Engine Context 不包含 root、buf、count、token；
4. root 作为 TaskArg，每次调用都可变化；
5. `LoadArg` 使用的 id 从 0 连续，Host `argNum` 完全一致；
6. token 覆盖完整用户 Buffer，任何 `base+offset+bytes` 都不得越界；
7. Chunk 不超过 256 MiB；
8. `ownerCount == rankSize`，`ownerIndex == ownerRank`；
9. `sliceStride` 为 4B 对齐，最后一片只使用真实 bytes，零长度片不发起 Mem2Mem；
10. root 原始 Buffer 不被任何 rank 写回；
11. 非 root 的 Seed Pull 只从 root 读取自己的 owner slice；
12. root 收齐全部 `SEED_DONE` 后才能发布 `PHASE2_START`；
13. 非 root 收到 `PHASE2_START` 后才能读取剩余 `N-1` 片，包括 root slice 和其他非 root owner 的片；
14. 非 root 的全部本地 Read Event 完成后才能发送 `READ_DONE`；
15. root 收齐全部 `READ_DONE` 后才能发布 `GLOBAL_DONE`；
16. 所有非 root 必须收到 `GLOBAL_DONE` 后才能结束 Kernel，确保远端 Read 的源 Buffer 生命周期安全；
17. 同一 `PHASE_SIGNAL` 每次 Record 前，上一次同方向信号已被唯一 Wait 消费；
18. 每个 Notify Record 都有唯一 Wait，每个 Wait 都存在生产者；
19. 连续执行 `root=0 -> root=7/1 -> root=0` 不残留信号；
20. 不打印 token，不在错误日志中输出安全信息；
21. 热路径 Launch 使用本次用户 stream 对应的 Thread。

---

## 9. 实施顺序

### 阶段 A：先跑通 CCU 最小闭环

1. 完成一 Thread、全 Channel、单 Kernel 的资源注册；
2. Kernel 只做地址/token 前同步和后同步，不搬数据；
3. 依次在 4/12/16 rank 验证 Kernel 可注册、Launch、退出；
4. 连续 Launch 多次，确认 Notify 无残留。

### 阶段 B：Direct 基线

1. 实现 `DIRECT_FANOUT`；
2. 先验证 512 KiB；
3. 再通过 Host Chunk 验证 512 MiB 和 400 MiB+4B；
4. 这条路径是功能保底，也是后续性能比较基线。

### 阶段 C：N 分片 Pull Scatter + AllGather

1. 按 `rankSize` 切成 N 片，验证 `ownerIndex == ownerRank`；
2. 非 root 从 root 读取自己的 owner slice；
3. 实现 `SEED_DONE -> PHASE2_START` 全局栅栏；
4. 非 root 从其余 owner 读取缺失的 `N-1` 片，其中包括 root slice；
5. 实现 `READ_DONE -> GLOBAL_DONE` 完成栅栏；
6. 验证 root=0 后再验证 root=7/1；
7. 验证 4/12/16 rank；
8. 最后接入大包多 Chunk。

### 阶段 D：性能优化

按收益/风险顺序：

1. 512 KiB 在三种拓扑分别比较 Direct 与 Pull Scatter + AllGather；
2. 确认多个 Event Read 是否真正并行展开，避免逐 Read 后立刻 Wait；
3. 删除所有非必要 Kernel 日志；
4. 确认前同步是否能减少到只交换必要 rank，但不要先牺牲通用 root；
5. 若 512 KiB Direct 胜出，只修改该拓扑的选择表；
6. 大包稳定后，再研究单 Kernel 内双 Chunk/流水，前提是官方明确允许且不违反 256 MiB 约束；
7. N 分片基线稳定后，再 A/B 测试 N-1 分片；只有 root 源流量确实成为瓶颈时才切换；
8. 最后才考虑 2×8/8+4 的层级专用 Kernel，不要一开始引入复杂物理 Server 映射。

---

## 10. 测试矩阵

### 10.1 官方 18 个功能用例

| 拓扑 | root | 数据量 |
| --- | --- | --- |
| 2×8 | 0、7 | 512 KiB、512 MiB、400 MiB+4B |
| 4×1 | 0、1 | 512 KiB、512 MiB、400 MiB+4B |
| 8+4 | 0、7 | 512 KiB、512 MiB、400 MiB+4B |

题面写法为“root rank 分别为 0、7/1”；上述对应关系是按 rank 数和拓扑推断，实际运行前应以官方通信域配置再次确认。

### 10.2 必加边界用例

```text
256 MiB - 4B
256 MiB
256 MiB + 4B
400 MiB
400 MiB + 4B
512 MiB
连续三次：root0 -> root7/1 -> root0
同一 root 连续 20 次
```

### 10.3 性能记录

每个拓扑和数据量都记录：

```text
预热次数
正式迭代次数
最小值
中位数
P95/最大值
算法选择
Chunk 数
是否出现偶发超时或长尾
```

512 KiB 必须保存 Direct 与 Pull Scatter + AllGather 两组数据，不能只看一次最好成绩。

---

## 11. 性能风险与备用策略

### 风险 1：512 KiB 的小 Read 数量过多

表现：Pull Scatter + AllGather 的两阶段同步或每 receiver 多个 Read 的固定开销超过数据量收益。

处理：按拓扑切到 Direct；大包仍保留 Pull Scatter + AllGather。

### 风险 2：CCU Kernel 资源不足

表现：注册返回 `CCU_E_UNAVAIL`，通常来自 Variable/Event/指令资源太多。

处理：

- 复用一个 Event；
- 固定数组只为 rankSize 上限 16；
- Direct 与 Pull Scatter + AllGather 分成两个 Kernel，避免一个超大 Kernel；
- 不注册 root 专用的 16 份 Kernel。

### 风险 3：用户 Buffer Mem2Mem 在 Checker 中受限

官方 CCU 示例支持通过 `HcommCcuGetMemToken` 直接访问用户 Buffer，因此主方案应先走零拷贝。如果实际比赛 Checker 明确拒绝，再增加 CCL Buffer staging 备用路径：

```text
root 用户 Buffer -> root CCL Buffer
CCU 在 CCL Buffer 间执行 Pull Scatter + AllGather
非 root CCL Buffer -> 用户 Buffer
```

该备用路径应使用不超过 200 MiB 的双缓冲窗口，以适配 400 MiB CCL Buffer；但它会增加 LocalCopy，不应作为第一版。

### 风险 4：动态 root 生成的 CCU 指令过大

先用一个动态 root Kernel。若注册资源或指令数超限，再退为“只注册题面需要的两个 root 专用 Kernel”，但仍共享同一组 Channel，绝不能为不同 root 再申请 Channel。

### 风险 5：多 Chunk 重复前同步产生额外开销

对 256 MiB 级大 Chunk，这部分通常占比很小。先保证两次 Launch 正确；只有性能证据表明它是瓶颈，才研究跨 Chunk 保留地址/token 或合并 Kernel。

---

## 12. 一句话实现蓝图

```text
Host：全 peer 单 Channel + 1 个当前 stream 的 CCU Thread + 注册 Direct/PullScatterAllGather 两个 Kernel
  -> ExecOp：完整 buf 获取 token，按 256 MiB 切 Chunk，构造连续 TaskArg
  -> CCU：交换地址/token
  -> N 分片：ownerIndex == ownerRank
  -> 非 root 从 root 读取自己的 owner slice
  -> root 全局 Seed Barrier
  -> 非 root 从其他 N-1 个 owner 并发读取剩余片（包括 root slice）
  -> root 全局 Completion Barrier 并释放所有 rank
  -> 下一 Chunk / 返回
```

第一版目标应是：18 个功能用例全部通过、连续调用无残留、三种拓扑统一运行。之后先对 512 KiB 做 Direct/Pull Scatter + AllGather 的小范围选择，再根据大包瓶颈决定是否把 N 分片调整为 N-1 分片，不要过早引入三套完全不同的大包算法。

---

## 13. 参考源码与文档

- [比赛仓库 README：初赛/决赛目录、拓扑和模板说明](https://github.com/ghhufue/HCCL-Competition/blob/main/README.md)
- [决赛原始题面](https://github.com/ghhufue/HCCL-Competition/blob/main/docs/Finals/problem.txt)
- [决赛 Host 模板](https://github.com/ghhufue/HCCL-Competition/blob/main/Hccl_Broadcast_Final/op_host/broadcast.cc)
- [决赛 CCU Kernel 模板](https://github.com/ghhufue/HCCL-Competition/blob/main/Hccl_Broadcast_Final/op_kernel_ccu/ccu_kernel.cc)
- [初赛完整 AICPU 数据面实现](https://github.com/ghhufue/HCCL-Competition/blob/main/Hccl_Broadcast_Preliminary/op_kernel_aicpu/exec_op.cc)
- [HCOMM CCU Quick Start](https://github.com/hicann/hcomm/blob/55f1fb5d430adb46e9d23603a7711a312bf63c5a/docs/zh/comm_op_dev_guide/ccu_quick_start.md)
- [HCOMM CCU 资源创建说明](https://github.com/hicann/hcomm/blob/55f1fb5d430adb46e9d23603a7711a312bf63c5a/docs/zh/comm_op_dev_guide/ccu_comm_op_dev/create_res.md)
- [HCOMM CCU 算法执行与 TaskArg 说明](https://github.com/hicann/hcomm/blob/55f1fb5d430adb46e9d23603a7711a312bf63c5a/docs/zh/comm_op_dev_guide/ccu_comm_op_dev/algo_exec.md)
- [HCOMM 2026 AllGather CCU Kernel 示例](https://github.com/hicann/hcomm/blob/55f1fb5d430adb46e9d23603a7711a312bf63c5a/test/ut/framework/next/comms/ccu/ccu_kernel_impl/ccu_groupcopy_kernel.cc)
