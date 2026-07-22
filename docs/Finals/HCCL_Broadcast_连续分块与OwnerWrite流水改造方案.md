# HCCL Broadcast：连续分块与 Owner-Write Tile 流水改造任务书

> 将本文直接交给负责修改仓库的 AI。请先完整阅读现有实现、画出当前数据通路和信号通路，再按本文实施。不要脱离现有 HCOMM/CCU API 凭空重构，也不要未经验证删除当前可运行基线。

## 一、任务目标

在保留前半段 `receiver pull` 的前提下，对大包 Broadcast 的数据布局和 AllGather 阶段进行改造：

1. Phase 0～2 的基本方向不变：从 root rank 获取 owner 分片时，始终由 owner rank 主动 Pull；禁止改回 root Push。
2. 将全局 Tile 的 `block-cyclic` 归属改为按 rank 划分的连续块归属。
3. 将 AllGather 从 `receiver pull` 改为分布式 `owner write`。
4. owner 的第一个 Tile ready 后立即开始向目标 rank 写入，不等待 owner 完整分片。
5. Push worker 空闲时，将当前所有已经连续 ready、尚未提交的 Tile 尽快发送。
6. 若 Push 执行期间自然积累了多个连续 Tile，允许顺便合并成较大的 Batch；禁止主动等待下一个 Tile 来凑 Batch。
7. 单次 Push Batch 设置上限，首选测试 `8 MiB` 与 `16 MiB`。
8. 增加一个布尔变量控制是否启用自然合并，便于与“每个 Tile 立即独立 Push”的基线进行 A/B 测试。

本次改造的核心数据通路应为：

```text
root完整用户Buffer
        │
        │ owner主动Read（receiver pull）
        ▼
各owner最终用户Buffer中的连续owner block
        │
        │ Tile ready后owner主动Write
        ▼
其他非root rank最终用户Buffer中的对应连续区间
```

## 二、必须遵守的设计约束

### 2.1 Root 到 owner 始终使用 owner Pull

Phase 0～2 沿用现有地址/token交换和 Seed Pull 框架。非 root owner 必须主动从 root 的用户 Buffer 读取自己的连续分片：

```text
owner rank发起Read：
rootUserBuffer[ownerOffset, ownerOffset + ownerBytes)
    → ownerUserBuffer[ownerOffset, ownerOffset + ownerBytes)
```

不得改成：

```text
root主动向各owner并行Write
```

root 对自己所属的连续块天然已经 ready，不进行自拷贝或无意义的 Pull。

### 2.2 保持 N 个 owner block

总数据仍按 `rankSize` 分成 N 个 owner block，每个 rank 都是一个 owner，包括 root。

本次禁止擅自改为 N−1 分片。root 所属分片需要在 AllGather 中由 root 写给所有非 root rank。

### 2.3 AllGather 使用 owner Write

owner 的某段数据 ready 后，由 owner 主动将该段写入其他 rank 的最终用户 Buffer。receiver 不再主动 Read owner，也不再等待 owner 的逐 Tile 远端 `READY`。

owner write 的目标集合必须正确：

```text
root owner：
  写给所有非root rank

非root owner：
  写给除root、自己以外的所有非root rank
```

不需要把非 root owner 的数据写回 root，因为 root 原本就拥有完整数据；也不需要写回 owner 自己。

### 2.4 直接写最终用户 Buffer

每个 owner 只写自己负责的连续地址区间，不同 owner 的目标区间不得重叠。AllGather 期间 source 和 destination 用户 Buffer 在算子完成前不得被复用或覆盖。

因此本方案不需要：

- receiver 的逐 Tile `READ_DONE`；
- receiver 的逐 Tile消费 ACK；
- owner 向每个 receiver 发送逐 Tile `READY`；
- 用 slot/window ACK 管理最终用户 Buffer 生命周期。

但全局结束前仍必须确认所有 owner 的所有远端 Write 已真正完成并对 receiver 可见。

## 三、连续块分配

### 3.1 删除 block-cyclic Tile 归属

旧布局示意：

```text
Tile 0  → owner 0
Tile 1  → owner 1
...
Tile N  → owner 0
```

这种布局使同一 owner 的多个 Tile 在用户 Buffer 中不连续，无法合并 Write。

新布局必须是连续区间：

```text
owner 0 → [blockBegin(0), blockEnd(0))
owner 1 → [blockBegin(1), blockEnd(1))
...
owner N-1 → [blockBegin(N-1), blockEnd(N-1))
```

每个 owner 再在自己的 block 内按 `tileSize` 进行流水切分：

```text
localTileOffset = ownerOffset + localTileId * tileSize
localTileBytes  = min(tileSize, ownerEnd - localTileOffset)
```

### 3.2 不允许使用会丢失尾部的简单整数除法

不要直接使用：

```cpp
tilesPerOwner = totalBytes / rankSize / tileSize;
```

这会遗漏不能整除的数据和最后尾块。应实现统一的连续分区函数，保证：

```text
blockBegin(0) = 0
blockEnd(N-1) = totalBytes
blockEnd(i) = blockBegin(i+1)
所有ownerBytes之和 = totalBytes
任意两个owner区间无重叠、无空洞
```

优先沿用仓库现有的对齐要求。若传输长度或地址必须对齐，可让前 N−1 个 block 按合法粒度切分，由最后一个 owner 吸收剩余尾部；但必须覆盖 4 B、`tileSize+4 B`、`400 MiB+4 B` 等非整除用例。

### 3.3 建立唯一的数据布局辅助函数

Host、AICPU、CCU kernel 不得各自复制一套略有差异的 owner/Tile 计算公式。应建立或复用统一的辅助逻辑，至少能得到：

```cpp
struct OwnerBlock {
    uint64_t offset;
    uint64_t bytes;
    uint32_t tileCount;
};
```

具体放置位置遵循现有工程结构。修改后全仓检查并删除旧的 `% rankSize`、`tileId % rankSize` 等 block-cyclic 归属逻辑，除非它们属于其他算法且已明确隔离。

## 四、Tile 流水与自然合并策略

### 4.1 首 Tile 必须立即 Push

owner 的第一个 Seed Tile 完成后，立即允许 Push worker 将其写给负责的 peer，不等待：

- 下一个 Tile；
- owner 的全部 Seed Pull；
- 其他 owner 完成；
- 全局 `SEED_DONE` barrier。

目标是让 Seed Pull 与 Owner Write AllGather 按 Tile 流水重叠。

### 4.2 长驻 Push worker，不为每个 Tile新建执行实体

“立即发送”应由已经存在的 Push worker/kernel完成，不得为每个 Tile：

- 创建新的 Host 线程；
- 重新建立通信资源；
- 重复申请 notify/event；
- 产生不必要的新 kernel launch。

推荐逻辑角色：

```text
Seed worker：
  顺序Pull owner block内的Tile
  → 等待该Read真正完成
  → 发布本rank内部的连续ready水位

Die 0 Push worker：
  消费ready水位
  → Write给Die 0负责的peer集合

Die 1 Push worker：
  消费ready水位
  → Write给Die 1负责的peer集合
```

可按现有代码能力合并角色，但必须保持上述完成语义。

### 4.3 使用连续 ready watermark

连续块内若 Seed Pull 按顺序完成，优先使用水位而不是每 Tile bitmap：

```cpp
seedReadyBytes;       // 已完成Seed Pull的最长连续前缀
pushSubmittedBytes;   // 已经提交给当前Push worker的连续前缀
pushCompletedBytes;   // 可选：已经确认远端完成的连续前缀
```

三个状态不可混淆：

- `ready`：源数据已经可以安全读取；
- `submitted`：Write 请求已经提交，但不一定远端可见；
- `completed`：Write 已完成且满足远端可见性要求。

若现有实现允许 Seed Tile 乱序完成，则必须通过完成 bitmap/序号推进“最长连续 ready 前缀”，不能越过尚未完成的 Tile。

### 4.4 Push worker 空闲时立即处理当前积压

Push worker 变为空闲后，读取当前 `availableBytes`：

```cpp
availableBytes = seedReadyBytes - pushSubmittedBytes;
```

只要 `availableBytes > 0` 就立即提交，不主动等待更多 Tile。

推荐伪代码：

```cpp
if (pushWorkerIdle) {
    uint64_t availableBytes = seedReadyBytes - pushSubmittedBytes;
    if (availableBytes > 0) {
        uint64_t batchBytes = SelectPushBatchBytes(availableBytes);
        SubmitOwnerWrite(ownerOffset + pushSubmittedBytes, batchBytes);
        pushSubmittedBytes += batchBytes;
    }
}
```

### 4.5 增加布尔合并开关

新增一个语义明确的布尔配置，例如：

```cpp
bool enablePushBatchMerge;
```

名称可按仓库规范调整，但不得把它与 `tileSize` 或 `pipelineDepth` 混为一谈。

行为必须是：

```cpp
uint64_t SelectPushBatchBytes(uint64_t availableBytes)
{
    if (!enablePushBatchMerge) {
        return std::min(availableBytes, currentTileRemainingBytes);
    }

    // 只合并当前已经ready的连续数据，不等待未来Tile
    return std::min(availableBytes, maxPushBatchBytes);
}
```

关闭合并时：

```text
每个基础Tile独立Push；尾Tile按实际长度发送。
```

启用合并时：

```text
首Tile ready后仍立即发送；
Push期间若自然积累了多个连续Tile，下一次可以合并；
不得通过轮询、sleep、额外barrier或延迟调度等待下一个Tile；
单次Batch不得超过maxPushBatchBytes。
```

### 4.6 Batch 上限

支持至少以下配置进行 A/B 测试：

```text
基础TileSize：沿用当前值，例如4 MiB
enablePushBatchMerge=false：单次最多一个基础Tile
enablePushBatchMerge=true，maxPushBatchBytes=8 MiB
enablePushBatchMerge=true，maxPushBatchBytes=16 MiB
```

不要把 8 MiB 或 16 MiB 硬编码散落在多个文件。Host 配置、资源估算、kernel 参数和日志必须一致。

## 五、双 Die 数据和信号协议

### 5.1 Seed Pull 完成后发布本地 Tile ready

若只有连接 root 的 `rootDie` 执行 Seed Pull，而两个 Die 都负责 owner write，则另一个 Die 在读取该 Tile 前必须知道 Seed Read 已真正完成。

因此流水版至少需要 owner rank 内部的 Tile ready 传递：

```text
rootDie完成Seed Read(Tile k)
    ├─ 本Die可以消费Tile k
    └─ 通知另一Die：Tile k已ready
```

该通知是本 rank 内部/跨 Die 的就绪同步，不是发给所有 receiver 的远端 `READY`。

若使用 watermark 或累计序号复用 notify，必须证明不会出现：

- 快 producer 覆盖慢 consumer 尚未消费的状态；
- 不同 window/Tile/连续轮次之间串信号；
- notify 累计值回绕或资源编号冲突；
- `pipelineDepth > 1` 时跨阶段误消费。

### 5.2 两个 Die 独立负责 peer 子集

两个 Die 根据现有 Channel 拓扑划分目标 peer：

```text
Die 0 → peer集合A
Die 1 → peer集合B
A与B无重复，A∪B等于该owner的完整目标集合
```

每个 ready Batch 到来后，两个 Die 分别向自己的 peer 集合发起 Write。Push 期间不做逐 Batch 的跨 Die完成 barrier，避免两个 Die互相锁步。

### 5.3 不允许把“最后一批已提交”视为 owner 完成

最终完成必须满足：

```text
Die 0：自己负责的每条peer Channel上的所有Write均完成
Die 1：自己负责的每条peer Channel上的所有Write均完成
两个Die完成状态汇合
→ OWNER_DONE
```

仅等待某一个 peer、某一个 Die 或某一个“最后提交的 Write”是不够的，除非已经从官方 API/现有实现证明该 Event/fence 能覆盖此前所有相关 Channel，并在代码注释中写清保证。

若 Channel 内保证 FIFO，可以对每条 peer Channel 等待最后一个 Write 的完成；若没有此保证，则必须等待全部对应 Event，或使用明确覆盖全部 Write 的 drain/fence。

必须确认所用 Write 完成事件的语义是：数据已到达远端并对 receiver 可见，而非仅仅“本地请求已提交”。

### 5.4 保留全局完成协议

receiver 被动接收 owner write，无法仅靠本地状态知道整个 Broadcast 完成。必须保留或建立：

```text
每个owner：
  两个Die的所有peer Channel均drain
  → 本rank跨Die汇合
  → 向root发送OWNER_DONE

root：
  自己的owner push已完成
  + 收齐全部非root OWNER_DONE
  → 向所有非root发送GLOBAL_DONE

所有rank：
  收到/形成GLOBAL_DONE后才允许算子返回
```

不要因为取消逐 Tile ACK 就取消最终全局完成。连续调用时，上一轮所有远端 Write 必须完成后才能复用 Buffer、notify、event 和通信资源。

## 六、建议的阶段定义

在尽量保持当前 Phase 命名和代码结构的前提下，将逻辑明确为：

```text
Phase 0～1：交换用户Buffer地址、token和必要元数据

Phase 2：SEED_PULL_PIPELINE
  非root owner从root Pull自己的连续block
  每个Tile完成后推进本地ready水位
  root owner的本地block初始即ready

Phase 3～4：OWNER_WRITE_ALLGATHER_PIPELINE
  第一个ready Tile立即Push
  Seed Pull与Owner Write重叠
  无逐Tile远端READY/ACK
  可选自然合并，绝不主动等待凑批

Phase 5：OWNER_DRAIN_AND_DONE
  每个Die drain全部peer Channel
  跨Die汇合后发送OWNER_DONE

Phase 6：GLOBAL_DONE
  root汇总所有owner完成状态
  通知所有rank后结束Broadcast
```

如果现有 Phase 0～2 的编号含义不同，不要求机械改名；但最终数据流和同步语义必须满足本文要求，并在修改报告中给出“旧阶段 → 新阶段”的对应表。

## 七、实施要求

### 7.1 修改前先完成代码审计

先在仓库中定位并列出：

1. 当前 block-cyclic owner 计算位置；
2. 当前 Seed Pull 的 kernel、线程和 Channel；
3. 当前 receiver-pull AllGather 的提交与 ready/read-done 信号；
4. 地址/token交换及动态 root 处理；
5. 两个 Die 的 peer 分工与跨 Die汇合；
6. notify/event 资源数量、编号和生命周期；
7. Host 侧算法选择、tiling、资源估算和环境变量；
8. 当前最终完成协议；
9. 小包算法与其他算法是否复用这些代码。

输出简短审计结果后再修改。不要只改表面 Tile 映射而遗漏 Host 资源申请、kernel 参数、日志和完成计数。

### 7.2 尽量复用现有抽象

优先复用现有：

- HCOMM Read/Write 提交接口；
- Channel 到 Die 的映射；
- 地址/token交换结果；
- notify/event 分配框架；
- 动态 root 逻辑；
- 尾块处理和合法长度检查；
- `OWNER_DONE/GLOBAL_DONE` 或等价完成框架。

不要在没有必要时重写整个 Broadcast。新增的数据结构和状态机应保持边界清晰，避免 Seed、Push 和完成协议相互隐式依赖。

### 7.3 算法和资源隔离

若仓库同时保留旧算法用于 A/B 对照：

- 新旧算法必须有明确选择入口；
- 不得共享会互相污染的 notify 编号和累计状态；
- 新算法失败时不得静默回退并伪装成新算法测试结果；
- 日志必须打印实际运行算法、连续分块、合并开关、TileSize 和 MaxBatch。

### 7.4 配置建议

按照项目现有风格提供编译期或运行时配置，至少能表达：

```text
algorithm=CONTIGUOUS_OWNER_WRITE
tileSizeBytes=4 MiB（或现有默认值）
enablePushBatchMerge=true/false
maxPushBatchBytes=8 MiB或16 MiB
```

若使用环境变量，必须：

- 解析非法值并给出明确错误或安全默认值；
- 保证所有 rank 使用相同配置；
- 将生效值打印到调试日志；
- 不改变无关小包路径的选择结果。

## 八、伪代码参考

以下仅表达状态机，不要求照抄 API：

```cpp
OwnerBlock block = GetOwnerBlock(totalBytes, rankSize, rank);

// root本来就拥有完整输入，其owner block直接ready。
if (rank == root) {
    PublishSeedReady(block.bytes);
} else {
    for (uint64_t localOffset = 0; localOffset < block.bytes;) {
        uint64_t bytes = std::min(tileSizeBytes, block.bytes - localOffset);

        SubmitReadFromRoot(
            rootBuffer + block.offset + localOffset,
            localBuffer + block.offset + localOffset,
            bytes);

        WaitSeedReadRemoteVisibleOrLocallyReadable();
        localOffset += bytes;
        PublishSeedReady(localOffset);  // 连续前缀水位
    }
}
```

每个 Die 的长期 Push worker：

```cpp
uint64_t submittedBytes = 0;

while (submittedBytes < block.bytes) {
    uint64_t readyBytes = ObserveSeedReadyPrefix();
    if (readyBytes == submittedBytes) {
        WaitForReadyProgress();
        continue;
    }

    uint64_t availableBytes = readyBytes - submittedBytes;
    uint64_t batchBytes;

    if (enablePushBatchMerge) {
        // 只消费已经自然积压的连续数据
        batchBytes = std::min(availableBytes, maxPushBatchBytes);
    } else {
        // 保持基础Tile边界；尾Tile使用实际长度
        batchBytes = BytesUntilCurrentTileEnd(submittedBytes, block.bytes,
                                               tileSizeBytes);
    }

    for (Rank peer : PeersOwnedByThisDie(rank, root)) {
        SubmitWriteToPeerFinalBuffer(
            localBuffer + block.offset + submittedBytes,
            peerBuffer  + block.offset + submittedBytes,
            batchBytes);
    }

    submittedBytes += batchBytes;
}

DrainEveryPeerChannelOwnedByThisDie();
PublishLocalDieDone();
JoinBothDiesAndPublishOwnerDone();
```

实际实现中必须避免两个 Die 竞争写同一个 `submittedBytes`。每个 Die 应维护自己的消费水位，或者由单一调度者发布不可丢失的 Batch 描述；最终两个 Die 都必须完整覆盖 owner block。

## 九、明确禁止的错误实现

1. 将 root→owner 改为 root 主动 Write。
2. 继续用 `globalTileId % rankSize` 决定 owner，却声称已经连续分块。
3. 用整数除法平均分块后遗漏尾部数据。
4. 等到两个或四个 Tile ready 才发送首批数据。
5. 开启合并后主动 sleep/轮询等待下一个 Tile。
6. 每个 Tile 创建一个新 Host 线程或重新 launch 一套临时 kernel。
7. owner write 后仍向每个 receiver 发送逐 Tile READY。
8. receiver 对最终用户 Buffer 中的每个 Tile 返回 ACK。
9. 两个 Die 每发送一个 Batch 就强制 barrier，破坏并行推进。
10. 把最后一个 Write“已提交”当成所有远端数据“已完成”。
11. 只 drain 一个 peer Channel 就上报 `OWNER_DONE`。
12. 非 root owner 把自己的 block 再写回 root，产生冗余流量。
13. `enablePushBatchMerge=false` 时仍在后台合并多个基础 Tile。
14. 不同连续 Broadcast 复用未清理的 notify/event 累计状态。
15. 为了让测试通过而增加 `aclrtSynchronizeStream`、全局流同步或 Direct fallback。

## 十、验证方案

### 10.1 单元级检查

为连续分块和 Batch 选择逻辑增加可独立验证的测试：

```text
totalBytes：0、4 B、tileSize−4 B、tileSize、tileSize+4 B、
            rankSize*tileSize±4 B、400 MiB+4 B
rankSize：1、2、4、12、16
root：0、中间rank、最后rank
```

检查：

- N 个 owner block 无空洞、无重叠；
- ownerBytes 总和等于 totalBytes；
- 每个 block 的尾 Tile 长度正确；
- owner write 的目标 peer 集合正确；
- 合并关闭时 Batch 不跨越基础 Tile 边界；
- 合并开启时 Batch 只覆盖已 ready 连续数据；
- Batch 不超过 `maxPushBatchBytes`；
- 首 Tile 不因合并开关而等待。

### 10.2 正确性回归矩阵

| 维度 | 用例 |
|---|---|
| 数据大小 | 4 B、512 KiB、4 MiB−4 B、4 MiB、4 MiB+4 B、64 MiB、400 MiB+4 B、512 MiB |
| Rank | 1、2、4、12、16 |
| root | 0、7、最后一个 rank |
| 动态 root | `0→7→0→7`、顺序轮换 |
| 连续次数 | 20、100，关键用例1000次 |
| 合并 | 关闭、8 MiB、16 MiB |
| 检查 | checker、超时、数据错误、kernel失败、notify残留 |

重点回归当前历史高风险用例：

```text
12 rank
root=7
连续100次以上
无aclrtSynchronizeStream
无Direct fallback
无第二轮/后续轮次notify误消费
```

### 10.3 性能 A/B 测试

至少比较：

| 方案 | 配置 | 目的 |
|---|---|---|
| Immediate Tile | merge=false | 每Tile立即Push基线 |
| Opportunistic 8M | merge=true, maxBatch=8 MiB | 推荐主方案 |
| Opportunistic 16M | merge=true, maxBatch=16 MiB | 测试控制开销收益 |
| 旧 block-cyclic receiver pull | 原实现 | 总体对照 |

记录：

- 端到端 Broadcast 延迟；
- 第一个 Seed Tile 完成时间；
- 第一个 owner write 提交时间；
- 全部 Seed Pull 完成时间；
- 最后一个远端 Write 完成时间；
- 每个 owner/Die 的 Write 数量、Event 数量和平均 Batch 大小；
- 4/8/16 MiB Batch 的实际出现比例；
- 两个 Die 的负载是否明显不均；
- Seed Pull 是否被同时发生的扇出 Write 明显拖慢；
- 链路吞吐和 CCU 队列是否出现拥塞。

不得只给单次最好值。至少报告预热后的多次平均值、P50 和波动范围，并确保比较时算法、root、数据量、rank 数和计时口径一致。

## 十一、推荐实施顺序

### 第一步：只完成连续块分配

保持当前 receiver-pull AllGather，先替换 block-cyclic owner 映射，验证：

- 地址计算；
- 动态 root；
- 尾块；
- N 块覆盖；
- 连续执行。

### 第二步：实现非流水 owner write

先在 owner 完整 Seed Pull 后再 Push，完成：

- Write 的 peer 集合；
- 每个 Die 的 Channel drain；
- 跨 Die `OWNER_DONE`；
- root 汇总 `GLOBAL_DONE`。

此阶段用于单独验证 owner-write 正确性和远端完成语义。

### 第三步：打开 Tile 流水，合并关闭

实现：

```text
Tile ready → 立即交给长期Push worker
enablePushBatchMerge=false
```

验证 Seed Pull 与 owner write 能安全重叠，不出现读取未完成 Tile、通知覆盖或最终提前返回。

### 第四步：加入自然合并开关

实现：

```text
enablePushBatchMerge=true
maxPushBatchBytes=8 MiB / 16 MiB
```

只合并已经自然积压的连续 Tile，不改变首 Tile 启动时机。

### 第五步：全矩阵回归与性能选择

完成正确性、连续调用、动态 root 和性能 A/B 测试后，再决定默认使用：

```text
merge=false
或 merge=true + 8 MiB
或 merge=true + 16 MiB
```

不要在没有数据前预设合并一定更快。

## 十二、最终交付要求

修改完成后，AI 必须提交一份简洁报告，包含：

1. 修改的文件及每个文件的职责；
2. 旧数据通路与新数据通路对比；
3. 旧信号通路与新信号通路对比；
4. 连续 owner block 的精确计算公式；
5. `enablePushBatchMerge` 和 `maxPushBatchBytes` 的配置位置；
6. Seed ready、Push submitted、Push completed 的状态推进方式；
7. 两个 Die 如何划分 peer、如何 drain、如何汇合；
8. 为什么不会提前发送、越界、重复写或提前返回；
9. 删除或停用的 receiver-pull AllGather READY/ACK 逻辑；
10. 编译、checker、连续调用和性能测试结果；
11. 尚存风险及下一步建议。

最终实现应满足以下一句话描述：

> Broadcast 数据按 rank 划分为 N 个连续 owner block；非 root owner 始终从 root 主动 Pull 自己的 block，并在每个 Tile ready 后立即通过长期 Push worker 向目标 rank 的最终用户 Buffer 发起 owner Write；若 Push 期间自然积累了多个连续 Tile，则可在布尔开关控制下合并为不超过 8/16 MiB 的 Batch，但绝不主动等待凑批；所有 owner 最终 drain 两个 Die 的全部 peer Channel，经 `OWNER_DONE → GLOBAL_DONE` 后才结束算子。
