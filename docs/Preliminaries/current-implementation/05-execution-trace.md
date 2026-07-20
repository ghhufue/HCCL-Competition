# 16-rank 具体执行跟踪

## 选用配置与可确认范围

采用仓库测试脚本 `scripts/run_hccl_vm_broadcast_perf.sh` 的真实配置：

```text
NP=16
root=7
dataType=fp32
totalBytes=400 MiB + 4 B=419,430,404 B
topology=ascend950_cluster_4_server_competition
```

[确认] 该数据量大于 1 MiB，因此请求 Distributed。若全连接资源检查通过且共同 HCCL Buffer 至少 120 MiB，则：

```text
ownerCount=15
pipelineDepth=2
tileBytes=4 MiB=4,194,304 B
tileCount=101
stripeCount=7
```

[待确认] HCCL Buffer 的运行时 size 不在仓库中。下面用 `tileBytes=4 MiB` 展开真实测试输入；如果运行时 Buffer 更小，代码会缩小 Tile，具体 Tile编号会变化，但每个 stripe 的执行顺序、Window/Slot 公式和 Notify 协议不变。

## owner 与线程映射示例

root=7 时：

```text
ownerIndex 0..6   -> rank 0..6
ownerIndex 7..14  -> rank 8..15
```

以 owner rank 0、最终目标 rank 8 为例：

- [确认] root rank 7 的 Channel按 remoteRank 升序，rank0 Channel 是 `workerIndex=1`。
- [确认] rank0 的 Channel按 remoteRank 升序，root rank7 Channel 是 `workerIndex=7`，所以 `threads[7]` 是 coordinator。
- [确认] rank0 到 rank8 的 Channel 是 `workerIndex=8`。
- [确认] rank8 到 rank0 的 Channel 是 `workerIndex=1`。

## 代码规定的提交顺序

不能为不同 rank 的独立 AICPU Kernel虚构单一墙钟顺序。表中的 `Tn` 表示 Notify 建立的偏序；同一 Thread 上按列出顺序排队，不同 Thread可并行。

- root 的 `ExecuteDistributedRoot` 外层先遍历 Channel/owner，内层对该 owner 按两 stripe 一批提交。源码调用顺序上会先提交 owner rank0 的所有 stripe，再 owner rank1；任务被放到不同 worker 后可并行执行。
- owner 的 `ExecuteDistributedOwner` 外层按 batch；每批严格是 A（接收两个 window并启动 worker）→ B（按 stripe、按 Channel提交 peer 交换）→ C（按 stripe等待 worker并归还 root Slot）。
- 每个 peer worker 在同一 Thread 上按收到的 window start 顺序执行自己的 Channel任务。

依据：文件：`hccl_broadcast_problem_template/op_kernel_aicpu/exec_op.cc`；符号：`ExecuteDistributedRoot`、`ExecuteDistributedOwner`；关键变量：`baseStripe`、`batchEnd`；行号：L636-L690、L779-L834。

## 代表 Tile

| 场景 | global Tile | stripe | ownerIndex/rank | Window | Slot | 用户偏移 | 长度 |
|---|---:|---:|---|---:|---:|---:|---:|
| Tile 0 | 0 | 0 | 0 / rank0 | 0 | 0 | 0 | 4 MiB |
| 第一次 Window 切换 | 15 | 1 | 0 / rank0 | 1 | 15 | 60 MiB | 4 MiB |
| 第一次 Slot 复用 | 30 | 2 | 0 / rank0 | 0 | 0 | 120 MiB | 4 MiB |
| 流水稳定阶段 | 45 | 3 | 0 / rank0 | 1 | 15 | 180 MiB | 4 MiB |
| 最后尾 Tile | 100 | 6 | 10 / rank11 | 0 | 10 | 400 MiB | 4 B |

`Slot` 是分析名称，公式为 `window*15+ownerIndex`。

## Tile 0：首个 Window

| 时刻 | Rank | Thread | Tile | Window | Slot | 当前动作 | 数据位置 | 等待信号 | 发出信号 | Slot 状态 |
|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| T0 | 7 | main -> worker1 | 0 | 0 | 0 | `StartAllWorkers` 启动 worker | 无数据变化 | worker Wait launch 32 | main Record launch 32 | 初始 |
| T0 | 0 | main -> coordinator7/peer workers | 0 | 0 | 0 | `StartAllWorkers` 启动全部 worker | 无数据变化 | worker Wait launch 32 | main Record launch 32 | 初始 |
| T1a | 7 | worker1 | 0 | 0 | 0 | root 用户 Tile复制到 root Slot | `input+0 -> root local slot0` | 无 | 无 | root Slot已装载 |
| T1b | 0 | coordinator7 | 0 | 0 | 0 | `SubmitReceiveTileData` 声明可写 | rank0 local slot0 | 无 | 向 root Record consumed 2 | owner Slot允许 root 写 |
| T2 | 7 | worker1 | 0 | 0 | 0 | Wait 初始许可后 Write/Fence/Record DATA | `root slot0 -> rank0 slot0` | consumed 2 | DATA 0 | owner Slot写入完成依赖已发布 |
| T3 | 0 | coordinator7 | 0 | 0 | 0 | Wait DATA，复制到 owner 用户 Buffer | `rank0 slot0 -> output+0` | DATA 0 | 随后启动 peer worker | own Tile已本地消费但仍供 fanout 读取 |
| T4 | 0 | coordinator7 -> worker8 | 0 | 0 | 0 | `SubmitOwnerStripeStart` 启动 rank8 peer worker | rank0 slot0 | worker8 Wait start 0 | Record start 0 | own Slot只读共享 |
| T5a | 0 | worker8 | 0 | 0 | 0 | 向 rank8 Write own Tile，Fence，Record DATA | `rank0 slot0 -> rank8 slot0` | start 0 | Channel DATA 0 | rank8 Slot含 Tile0 |
| T5b | 8 | worker1 | 0 | 0 | 0 | 与 rank0 对称交换；Wait Tile0 DATA并复制 | `rank8 slot0 -> output+0` | Channel DATA 0 | consumed 2 | rank8 用户 Tile0完成 |
| T6 | 0 | worker8 | 0 | 0 | 0 | Wait rank8 consumed，向 coordinator报告本 window完成 | rank0 slot0 | consumed 2 | worker done `2+(8-1)=9` | rank8不再读该 Tile |
| T7 | 0 | coordinator7 | 0 | 0 | 0 | 等所有 14 个非 root peer worker done | rank0 slot0 | window0 done 2..16（跳过 coordinator对应项） | 向 root Record最终 consumed 2 | 所有 peer不再读 own Slot |
| T8 | 7 | worker1 | 0 | 0 | 0 | root 的批末第二次 Wait consumed | root/rank0 slot0 | consumed 2 | 无 | window0/owner0 可复用 |

[确认] T1a 与 T1b 在不同 rank 上没有代码可确定的先后；T2 的 Wait/Record 形成它们汇合后的跨 rank依赖。

## Tile 15：第一次 Window 切换

| 时刻 | Rank | Thread | Tile | Window | Slot | 当前动作 | 数据位置 | 等待信号 | 发出信号 | Slot 状态 |
|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| U1 | 7 | worker1 | 15 | 1 | 15 | 复制第二 stripe 的 owner0 Tile | `input+60MiB -> root slot15` | 无 | 无 | window1已装载 |
| U2 | 0 | coordinator7 | 15 | 1 | 15 | 初始许可、Wait DATA、LocalCopy | `rank0 slot15 -> output+60MiB` | DATA 1 | 初始 consumed 3；start 1 | window1进入 fanout |
| U3 | 0/8 | peer worker | 15 | 1 | 15 | owner-peer Write/接收/复制 | 两端 slot15 与用户偏移60MiB | DATA 1 | DATA 1、consumed 3 | peer逐个消费 |
| U4 | 0 | coordinator7 | 15 | 1 | 15 | Wait window1 worker done | rank0 slot15 | done 17..31 | 最终 consumed 3 | window1可复用 |
| U5 | 7 | worker1 | 15 | 1 | 15 | batch末 Wait最终 consumed | rank0 slot15 | consumed 3 | 无 | root获知可复用 |

[确认] root 对 stripe0/1 的“发出阶段”都提交后，才进入第二个循环等待两者最终 consumed；owner 也在同一 batch 的 A/B/C 阶段内交错两个 Window。

## Tile 30：第一次复用 Slot 0

| 时刻 | Rank | Thread | Tile | Window | Slot | 当前动作 | 数据位置 | 等待信号 | 发出信号 | Slot 状态 |
|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| V0 | 7/0 | owner0相关线程 | 0 | 0 | 0 | 上一批 T7/T8 已完成 | 旧 Tile0 | 最终 consumed 2、window0 done | 已消费 | Slot 0可覆盖 |
| V1 | 7 | worker1 | 30 | 0 | 0 | 用 Tile30覆盖 root Slot0 | `input+120MiB -> root slot0` | 同线程上 T8在前 | 无 | root Slot0含 Tile30 |
| V2 | 0 | coordinator7 | 30 | 0 | 0 | 新一轮初始许可/接收/本地复制 | `rank0 slot0 -> output+120MiB` | DATA 0 | consumed 2、start 0 | owner Slot0含 Tile30 |
| V3 | 0/peer | peer workers | 30 | 0 | 0 | 按 Tile0相同协议 fanout | slot0 -> 各 peer slot0 | start/DATA/consumed 0/2 | done window0 | peer完成 Tile30 |
| V4 | 0 -> 7 | coordinator7/root worker1 | 30 | 0 | 0 | 全 peer done 后归还 root | rank0 slot0 | done window0、consumed 2 | 最终 consumed 2 | Slot0再次可复用 |

## Tile 45：稳定流水阶段

| 时刻 | Rank | Thread | Tile | Window | Slot | 当前动作 | 数据位置 | 等待信号 | 发出信号 | Slot 状态 |
|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| W1 | 7 | worker1 | 45 | 1 | 15 | 在第二批的 window1提交 scatter | `input+180MiB -> rank0 slot15` | 初始 consumed 3 | DATA 1 | 写入 |
| W2 | 0 | coordinator7 | 45 | 1 | 15 | 接收、本地复制并启动14个 peer worker | `slot15 -> output+180MiB` | DATA 1 | start 1 | fanout中 |
| W3 | 0/所有 peer | 各 Channel worker | 45 | 1 | 15 | pairwise交换与消费确认 | slot15/用户偏移180MiB | DATA 1、consumed 3 | done window1 | 各 Channel独立推进 |
| W4 | 0/7 | coordinator/root worker | 45 | 1 | 15 | 聚合 done 并最终归还 | slot15 | done 17..31、consumed 3 | 最终 consumed 3 | 可供 stripe5后续复用 |

## Tile 100：最后 4B 尾块

| 时刻 | Rank | Thread | Tile | Window | Slot | 当前动作 | 数据位置 | 等待信号 | 发出信号 | Slot 状态 |
|---:|---:|---|---:|---:|---:|---|---|---|---|---|
| X1 | 7 | rank11 Channel worker11 | 100 | 0 | 10 | 只复制 4B 到 root Slot10 | `input+400MiB -> root slot10` | 无 | 无 | 含4B有效数据 |
| X2 | 11 | root-Channel coordinator | 100 | 0 | 10 | 初始许可、Wait DATA并只复制 4B | `rank11 slot10 -> output+400MiB` | DATA 0 | consumed 2、peer start 0 | owner本地尾块完成 |
| X3 | 7 | worker11 | 100 | 0 | 10 | Write/Fence/Record 的长度参数是4 | `root slot10 -> rank11 slot10` | consumed 2 | DATA 0 | 无越界扩写 |
| X4 | 11 -> 8 | peer worker | 100 | 0 | 10 | 若对端本 stripe的 own Tile无效，rank11走仅发送半边 | `rank11 slot10 -> rank8 slot10` | start 0 | DATA 0；Wait consumed 2 | rank8收到4B |
| X5 | 8 | rank11 Channel worker | 100 | 0 | 10 | 对称地走仅接收半边，复制4B | `rank8 slot10 -> output+400MiB` | DATA 0 | consumed 2、worker done | 用户尾4B完成 |
| X6 | 11 | coordinator | 100 | 0 | 10 | 等所有 peer worker（含无数据分支）done，向 root最终归还 | rank11 slot10 | window0 done | 最终 consumed 2 | Slot与Notify可复用 |

[确认] 最后 stripe 的 global Tile 90..100 有效、101..104 无效。每一对 owner根据双方有效性选择双向/单向协议；无效 Tile仍会参与 Thread start/done 控制，但不会产生 Channel DATA/consumed Wait。
