你现在位于一个 HCCL 自定义 Broadcast 算子工程中。请直接检查仓库、修改代码、编译验证，不要只输出设计说明。

# 一、任务目标

实现基于 **AICPU + TS 通信引擎**的 HCCL Broadcast 算子：

```cpp
HcclResult HcclBroadcast(
    void *buf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    HcclComm comm,
    aclrtStream stream);
```

算子语义：

* root rank 的 `buf` 是源数据。
* 非 root rank 的 `buf` 是接收数据。
* 操作结束后，所有 rank 的 `buf` 必须与 root 的原始数据完全一致。
* root 可能是任意合法 rank，不得假设 root 为 0。
* 当前数据类型至少覆盖 `HCCL_DATA_TYPE_FP32`。
* 必须正确处理：

  * 4B
  * 512KB
  * 512MB
  * 400MB + 4B
* 评测拓扑为 2 个 Server、每个 Server 8 个 NPU：

  * Server 内 8 卡 Full-Mesh。
  * Server 间通过 Clos 网络连接。
  * 每张卡的 Clos 带宽约等于 Server 内单条直连链路带宽的 8 倍。

首要目标是功能正确，其次才是性能。最终代码必须保留清晰的模块边界，方便后续替换算法、调整线程数、Tile 大小、Pipeline 深度和通知协议。

---

# 二、硬性修改约束

只允许修改以下三个文件：

```text
include/custom.h
op_host/broadcast.cc
op_kernel_aicpu/exec_op.cc
```

不得修改：

```text
include/common.h
include/binary_stream.h
op_kernel_aicpu/exec_op.h
op_kernel_aicpu/aicpu_kernel.cc
op_host/launch_aicpu_kernel.cc
任何 CMakeLists.txt
build.sh
```

不得新增源文件或头文件。

所有辅助结构、辅助函数和算法实现必须放在上述三个允许修改的文件中：

* 通用序列化资源结构放在 `custom.h`。
* Host 资源规划和申请函数放在 `broadcast.cc` 的匿名命名空间。
* Device 算法规划和任务编排函数放在 `exec_op.cc` 的匿名命名空间。

保留现有版权头和代码风格。

---

# 三、最重要的工作原则

## 3.1 不得猜测 HCCL/HCOMM API

当前工程依赖 CANN 9.1.0 附近版本，但不同环境的函数签名可能不同。

在写任何 Channel 或 HCOMM 原语代码前，先在本机 CANN 安装目录中核对真实声明。

执行类似命令：

```bash
echo "${ASCEND_HOME_PATH}"

grep -R "HcclRankGraphGetLinks" \
    "${ASCEND_HOME_PATH}/include" \
    "${ASCEND_HOME_PATH}/pkg_inc" 2>/dev/null

grep -R "HcclChannelDescInit" \
    "${ASCEND_HOME_PATH}/include" \
    "${ASCEND_HOME_PATH}/pkg_inc" 2>/dev/null

grep -R "HcclChannelAcquire" \
    "${ASCEND_HOME_PATH}/include" \
    "${ASCEND_HOME_PATH}/pkg_inc" 2>/dev/null

grep -R "HcclChannelGet" \
    "${ASCEND_HOME_PATH}/include" \
    "${ASCEND_HOME_PATH}/pkg_inc" 2>/dev/null

grep -R "Hcomm.*OnThread" \
    "${ASCEND_HOME_PATH}/include" \
    "${ASCEND_HOME_PATH}/pkg_inc" 2>/dev/null
```

重点确认：

* `HcclRankGraphGetLinks`
* `HcclChannelDescInit`
* `HcclChannelAcquire`
* 获取远端 HCCL Buffer 的接口
* Channel Notify Record/Wait 接口
* Thread Notify Record/Wait 接口
* 本地内存复制接口
* Channel Write/Read 接口
* Fence、Flush 或等待通信完成的接口
* `ChannelHandle` 是否支持双向通信
* Channel Notify 编号是否区分本端和远端方向
* 同一 Channel 上同时收发的语义
* Write 返回代表“任务已入队”还是“源 Buffer 已不再被读取”

不得因为函数名称看起来合理就直接调用。

如果计划使用的接口在当前版本不存在：

1. 搜索当前版本等价接口；
2. 根据真实声明调整实现；
3. 不要自行伪造 wrapper；
4. 在最终报告中说明替换关系。

---

# 四、先检查当前模板

开始修改前，完整阅读：

```text
include/custom.h
include/common.h
include/binary_stream.h
op_host/broadcast.cc
op_host/launch_aicpu_kernel.cc
op_kernel_aicpu/aicpu_kernel.cc
op_kernel_aicpu/exec_op.h
op_kernel_aicpu/exec_op.cc
```

特别注意以下已有流程，不得破坏：

```text
Host stream
    → CPU_TS Thread Notify Record
    → Launch AICPU Kernel
    → AICPU Thread Notify Wait
    → ExecOp
    → AICPU Thread Notify Record
    → Host Thread Notify Wait
```

`aicpu_kernel.cc` 已经负责：

* 反序列化 `AlgResourceCtx`
* `HcommBatchModeStart`
* 等待 Host 通知
* 调用 `ExecOp`
* 通知 Host
* `HcommBatchModeEnd`

因此 `ExecOp` 不得重复调用 BatchMode Start/End，也不得破坏主线程和 Host stream 之间已有的 Notify 0。

---

# 五、总体实现策略

实现两套算法，并通过统一的执行计划选择。

```text
AlgorithmKind::DIRECT_FANOUT
AlgorithmKind::DISTRIBUTED_SCATTER_FANOUT
```

初始选择策略：

```text
count == 0：
    直接成功，不下发数据任务

bytes <= 1MiB：
    DIRECT_FANOUT

bytes > 1MiB：
    DISTRIBUTED_SCATTER_FANOUT
```

其中：

* 4B 默认使用 Direct。
* 512KB 默认使用 Direct。
* 512MB 使用 Distributed。
* 400MB + 4B 使用 Distributed。

阈值必须集中定义，不能散落在多个函数里。后续应能通过修改一个常量切换 512KB 的算法。

如果 Distributed 算法的资源条件不满足，例如：

* rank 数不足；
* Channel 不完整；
* Thread 数不足；
* HCCL Buffer 太小；
* 当前 HCOMM API 不支持必要的安全同步；

则必须自动退化到**分块 Direct Broadcast**，不能直接返回成功但不通信。

---

# 六、可扩展代码结构

## 6.1 `custom.h` 的设计目标

`custom.h` 中的数据只保存静态资源，不保存某一次调用特有的数据长度、root 或用户 Buffer 地址。

原因：

* Engine Context 会被后续 Broadcast 调用复用。
* root、count 和 buf 每次调用都可能变化。
* Channel 资源应与 root 无关，便于同一通信域上复用。

建议添加以下概念，具体类型根据真实 API 调整：

```cpp
enum class ResourceLayoutVersion : uint32_t {
    VERSION_1 = 1,
};

struct ChannelInfo {
    uint32_t remoteRank;
    uint32_t workerIndex;
    uint32_t notifyNum;
    ChannelHandle handle;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    uint32_t layoutVersion;
    uint32_t rankSize;
    uint32_t workerCount;
    uint32_t notifyNumPerThread;
    uint32_t channelNotifyNum;
    uint32_t pipelineDepth;

    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

    std::vector<char> Serialize();
    void DeSerialize(std::vector<char> &data);
};
```

### 必须满足的资源不变量

* `threads[0]` 是主控制线程。
* `aicpuThread == threads[0]`。
* 每个远端 rank 恰好对应一个 `ChannelInfo`。
* `channels` 按 `remoteRank` 升序排列，禁止依赖 Channel 申请返回顺序。
* `ChannelInfo::workerIndex` 指向负责该 peer 的工作线程。
* root 变化时，不重新申请 Channel。
* 不在序列化结构中存储 STL 迭代器、引用、Lambda、Host 临时指针。
* 新增字段后同步更新 Serialize 和 DeSerialize，顺序必须完全一致。
* 资源结构添加版本号，避免后续布局变化难以诊断。

如果 `BinaryStream` 对某个新增类型无法安全序列化，使用固定宽度整数或简单 POD 字段表示，不修改 `binary_stream.h`。

---

## 6.2 `broadcast.cc` 的内部模块

在匿名命名空间中拆分为小函数，避免所有逻辑堆在 `HcclBroadcast` 中。

建议结构：

```cpp
namespace {

constexpr char kResourceTag[] = "hccl_custom_broadcast_v1";

struct ResourcePlan {
    uint32_t threadNum;
    uint32_t notifyNumPerThread;
    uint32_t channelNotifyNum;
    uint32_t pipelineDepth;
};

HcclResult ValidateBroadcastParam(...);

HcclResult BuildResourcePlan(
    const OpParam &param,
    const CommBuffer &localBuffer,
    ResourcePlan &plan);

HcclResult QueryBestLinkToPeer(
    HcclComm comm,
    uint32_t localRank,
    uint32_t remoteRank,
    ...);

HcclResult AcquireThreads(
    HcclComm comm,
    const ResourcePlan &plan,
    AlgResourceCtx &resource);

HcclResult AcquireChannels(
    HcclComm comm,
    const OpParam &param,
    const ResourcePlan &plan,
    AlgResourceCtx &resource);

HcclResult CreateAndStoreEngineContext(...);

}
```

函数名可根据实际代码调整，但必须维持这些职责边界。

---

## 6.3 `exec_op.cc` 的内部模块

在匿名命名空间中实现：

```cpp
enum class AlgorithmKind : uint32_t {
    DIRECT_FANOUT,
    DISTRIBUTED_SCATTER_FANOUT,
};

struct ExecutionPlan {
    AlgorithmKind algorithm;
    uint64_t totalBytes;
    uint64_t tileBytes;
    uint64_t stripeCount;
    uint32_t ownerCount;
    uint32_t pipelineDepth;
};

struct TileDesc {
    uint64_t offset;
    uint64_t bytes;
    uint32_t ownerIndex;
    uint32_t ownerRank;
    uint32_t windowIndex;
    bool valid;
};
```

建议辅助函数：

```cpp
HcclResult ValidateExecutionContext(...);

HcclResult BuildExecutionPlan(
    const OpParam &param,
    const AlgResourceCtx &resource,
    ExecutionPlan &plan);

const ChannelInfo *FindChannelByRemoteRank(...);

ThreadHandle FindWorkerByRemoteRank(...);

uint32_t OwnerIndexToRank(...);

bool RankToOwnerIndex(...);

TileDesc MakeTileDesc(...);

uint64_t ComputeSlotOffset(...);

HcclResult ExecuteDirectFanout(...);

HcclResult ExecuteDistributedScatterFanout(...);

HcclResult WaitAllWorkers(...);
```

不得在 `ExecOp` 中直接写数百行通信逻辑。

`ExecOp` 只负责：

```cpp
验证参数
→ 创建 ExecutionPlan
→ 打印一次算法摘要
→ 分派到对应算法
→ 等待所有工作线程结束
→ 返回
```

---

# 七、参数与整数安全

在 Host 和 Device 侧都要处理以下情况：

## 7.1 数据类型

当前只支持：

```cpp
HCCL_DATA_TYPE_FP32
```

通过 `SIZE_TABLE` 查询元素大小，不得写死所有场景都是 4 字节。

未知类型返回参数错误。

## 7.2 root

必须检查：

```cpp
root < rankSize
```

不得假设 root 为 0。

## 7.3 count

正确处理：

```text
count == 0
count == 1
超大 count 导致 count * elementSize 溢出
```

计算字节数前检查：

```cpp
count <= UINT64_MAX / elementSize
```

## 7.4 指针运算

禁止对 `void *` 直接做加法。

使用：

```cpp
auto *base = static_cast<uint8_t *>(ptr);
void *address = base + offset;
```

或者安全的 `uintptr_t` 转换。

任何地址偏移都必须先证明：

```text
offset <= bufferSize
bytes <= bufferSize - offset
```

## 7.5 尾块

不能假设总数据能被：

* rank 数；
* owner 数；
* Tile 大小；
* Pipeline 深度

整除。

最后一个 Tile 统一使用：

```cpp
tileBytes = std::min(preferredTileBytes, totalBytes - offset);
```

`400MB + 4B` 的最后 4 字节必须被正常传输。

---

# 八、Host 侧资源申请方案

## 8.1 Resource Tag

将 Tag 设计为包含资源布局版本，例如：

```text
hccl_custom_broadcast_v1
```

不要把 root、count 或数据大小放入 Tag，否则每种输入都会重复申请资源。

后续修改序列化布局时，只需将版本升级为 `v2`，避免旧 Engine Context 被错误反序列化。

## 8.2 Thread 设计

优先方案：

```text
1 个主线程
+
每个 remote rank 1 个 worker
```

对于 16 rank：

```text
threadNum = 16
workerCount = 15
```

映射：

```text
threads[0]：主线程
threads[1...]：remote rank worker
```

但是不得盲目假设当前环境允许申请 16 个 Thread。

先检查头文件、示例和资源限制。

如果 16 个 Thread 申请失败且存在明确的资源释放或重试方式，可以实现降级：

```text
15 worker
→ 8 worker
→ 4 worker
→ 1 worker
```

如果 API 不支持安全地释放部分申请资源，不要进行会泄漏资源的反复试探。此时采用文档推荐的合法数量，并通过多个 peer 映射到同一 worker 实现分组调度。

因此，代码结构必须支持：

```text
channel 数量 != worker 数量
```

`ChannelInfo::workerIndex` 不能简单等于 Channel 下标。

## 8.3 Thread Notify

主线程已有 Notify 0 用于 Host/Device 同步，不能复用为内部 worker 同步。

为内部同步定义具名索引，例如：

```cpp
enum class ThreadNotifyIndex : uint32_t {
    WORKER_START = 1,
    WORKER_DONE_BASE = 2,
};
```

具体索引布局应集中定义，并明确：

* 哪个线程 Record；
* 哪个线程 Wait；
* Notify 能否重复使用；
* Pipeline 重用前需要满足什么条件。

不得在代码中出现大量无语义的 `0`、`1`、`2`。

## 8.4 Channel

每个 rank 申请与其他所有 rank 的 Channel：

```text
channelCount = rankSize - 1
```

不能只申请 root 相关 Channel，因为大包算法中非 root rank 之间需要互相传播分片。

Channel 申请步骤：

1. 对每个 `remoteRank != myRank` 查询可用 Link。
2. 根据真实 Link 信息初始化 Channel 描述。
3. 设置真实协议和必要属性。
4. 申请 Channel。
5. 获取 Channel Handle。
6. 获取对应远端 HCCL Buffer 地址及大小。
7. 将结果保存到 `ChannelInfo`。
8. 最终按 `remoteRank` 排序。

不得写死所有链路都是 HCCS。

Server 内和 Server 间可能返回不同的 Link 类型或协议，应使用拓扑查询结果。

如果同一 peer 返回多个 Link：

* 阅读结构体字段和官方示例；
* 选择符合 AICPU_TS、数据传输和当前拓扑要求的 Link；
* 将选择策略封装到 `QueryBestLinkToPeer`；
* 写清选择依据；
* 不要在主循环中散落判断。

## 8.5 Channel Notify

为了支持双缓冲，设计具名通知：

```cpp
enum class ChannelNotifyIndex : uint32_t {
    DATA_READY_SLOT_0 = 0,
    DATA_READY_SLOT_1 = 1,
    SLOT_CONSUMED_0 = 2,
    SLOT_CONSUMED_1 = 3,
};
```

若实际 API 的通知语义不同，根据真实头文件调整，但必须保持下面两个逻辑事件：

```text
DATA_READY：
发送方已将本轮数据写入接收方的 HCCL Buffer 槽位。

SLOT_CONSUMED：
接收方已将该槽位的数据复制到最终用户 Buffer，
发送方下一轮可以覆盖这个远端槽位。
```

如果采用 Pipeline 深度大于 2，通知数量和索引必须通过公式计算，不能继续写死 4。

## 8.6 HCCL Buffer

获取本地 HCCL Buffer：

```cpp
HcclGetHcclBuffer(...)
```

保存：

```cpp
CommBuffer{addr, size}
```

不要假设一定正好是 400MB。

算法执行时根据：

* 本地 Buffer 大小；
* 所有远端 Buffer 大小；
* owner 数量；
* Pipeline 深度；
* 对齐要求

动态计算安全 Tile 大小。

---

# 九、统一执行计划

在 `exec_op.cc` 中集中配置调优参数：

```cpp
struct AlgorithmConfig {
    static constexpr uint64_t kDirectThresholdBytes = 1ULL << 20;
    static constexpr uint64_t kPreferredTileBytes = 4ULL << 20;
    static constexpr uint64_t kMinimumTileBytes = 4ULL << 10;
    static constexpr uint32_t kPreferredPipelineDepth = 2;
    static constexpr uint64_t kAddressAlignment = 32;
};
```

具体对齐值必须根据 API 要求确认。如果 API 没有额外对齐要求，不要虚构限制，但 Tile 至少保持 `sizeof(float)` 对齐。

`BuildExecutionPlan` 应完成：

1. 计算 `totalBytes`。
2. 判断算法类型。
3. 计算 `ownerCount = rankSize - 1`。
4. 检查 Channel 是否完整。
5. 计算所有相关 HCCL Buffer 的最小可用大小。
6. 计算 Pipeline 深度。
7. 计算单 owner 单窗口 Tile 大小。
8. 向下对齐 Tile。
9. 检查 Tile 不为 0。
10. 计算 stripe 数量。
11. 不满足分布式算法条件时退化为 Direct。

执行计划计算必须是纯逻辑，不下发 HCOMM 任务，便于以后独立检查和调整。

---

# 十、小包与保底算法：分块 Direct Fanout

Direct 算法不能只支持小包，它还必须是大包分布式算法失败时的正确性保底路径。

## 10.1 语义

root 将每个数据块发送给所有非 root rank：

```text
root → rank 1
root → rank 2
...
root → rank p-1
```

## 10.2 必须分块

即使是 Direct，也不能假设输入能一次放入 HCCL Buffer。

对总数据循环：

```text
offset = 0

while offset < totalBytes:
    currentBytes = min(chunkCapacity, totalBytes - offset)
    发送当前块
    确认所有 peer 已消费
    offset += currentBytes
```

## 10.3 Buffer 布局

至少采用双缓冲或严格的消费确认。

禁止出现：

```text
root 覆盖本地源槽位
但前一次 Write 仍可能读取该槽位
```

也禁止出现：

```text
root 覆盖某个远端槽位
但接收方尚未把旧数据复制到用户 buf
```

## 10.4 root 流程

对于每个 chunk：

1. 将用户 Buffer 当前块复制到 root 本地 HCCL 槽位。
2. 确保本地 Copy 完成后再启动 peer Write。
3. 所有 peer worker 把同一块写入各自远端 HCCL 槽位。
4. 每个 peer 收到后复制到自己的最终用户 Buffer。
5. 每个 peer 返回 `SLOT_CONSUMED`。
6. root 确认：

   * 所有发送已不再读取本地源槽位；
   * 所有远端槽位均已消费；
7. 才能复用该窗口。

需要根据真实 HCOMM API 区分：

* 通信任务入队；
* 远端写完成；
* 本地源 Buffer 可复用；
* 远端目标 Buffer 可复用。

不得将这些事件错误地视为同一个事件。

## 10.5 非 root 流程

对于每个 chunk：

1. 等待来自 root 的 `DATA_READY`。
2. 从本地 HCCL Buffer 槽位复制到：

   ```cpp
   outputPtr + offset
   ```
3. 确认本地 Copy 已正确建立依赖。
4. 向 root Record `SLOT_CONSUMED`。
5. 进入下一轮。

root 自己不需要把数据从 `inputPtr` 再复制回 `outputPtr`，因为二者是同一个 `buf`。

---

# 十一、大包算法：Distributed Scatter + Fanout

## 11.1 Owner 定义

root 不作为 owner。

```text
ownerCount = rankSize - 1
```

对任意 root，定义稳定映射：

```cpp
uint32_t OwnerIndexToRank(uint32_t ownerIndex, uint32_t root)
{
    return ownerIndex < root ? ownerIndex : ownerIndex + 1;
}
```

反向映射：

```cpp
bool RankToOwnerIndex(
    uint32_t rank,
    uint32_t root,
    uint32_t &ownerIndex)
{
    if (rank == root) {
        return false;
    }

    ownerIndex = rank < root ? rank : rank - 1;
    return true;
}
```

不得在算法中写 root 0 或 root 7 的特判。

## 11.2 Block-cyclic Tile 分配

将数据划分为连续 Tile，并按 owner 轮转分配：

```text
tile 0  → owner 0
tile 1  → owner 1
...
tile 14 → owner 14
tile 15 → owner 0
...
```

公式：

```cpp
globalTileIndex = stripeIndex * ownerCount + ownerIndex;
offset = globalTileIndex * tileBytes;
```

若：

```cpp
offset >= totalBytes
```

则该 owner 在当前 stripe 没有有效 Tile。

否则：

```cpp
bytes = min(tileBytes, totalBytes - offset);
```

所有 rank 必须使用完全相同的纯函数计算：

* owner；
* offset；
* bytes；
* window；
* 本地槽位；
* 远端槽位。

不得额外发送 Tile 元数据。

## 11.3 本地 HCCL Buffer 布局

每个 rank 的本地 HCCL Buffer 按：

```text
pipeline window
    × owner slot
        × tile capacity
```

组织。

地址公式：

```cpp
slotOffset =
    (windowIndex * ownerCount + ownerIndex) * tileCapacity;
```

必须检查：

```text
slotOffset + currentBytes <= localBuffer.size
```

每个 owner 使用独立槽位，因此多个 rank 可以并发写入同一接收 rank 的不同区域。

## 11.4 第一阶段：root Scatter

对于每个 stripe：

* root 向每个有效 owner 发送其负责的 Tile。
* root 只发送一份完整数据的总量，而不是将完整数据复制 15 次。
* 每个 owner 的数据写入 owner rank 本地 HCCL Buffer 中属于该 owner 的槽位。
* owner 收到 Tile 后才能开始传播。

root 对每个 owner 的 worker 负责：

```text
从 root 用户 buf 对应 offset 取数据
→ 安全写入 owner 的远端 owner-slot
→ DATA_READY
→ 等待 owner 在传播完成后返回 SLOT_CONSUMED
→ 复用该远端窗口
```

如果 API 要求通过 root 本地 HCCL Buffer 中转，则使用该 worker 对应的本地槽位，并明确等待本地源槽位可复用。

如果 API 支持安全的用户 Buffer 到远端通信操作，也必须先核对 API 的内存类型和生命周期要求，不能自行假定支持。

## 11.5 第二阶段：owner 分布式复制

每个非 root rank 是一个 owner。

收到自己的 Tile 后：

1. 将自己的 Tile 复制到最终用户 Buffer 对应位置。
2. 通知与其他非 root peer 对应的 worker：

   ```text
   own tile ready
   ```
3. 每个 peer worker 将 owner 的 Tile 写给该 peer。
4. 每个 peer worker同时等待该 peer 发来的 Tile。
5. 收到 peer Tile 后，从本地 HCCL Buffer 中 peer 对应槽位复制到最终用户 Buffer。
6. 向 peer 返回远端槽位已消费的 ACK。
7. 本 owner 等所有向外发送均已不再读取自己的本地 owner 槽位。
8. owner 才向 root 返回 `SLOT_CONSUMED`，允许 root 写入下一轮同窗口数据。

## 11.6 Pairwise peer worker 的死锁规避

两个非 root rank 之间可能同时互发。

禁止双方都执行：

```text
先 Wait 对方数据
再 Send 自己数据
```

这会形成死锁。

在确认 API 支持同一 Channel 双向收发后，采用统一顺序：

```text
等待自己的 owner Tile Ready
→ 提交自己的 Write
→ 再等待对方 DATA_READY
→ Copy 对方 Tile
→ 返回对方 SLOT_CONSUMED
→ 等待或 Fence 本地 Write 源读取完成
→ 通知本地 owner 当前 peer 发送完成
```

所有 rank 使用同一顺序。

如果当前 Channel API 不支持安全的双向并发：

* 根据 rank 大小关系制定确定性方向；
* 或使用 API 支持的独立方向资源；
* 或退化为分阶段 Pairwise 调度；
* 不得保留潜在死锁实现。

## 11.7 Pipeline

初始 Pipeline 深度使用 2：

```text
slot 0：处理偶数 stripe
slot 1：处理奇数 stripe
```

目标流水：

```text
root 分发 stripe k+1
+
owner 扩散 stripe k
```

复用某个窗口前必须同时满足：

### root → owner Channel

owner 已经：

* 把 Tile 复制到最终用户 Buffer；
* 完成向其他 peer 的所有源读取；
* 向 root 返回 consumed。

### owner → peer Channel

peer 已经：

* 收到 Tile；
* 复制到最终用户 Buffer；
* 向 owner 返回 consumed。

### owner 本地源槽位

所有向外 Write 均已通过真实 Fence/Completion 语义确认，不再读取该源槽位。

不得只依赖固定延迟或认为提交下一条任务就意味着上一条已完成。

---

# 十二、主线程与 Worker 同步

`threads[0]` 是主线程，也是 `resCtx.aicpuThread`。

AICPU Kernel 已经在主线程 Notify 0 上等待 Host。因此内部算法不得覆盖或错误消费该 Notify。

建议流程：

## 12.1 开始

主线程为每个 worker Record 内部启动 Notify：

```text
main thread → worker start
```

worker 的第一条任务是等待启动 Notify。

如果不同 worker 分配到同一 Thread，则不能重复构造互相冲突的启动依赖，要按实际映射合并。

## 12.2 结束

每个实际 worker 完成全部任务后：

```text
worker → main thread done
```

主线程必须等待所有实际 worker 完成。

只有主线程完成等待后，`ExecOp` 才能返回。

否则 `aicpu_kernel.cc` 会过早通知 Host，导致用户 stream 继续执行时 Broadcast 仍未完成。

## 12.3 无工作 Worker

某些小输入或最后一个 stripe 中，部分 owner 没有有效 Tile。

仍然必须保证对应 worker 的控制流一致：

* 不发送无效长度；
* 不等待一个永远不会产生的 Notify；
* 最终仍能通知主线程完成。

---

# 十三、错误处理与日志

## 13.1 错误返回

每一个 HCCL/HCOMM 调用都要检查返回值。

优先使用工程已有的：

```cpp
CHK_RET(...)
CHK_PTR_NULL(...)
```

如果宏不适合 Device 辅助函数，编写轻量 helper，但不能吞掉错误。

失败时日志至少包含：

* 当前 rank；
* root；
* remote rank；
* algorithm；
* stripe；
* owner；
* window；
* offset；
* bytes；
* API 名称；
* 返回码。

## 13.2 日志数量

禁止在每个 Tile、每个 peer、每个循环中无条件打印 INFO，会严重影响性能。

默认只打印：

```text
每次调用一条执行计划摘要
资源首次创建一条摘要
算法退化一条 WARNING
错误路径的详细日志
```

Tile 级日志只允许在显式 Debug 常量开启时打印。

建议摘要：

```text
rank=...
root=...
rankSize=...
bytes=...
algorithm=...
tileBytes=...
pipelineDepth=...
workerCount=...
channelCount=...
localBufferSize=...
```

---

# 十四、可扩展性要求

代码必须满足以下后续迭代场景：

## 14.1 调整算法阈值

只修改一个配置值即可将 512KB 切换到 Distributed。

## 14.2 调整 Tile 大小

只修改：

```cpp
kPreferredTileBytes
```

不需要改地址公式或循环结构。

## 14.3 调整 Pipeline 深度

代码不能把 Pipeline 深度 2 散落写死。

即使当前通知资源只支持深度 2，也要把：

* 计划字段；
* 地址公式；
* window 计算

写成通用形式：

```cpp
window = stripe % pipelineDepth;
```

通知映射可以暂时校验只允许 2，但要在一个位置明确限制。

## 14.4 Worker 数减少

允许多个 Channel 共享一个 worker。

算法不得假设：

```text
workerIndex == channelIndex + 1
```

## 14.5 rank 数变化

虽然比赛固定 16 rank，但大部分代码应基于：

```cpp
param.rankSize
```

运行。

只在拓扑优化策略中针对 16 rank 做优化，不在正确性算法里硬编码 16、15、7、8。

## 14.6 添加新算法

后续应能增加：

```cpp
AlgorithmKind::HIERARCHICAL
AlgorithmKind::TREE
AlgorithmKind::RING
```

而不重写资源查询、参数校验和公共 Tile 逻辑。

因此算法选择必须集中在：

```cpp
BuildExecutionPlan
ExecOp
```

---

# 十五、分阶段实现与验证

严格按以下顺序工作，不要一次性写完后才首次编译。

## 阶段 A：仓库与 API 检查

1. 阅读全部模板。
2. 检查 git 状态。
3. 查找真实 HCCL/HCOMM API。
4. 列出准备使用的 API 和真实签名。
5. 确认只修改三个允许文件。

不要修改代码前就假定 API。

## 阶段 B：资源结构和 Host 资源申请

实现：

* `custom.h` 版本化资源结构。
* Thread 资源申请。
* 全 peer Channel 资源申请。
* 远端 HCCL Buffer 获取。
* 序列化和反序列化一致。
* Engine Context 复用。
* root 无关资源布局。

执行：

```bash
bash build.sh --format
bash build.sh --debug
```

解决所有：

```text
-Werror
-Wall
序列化编译错误
API 参数错误
未使用变量
符号类型不匹配
```

## 阶段 C：分块 Direct 基线

先让所有数据规模都走 Direct。

确保设计上支持：

* root 0；
* root 7；
* 任意合法 root；
* 大于 HCCL Buffer 的输入；
* 400MB + 4B 尾块；
* count 0；
* 多次调用 Engine Context 复用。

再次编译。

如果有可运行环境，先完成功能测试再继续优化。

## 阶段 D：加入 Distributed 算法

保留 Direct 作为独立函数和 fallback，不要覆盖或删除。

实现：

* owner 映射；
* Block-cyclic Tile；
* Buffer slot 布局；
* root Scatter；
* owner Fanout；
* peer 接收；
* 双缓冲；
* consumed ACK；
* worker 完成同步；
* root 任意值。

再次编译。

## 阶段 E：静态死锁审查

逐条检查每一种 Wait 是否存在匹配 Record。

制作内部检查表，不需要新建文件：

```text
Wait 位置
Record 位置
Record 所在线程
Wait 所在线程
使用的 Notify
是否可能因无效 Tile 跳过 Record
是否可能循环次数不一致
是否可能双方先 Wait
```

重点检查：

* root 为 0；
* root 为 7；
* 最后一个 stripe 只有少数有效 owner；
* totalBytes 小于一个 Tile；
* Pipeline 最后两个窗口 Drain；
* Direct fallback；
* 多个 Channel 共用一个 worker。

## 阶段 F：最终构建检查

执行：

```bash
bash build.sh --format
bash build.sh --debug
bash build.sh
git diff --check
git status --short
git diff -- include/custom.h op_host/broadcast.cc op_kernel_aicpu/exec_op.cc
```

确保：

* 只有三个允许文件发生变化；
* Debug 和 Release 均通过；
* 无格式问题；
* 无尾随空格；
* 无临时代码；
* 无未使用 Debug 日志；
* 无硬编码绝对路径；
* 无修改系统 CANN 文件；
* 无生成文件被加入仓库。

---

# 十六、性能调优接口

先不要为了追求性能破坏结构。

将以下参数集中：

```cpp
kDirectThresholdBytes
kPreferredTileBytes
kMinimumTileBytes
kPreferredPipelineDepth
kMaxWorkerCount
```

最终报告中列出建议测试矩阵：

```text
Direct threshold:
256KB
1MB
2MB

Tile:
1MB
2MB
4MB
8MB

Pipeline:
1
2

Worker:
1
4
8
15
```

当前默认建议：

```text
Direct threshold = 1MB
Tile = 4MB
Pipeline = 2
Worker = 能合法申请的最大并行数
```

但必须以真实评测或仿真数据为准，不得在没有测试数据时声称已经达到理论上限。

---

# 十七、禁止事项

禁止：

* 只输出伪代码而不修改工程。
* 修改三个允许文件以外的文件。
* 假设 root 为 0。
* 针对 root 7 写单独算法。
* 假设 rankSize 永远为 16而不做边界检查。
* 假设 HCCL Buffer 一定是 400MB。
* 假设输入可以被 15 或 Tile 大小整除。
* 调用不存在的 HCOMM API。
* 忽略函数返回值。
* 使用固定 sleep 代替同步。
* 用全局 Barrier 解决所有依赖。
* 在每个 Tile 后做全局同步。
* 在未确认源 Buffer 可复用时覆盖它。
* 在未收到远端 consumed 时覆盖远端槽位。
* 双方都先 Wait 再 Send。
* 让 `ExecOp` 在 worker 完成前返回。
* 删除 Direct fallback。
* 将所有算法逻辑写在一个大函数中。
* 在没有运行数据时宣称性能最优。
* 提交、推送或创建 PR，除非用户另外明确要求。

---

# 十八、最终输出格式

完成修改后，给出以下报告。

## 1. 修改摘要

说明三个文件分别修改了什么。

## 2. 实际使用的 API

列出从本机头文件确认过的：

```text
API 名称
声明所在头文件
用途
关键完成语义
```

特别说明：

* Write 完成语义；
* Fence 语义；
* Channel Notify 方向；
* Channel 是否支持双向并发。

## 3. 资源布局

说明：

```text
Thread 数量
worker 映射
Channel 数量
Notify 数量
HCCL Buffer 布局
Engine Context 版本
```

## 4. 算法流程

分别说明：

```text
Direct
Distributed Scatter + Fanout
Fallback 条件
root 任意值处理
尾块处理
```

## 5. 同步不变量

明确说明代码如何保证：

```text
本地源槽位不会提前覆盖
远端目标槽位不会提前覆盖
worker 完成前主线程不会结束
无效 Tile 不会产生悬空 Wait
Pairwise 不会双方先 Wait
```

## 6. 构建结果

贴出：

```text
bash build.sh --debug
bash build.sh
git diff --check
```

的结果摘要。

如果构建失败，不要声称完成。给出：

* 首个真实错误；
* 对应文件和行；
* 已确认原因；
* 未完成部分。

## 7. 风险与后续调优

区分：

```text
已验证的正确性
静态推导的正确性
尚未在 16 卡环境验证的部分
需要北极星或官方评测决定的性能参数
```

现在开始执行。先检查模板和本机 API，再分阶段修改、格式化和编译。
