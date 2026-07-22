# HCCL Broadcast：删除 Direct 并将小包改为 Receiver Pull

## 1. 任务目标

请结合当前仓库的真实代码完成以下修改：

1. **彻底删除小包 Direct 并行写实现**，不保留运行时 fallback、调试分支或环境变量入口。
2. 将小包 Broadcast 改为 **receiver pull**：root 只发布自己的源 Buffer 地址和 token，各非 root rank 主动从 root 读取完整数据。
3. 小包尽量复用大包 Pull 已有的地址/token发布机制与完成信号，使两条数据通路保持一致。
4. 非 root rank 只在“连接 root 的 Channel 所属 Die”上启动通信 Kernel；不要为无工作的另一个 Die 启动空 Kernel。
5. 保持大包 `PULL_SCATTER_ALLGATHER` 的现有行为不变，除非为了抽取公共 helper 必须重构。

最终算法只保留两条生产路径：

```text
小包：SMALL_RECEIVER_PULL
大包：PULL_SCATTER_ALLGATHER
```

旧 Direct 实现只应存在于 Git 历史中，不能继续存在于最终源代码中。

---

## 2. 强制约束

### 2.1 必须删除的行为

最终代码中不得再出现以下行为：

```text
root 通过 ccu::Write 向所有非 root rank 并行写入 Broadcast 数据
```

不得以任何形式保留：

- Direct 自动选择分支；
- `HCCL_BROADCAST_ALGO=direct`；
- Direct Kernel 注册与句柄；
- Direct 专用 phase；
- Direct 专用 ACK/Notify；
- `RunDirectRoot`、`RunDirectReceiver`、`LaunchDirectChunk` 等 Direct 数据路径；
- 失败时回退到 Direct 的逻辑；
- 仅用于保留旧路径的死代码和注释块。

### 2.2 不要把“小包 Pull”实现成全量地址交换

小包只有 root 是远端数据源，因此：

- root 向各 receiver 发布 **root 源 Buffer 地址和 token**；
- receiver 的本地目标地址来自自己的 Kernel 参数；
- 非 root 不需要向其他 rank 发布自己的地址/token；
- receiver 只等待 root 的地址/token，不等待其他 peer。

### 2.3 不要让同一 receiver 双 Die 拉取 root

当前资源模型为每个 peer 只选择一条最佳 Channel，该 Channel 固定属于一个本地 Die。因此第一版小包实现必须采用：

```text
每个 receiver 使用 root Channel 所属的单个 Die，读取完整小包
```

不要把同一份小包切成两半后强行让两个 Die 同时读取 root。

---

## 3. 目标数据通路

设 root 为 `R`，任意非 root rank 为 `P`。

```text
R：发布 root 源 Buffer 地址/token
P：等待 root 地址/token
P：Read(root source → local destination)
P：等待 Read Event 完成
P：向 root 发送 READ_DONE
R：收齐全部 receiver 的 READ_DONE
R：向全部 receiver 发送 GLOBAL_DONE
P：等待 GLOBAL_DONE 后结束
```

注意：root 发布的是“源地址”，不是 receiver 的“目标地址”。

对应读取语义：

```cpp
ccu::Read(
    rootChannel,
    localDestination,  // receiver 本地用户 Buffer
    remoteSource,      // root 发布的源 Buffer
    bytes,
    event,
    mask);
```

小包不执行 Seed 和 AllGather，也不需要 owner 分片计算。

---

## 4. 小包信号协议

小包使用两个严格串行的逻辑 phase，禁止 pipeline/window 重叠。

### Phase 0：`TRANSFER`

root 在每个活跃 Die 上处理归属于该 Die 的 receiver Channel：

1. 向本 Die 管理的 receiver 发布 root 地址/token；
2. 等待这些 receiver 返回 `READ_DONE`。

每个 receiver 只在 `rootDie` 上执行：

1. 等待 root 地址/token；
2. 从 root 读取完整小包；
3. 等待 Read Event，确认本地数据写入完成；
4. 向 root 发送 `READ_DONE`。

Phase 0 结束前，root 的两个 Die 必须在 Host/调度线程层完成汇合。只有两边负责的 receiver 都已经完成，才允许进入 Phase 1。

### Phase 1：`GLOBAL_DONE`

```text
root：向所有 receiver 发送 GLOBAL_DONE
receiver：等待来自 root 的 GLOBAL_DONE
```

`GLOBAL_DONE` 必须发生在 root 双 Die 汇合之后，不能由某个 Die 在局部收齐 `READ_DONE` 后提前发送。

---

## 5. 与大包 Pull 复用的信号

优先复用现有大包 Pull 资源，不新增没有必要的 Notify。

| 资源/信号 | 小包是否使用 | 小包语义 |
|---|---:|---|
| `CKE_PRESYNC` | 是 | root 地址/token发布 |
| `MASK_BUFFER_READY` | 是 | root 源地址已发布 |
| `MASK_TOKEN_READY` | 是 | root 源 token 已发布 |
| `CKE_PHASE` | 是 | Pull 阶段完成信号 |
| `NOTIFY_READ_DONE` | 是 | receiver 已完成整包 Read |
| `NOTIFY_GLOBAL_DONE` | 是 | 所有 receiver 已完成 |
| `NOTIFY_SEED_DONE` | 否 | 小包无 Seed |
| `NOTIFY_PHASE2_START` | 否 | 小包无 AllGather |
| `NOTIFY_DIRECT_DONE_ACK` | 删除 | 已被 `READ_DONE/GLOBAL_DONE` 取代 |

如果当前配置为：

```cpp
CHANNEL_NOTIFY_NUM = 2;
THREAD_NOTIFY_NUM = 1;
```

在两个 phase 严格串行且无 window 重叠的前提下应保持不变，不要无依据地增加 Notify 数量。

---

## 6. Die 调度规则

必须将“默认在所有活跃 Die 上启动 phase”改为“按角色和 phase 使用明确的 Die Mask”。

| 本地角色 | `TRANSFER` | `GLOBAL_DONE` |
|---|---:|---:|
| root | 所有活跃 Die | 所有活跃 Die |
| receiver | 仅 `rootDie` | 仅 `rootDie` |

建议新增或重构为：

```cpp
uint32_t GetSmallPullDieMask(
    const OpParam &param,
    const AlgResourceCtx &resCtx)
{
    if (param.myRank == param.root) {
        return resCtx.activeDieMask;
    }

    const uint32_t rootDie = resCtx.peerDieByRank[param.root];
    return 1U << rootDie;
}
```

将无条件遍历全部 Die 的接口改造成类似：

```cpp
LaunchPhaseOnDieMask(..., dieMask);
```

要求：

- root 使用两个 Die 覆盖各自管理的 receiver Channel；
- receiver 只启动 root Channel 所属 Die；
- 不允许 receiver 在另一个 Die 上启动空 Kernel；
- 双 Die barrier 只用于 root 等真正启动了两个 Die 的场景。

---

## 7. 文件级修改要求

开始修改前，请先搜索并核对当前分支中所有 Direct 相关符号及实际调用关系。以下名称如与当前代码略有差异，应按真实代码完成等价修改，不能因名称不同而漏删。

### 7.1 `include/custom.h`

将算法类型调整为只包含：

```cpp
enum class KernelKind : uint32_t {
    SMALL_RECEIVER_PULL = 0,
    PULL_SCATTER_ALLGATHER = 1,
};
```

新增小包 phase：

```cpp
constexpr uint32_t SMALL_PULL_PHASE_COUNT = 2;

enum class SmallPullPhase : uint64_t {
    TRANSFER = 0,
    GLOBAL_DONE = 1,
};
```

资源结构中删除：

```cpp
directKernels
```

替换为：

```cpp
smallPullKernels
```

资源布局发生变化后，必须同步升级：

- `RESOURCE_LAYOUT_VERSION`；
- 资源 tag/cache key；
- 所有依赖旧布局的初始化与校验。

不要让旧缓存中的 Direct Kernel 句柄被误解释为 Small Pull Kernel 句柄。

### 7.2 `op_host/broadcast.cc`

删除 Direct Kernel 的注册和资源初始化，注册新的：

```cpp
CcuBroadcastSmallReceiverPullKernel
```

同步完成：

- `directKernels` → `smallPullKernels`；
- 资源版本/tag升级；
- 删除 Direct 专用资源申请逻辑；
- 保留大包需要的全 peer Channel 资源。

虽然小包只读取 root，但 root 可动态变化，且大包仍需访问其他 owner，因此不要为了小包单独破坏公共 Channel 资源模型。

### 7.3 `op_kernel_ccu/ccu_kernel.h`

新增：

```cpp
CcuResult CcuBroadcastSmallReceiverPullKernel(CcuKernelArg arg);
```

删除：

```text
CcuBroadcastDirectKernel
NOTIFY_DIRECT_DONE_ACK
所有 Direct 专用声明
```

### 7.4 `op_kernel_ccu/ccu_kernel.cc`

删除全部 Direct Kernel 和 Direct 数据操作，包括 root 侧 `ccu::Write` 循环。

新增或抽取以下 helper：

```cpp
WaitRootBufferInfo()
ReadFullChunkFromRoot()
CcuBroadcastSmallReceiverPullKernel()
```

`WaitRootBufferInfo()`只能等待 `remoteRank == root` 的 Channel，不能复用会等待全部 peer 的 `WaitBufferInfo()`。

`ReadFullChunkFromRoot()`应读取当前 chunk 的完整有效字节：

```text
local  = localBuffer + chunkOffset
remote = rootBuffer + chunkOffset
bytes  = chunkBytes
```

不得加入 owner slice 偏移。尾块必须使用实际 `chunkBytes`，不能越界读取或按固定 tile 大小补读。

推荐的 Kernel 主体结构：

```cpp
CcuResult CcuBroadcastSmallReceiverPullKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<BroadcastKernelArg *>(arg);
    BroadcastContext ctx;

    CCU_CHK_RET(InitBroadcastResource(ctx, kernelArg));
    CCU_CHK_RET(LoadBroadcastArgs(ctx));
    CCU_CHK_RET(ccu::LoadArg(ctx.kernelPhase, 8));

    CCU_IF(ctx.kernelPhase ==
           static_cast<uint64_t>(SmallPullPhase::TRANSFER)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(PublishBufferInfo(ctx));
            CCU_CHK_RET(RunPullReadDoneRoot(ctx));
        }

        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(WaitRootBufferInfo(ctx));
            CCU_CHK_RET(ReadFullChunkFromRoot(ctx));
            CCU_CHK_RET(NotifyRoot(ctx, NOTIFY_READ_DONE, false));
        }
    }

    CCU_IF(ctx.kernelPhase ==
           static_cast<uint64_t>(SmallPullPhase::GLOBAL_DONE)) {
        CCU_IF(ctx.root == kernelArg->rankId) {
            CCU_CHK_RET(RunPullGlobalDoneRoot(ctx));
        }

        CCU_IF(ctx.root != kernelArg->rankId) {
            CCU_CHK_RET(NotifyRoot(ctx, NOTIFY_GLOBAL_DONE, true));
        }
    }

    return CCU_SUCCESS;
}
```

上面是目标结构，不要求机械照抄。必须结合现有 CCU DSL 的控制流、参数加载、EventWait 和 mask 语义实现。

### 7.5 `op_host/exec_op.cc`

修改自动选择器：

```cpp
algorithm = totalBytes <= SMALL_PULL_THRESHOLD_BYTES
    ? KernelKind::SMALL_RECEIVER_PULL
    : KernelKind::PULL_SCATTER_ALLGATHER;
```

初始阈值保持当前小包边界，例如：

```cpp
constexpr uint64_t SMALL_PULL_THRESHOLD_BYTES = 512ULL * 1024ULL;
```

新增：

```cpp
LaunchSmallReceiverPullChunk()
```

其职责为：

1. 计算本 rank 的 `dieMask`；
2. 严格串行启动 `TRANSFER`；
3. 完成必要的本地 Die 汇合；
4. 再启动 `GLOBAL_DONE`；
5. 正确处理 chunkOffset、chunkBytes 和尾块。

删除：

```text
LaunchDirectChunk
Direct phase调度
Direct fallback
HCCL_BROADCAST_ALGO=direct
所有 Direct 分支
```

环境变量如需保留算法强制选择，只允许：

```text
HCCL_BROADCAST_ALGO=auto
HCCL_BROADCAST_ALGO=small_pull
HCCL_BROADCAST_ALGO=pull
```

对于未知值必须明确报错或回到 `auto`，不得静默选择 Direct。

---

## 8. Direct 代码删除检查表

修改完成后，使用 `rg` 搜索整个目标实现目录，确保以下内容为零或仅存在于与实现无关的历史说明中：

```bash
rg -n -i "direct|RunDirect|LaunchDirect|DirectKernel|DIRECT_DONE|directKernels" Hccl_Broadcast_Final
```

必须逐项确认：

- [ ] `KernelKind::DIRECT` 已删除；
- [ ] `CcuBroadcastDirectKernel` 已删除；
- [ ] `RunDirectRoot` 已删除；
- [ ] `RunDirectReceiver` 已删除；
- [ ] `LaunchDirectChunk` 已删除；
- [ ] `NOTIFY_DIRECT_DONE_ACK` 已删除；
- [ ] `directKernels` 已删除；
- [ ] `HCCL_BROADCAST_ALGO=direct` 已删除；
- [ ] root 对 peer Buffer 的 Direct `ccu::Write` 数据循环已删除；
- [ ] 自动选择器和错误回退不会进入 Direct；
- [ ] 构建脚本、注册表、日志和注释不再引用已删除符号。

如果 `direct` 一词仍用于与本任务无关的底层 API 或通用概念，应在最终报告中逐处说明，不能简单忽略搜索结果。

---

## 9. 正确性与同步要求

必须保证以下不变量：

1. receiver 只有在收到 root 地址和 token 后才能发起 Read；
2. receiver 只有在 Read Event 完成后才能发送 `READ_DONE`；
3. root 必须收齐所有 receiver 的 `READ_DONE`；
4. root 必须在两个 Die 的局部结果汇合后才能发送 `GLOBAL_DONE`；
5. receiver 必须消费 `GLOBAL_DONE` 后才能结束本次 Broadcast；
6. 同一 Channel 上的 Notify 必须在进入下一次 Broadcast 前被完整消费；
7. 动态 root 时不能复用上一次 root 的 Channel、地址或 token；
8. 4 B、小于阈值、等于阈值和尾部非对齐数据都不能越界；
9. `rankSize == 1` 时应直接成功，不等待任何 peer 信号；
10. 任一 CCU/Host 错误必须正常向上传递，禁止用 Direct 回退掩盖错误。

---

## 10. 验证矩阵

至少覆盖：

| 维度 | 用例 |
|---|---|
| 数据大小 | 4 B、512 KiB−4 B、512 KiB、512 KiB+4 B、1 MiB |
| Rank 数 | 1、2、4、12、16 |
| root | 0、7、最后一个 rank |
| 动态 root | `0→7→0→7`、顺序轮换 |
| 连续执行 | 20、100；关键用例建议1000次 |
| 算法模式 | `small_pull`、`pull`、`auto` |
| 检查项 | checker、超时、数据错误、Kernel失败、Notify残留 |

关键回归用例：

```text
12 ranks
root = 7
512 KiB
连续至少100次
```

要求：

- 数据校验全部通过；
- 无同步超时；
- 无 Kernel 执行失败；
- 无依赖 `aclrtSynchronizeStream` 的临时正确性修复；
- 小包日志显示 `SMALL_RECEIVER_PULL`；
- receiver 的 Die Mask 仅包含 `rootDie`；
- root 不再执行面向所有 receiver 的数据 `ccu::Write`；
- 全局实际 Read 数量符合 `rankSize - 1` 个 receiver 各拉一次完整数据。

性能对比至少记录：

```text
512 KiB端到端耗时
root/receiver各自的Kernel Launch数量
两个Die的实际工作分布
root侧Write数量
receiver侧Read数量
```

---

## 11. 建议实施顺序

1. 搜索并画出当前 Direct 的选择、注册、启动和 CCU 执行调用链；
2. 实现 `SMALL_RECEIVER_PULL` Kernel 与 root-only 地址/token等待逻辑；
3. 实现按角色选择的 Die Mask；
4. 用 `small_pull` 强制模式验证小包协议；
5. 将 `auto` 的小包分支切换到 `SMALL_RECEIVER_PULL`；
6. 删除所有 Direct 代码、资源、信号和环境变量；
7. 升级资源布局版本与 tag；
8. 完成全量构建、正确性回归和连续运行测试；
9. 使用 `rg` 执行 Direct 残留检查；
10. 最后再进行性能测量，不要在协议尚未稳定时提前加入 pipeline。

---

## 12. AI 最终输出要求

完成修改后，请给出：

1. 根因与旧 Direct 数据通路的简要说明；
2. 实际修改的文件列表；
3. 新小包 Receiver Pull 的数据通路和信号通路；
4. root 与 receiver 的 Die Mask 计算方式；
5. 已删除的 Direct 符号清单；
6. `rg` 残留搜索结果及解释；
7. 编译命令与结果；
8. 所有已运行测试、结果和耗时；
9. 尚未验证的风险，不得用“应该可行”代替测试结论。

不要只给方案或伪代码。请直接修改代码、编译并验证；如果当前环境不能运行 HCCL VM/NPU 测试，也必须完成静态检查和构建，并明确列出需要用户在目标环境执行的具体命令。

