# HCCL Broadcast 多 Die 修复方案

## 1. 结论

当前实现的核心问题不是某一条 CCU API 调用错误，而是：**按 Die 拆分 CCU kernel 后，算法仍然按照全局同步模型推进，但不同 Die 之间没有设备侧阶段屏障。**

推荐的最终修复方向是：

1. 为双 Die 场景申请一个绑定用户 stream 的 main CCU thread 和一个 slave CCU thread；
2. 在 Pull 的每个阶段前后，通过 thread notify 建立设备侧跨 Die 屏障；
3. 将 channel notify 改为按语义固定分配 bit，避免动态 root 和连续调用时通知串扰；
4. 在同步协议验证完成后，删除每个 chunk 后的 `aclrtSynchronizeStream()`；
5. 修复验证脚本，等待 checker 真正 drain 后再退出；
6. 最后再减少 phase/kernel launch 数，并对 Direct/Pull 做性能 A/B 测试。

当前的“多 Die 强制 Direct + 每 chunk 同步 stream”应保留为调试和正确性基线，不应作为最终竞赛实现。

---

## 2. 当前问题

### 2.1 按 Die 拆 kernel，但没有拆掉全局阶段假设

当前代码把 channel 按 `localDieId` 分组，每个 Die 注册一套 Pull kernel。这解决了以下注册错误：

```text
GetDieIdByChannels ... not same
```

但是 Pull 协议仍然假设以下阶段是全局完成的：

```text
全局 Seed 完成
→ Scatter/Seed 阶段结束
→ 全局 AllGather 完成
→ Global Done
```

每个 Die kernel 实际只能观察本 Die 的 peer 子集，因此可能出现：

1. root 的 Die 0 收齐自己负责的 SEED 通知；
2. Die 1 还没有收齐；
3. Die 0 已经进入 `PHASE2_START`；
4. 非 root 在某个 Die 上提前进入 AllGather；
5. 不同 rank、不同 Die 之间形成环形等待。

当前 [`BuildDieLaunchOrder()` 和 phase launch`](https://github.com/ghhufue/HCCL-Competition/blob/codex/hccl-broadcast-final/Hccl_Broadcast_Final/op_host/exec_op.cc) 只能控制 host 提交顺序，不能保证设备执行顺序。

### 2.2 两个 Die 之间没有设备侧 happens-before

当前只有一个和用户 stream 绑定的 `cpuThread`，并且 main thread 没有申请 thread notify，也没有 slave thread。

代码缺少以下设备侧依赖：

```text
Die 0 阶段结束 ─┐
                ├─ 两个 Die 全部完成 → 下一阶段
Die 1 阶段结束 ─┘
```

host 先 launch Die 0、再 launch Die 1，不等于 Die 0 和 Die 1 在设备上已经完成阶段同步。这是多 Die Pull 概率性死锁的根因。

### 2.3 Channel notify bit 生命周期不安全

当前 [`ccu_kernel.cc`](https://github.com/ghhufue/HCCL-Competition/blob/codex/hccl-broadcast-final/Hccl_Broadcast_Final/op_kernel_ccu/ccu_kernel.cc) 的多个 Pull 阶段共用一个 phase bit。

动态 root 场景可能出现：

```text
调用 N：   A 是 root，A → B 发送 GLOBAL_DONE
调用 N+1： B 是 root，A → B 发送 SEED_DONE
```

这两个通知方向相同、语义不同。如果复用同一个尚未完全消费的 bit，可能发生旧通知被错误消费或两次 Record 合并。

此前使用“按调用次数轮换 bit”的方案导致 CCU 微码分支增加并触发 `CCU_E_UNAVAIL`，说明调用代数不应进入 CCU 动态控制流。

### 2.4 Direct 回退破坏异步语义和性能

当前多 Die 强制 Direct，并在每个 chunk 后执行：

```cpp
aclrtSynchronizeStream(stream);
```

它能降低固定 notify bit 跨调用残留造成的问题，但代价包括：

- host 被阻塞；
- chunk 之间不能正常流水；
- 连续 Broadcast 不能充分重叠；
- 多 Die Pull 完全失效；
- 小包和中包性能容易被同步开销主导。

因此，这一方案适合作为正确性基线，不适合作为最终版本。

### 2.5 Checker 没有完成 drain

当前验证流程混淆了三种数量：

- runner 参数 `-n 20`；
- HCCL 的预热或内部执行轮次；
- checker 实际接收并处理的 op 数量。

runner 结束后立即向 HVM 发送 `exit`，会导致 checker 的异步队列尚未处理完就退出。

因此：

```text
runner success + Checker Success 只有 8 条
```

只能说明 checker 提前退出，既不能判定 Broadcast 失败，也不能证明 20 次检查全部成功。

---

## 3. 修复后必须满足的正确性不变量

### 3.1 Pull 算法不变量

1. 所有 Die 的 `PRESYNC` 完成后，才能进入 SEED；
2. 所有 peer 的 SEED 完成后，root 才能发布 `PHASE2_START`；
3. 一个 rank 的两个 Die 都完成 AllGather 后，才能发送 `READ_DONE`；
4. root 收齐所有 peer 的 `READ_DONE` 后，才能发布 `GLOBAL_DONE`；
5. 用户 stream 完成该 Broadcast 时，slave Die 上的工作也必须完成。

### 3.2 Notify 生命周期不变量

1. 不同语义阶段不能依赖同一个尚未确认消费的 bit；
2. 一个 bit 再次 Record 前，上一次 Record 必须已被对应 Wait 消费；
3. 不使用调用次数、root 或 chunk 编号产生 CCU 动态分支；
4. 跨 Die 同步使用 thread notify；
5. 跨 rank 同步使用 channel notify；
6. thread notify 和 channel notify 的生命周期分别管理。

---

## 4. 核心修复：main/slave CCU thread

官方多 Die NHR Broadcast 使用主、从 CCU thread，并在不同 thread 之间做 pre/post sync，而不是依靠 host launch 顺序。可参考[官方 NHR host 模板](https://gitcode.com/cann/hccl/blob/master/src/ops/broadcast/template/ccu/ccu_temp_broadcast_nhr_1D_mem2mem.cc)。

建议采用以下结构：

```text
mainThread  ：绑定用户 stream，执行第一个活跃 Die
slaveThread ：CCU slave thread，执行第二个活跃 Die
```

### 4.1 修改资源结构

修改 [`include/custom.h`](https://github.com/ghhufue/HCCL-Competition/blob/codex/hccl-broadcast-final/Hccl_Broadcast_Final/include/custom.h)：

```cpp
constexpr uint32_t RESOURCE_LAYOUT_VERSION = 6;

struct AlgResourceCtx {
    // 现有字段
    uint32_t activeDieMask;
    uint32_t peerDieByRank[MAX_RANK_SIZE];
    KernelHandle directKernel[BROADCAST_CCU_DIE_NUM];
    KernelHandle pullKernel[BROADCAST_CCU_DIE_NUM];

    // 新增字段
    ThreadHandle slaveThread;
    uint32_t slaveThreadCount;
};
```

同时需要更新：

- 资源序列化；
- 资源反序列化；
- resource size 检查；
- layout version 检查；
- Debug 日志。

资源 tag 建议改为：

```cpp
"hccl_custom_broadcast_v6"
```

避免旧注册资源被错误复用。重新测试前应确认旧 HVM/resource 缓存已失效。

### 4.2 给 main thread 申请 notify

修改 [`op_host/broadcast.cc`](https://github.com/ghhufue/HCCL-Competition/blob/codex/hccl-broadcast-final/Hccl_Broadcast_Final/op_host/broadcast.cc)：

```cpp
// notifyNumOnMainThread 从 0 改为 1
HcclThreadAcquireWithStream(
    comm,
    COMM_ENGINE_CCU,
    stream,
    1,
    &param.cpuThread);
```

存在两个活跃 Die 时，再申请一个 slave thread：

```cpp
if (PopCount(resCtx.activeDieMask) > 1) {
    CHK_RET(HcclThreadAcquire(
        comm,
        COMM_ENGINE_CCU,
        1,  // threadNum
        1,  // notifyNumPerThread
        &resCtx.slaveThread));

    resCtx.slaveThreadCount = 1;
}
```

> 注意：以上是结构性伪代码。`HcclThreadAcquire*` 的准确参数顺序和 handle 类型必须以比赛环境实际使用的 HCCL 头文件为准。

### 4.3 增加统一的跨 Die phase helper

在 `exec_op.h/.cc` 增加：

```cpp
HcclResult LaunchPhaseAcrossDies(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    const ExecutionPlan &plan,
    PullPhase phase,
    uint64_t chunkOffset,
    uint64_t chunkSize);
```

双 Die 情况的结构如下：

```cpp
HcclResult LaunchPhaseAcrossDies(...)
{
    const uint32_t mainDie = activeDies[0];
    const uint32_t slaveDie = activeDies[1];

    ThreadHandle mainThread = param.cpuThread;
    ThreadHandle slaveThread = resCtx.slaveThread;

    // main queue: Record START
    CHK_RET(HcommThreadNotifyRecordOnThread(
        mainThread, slaveThread, THREAD_NOTIFY_INDEX));

    // slave queue: Wait START
    CHK_RET(HcommThreadNotifyWaitOnThread(
        slaveThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));

    // 分别提交到两个 CCU thread
    CHK_RET(LaunchPullKernel(
        mainThread,
        resCtx.pullKernel[mainDie],
        phase,
        chunkOffset,
        chunkSize));

    CHK_RET(LaunchPullKernel(
        slaveThread,
        resCtx.pullKernel[slaveDie],
        phase,
        chunkOffset,
        chunkSize));

    // slave queue: Record DONE
    CHK_RET(HcommThreadNotifyRecordOnThread(
        slaveThread, mainThread, THREAD_NOTIFY_INDEX));

    // main queue: Wait DONE
    CHK_RET(HcommThreadNotifyWaitOnThread(
        mainThread, THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT));

    return HCCL_SUCCESS;
}
```

真正需要建立的是以下设备队列顺序：

```text
main : START record → mainDie kernel  → DONE wait
slave: START wait   → slaveDie kernel → DONE record
```

Host API 的调用顺序负责把命令放入对应队列，跨 Die 的实际 happens-before 由 thread notify 建立。

### 4.4 单 Die 快路径

单 Die 不需要额外 thread notify：

```cpp
if (activeDieCount == 1) {
    return LaunchPullKernel(
        param.cpuThread,
        resCtx.pullKernel[activeDie],
        phase,
        chunkOffset,
        chunkSize);
}
```

这可以避免单 Die 拓扑承担额外同步开销。

### 4.5 第一版对所有 phase 建立 barrier

第一版不要尝试减少 barrier。当前所有 Pull phase 都通过统一 helper：

```cpp
for (PullPhase phase : pullPhases) {
    CHK_RET(LaunchPhaseAcrossDies(..., phase, ...));
}
```

稳定后再考虑：

- 合并 `PRESYNC_PUBLISH + PRESYNC_WAIT`；
- 合并不包含跨 Die 全局依赖的相邻 phase；
- 减少 CCU kernel launch 数。

---

## 5. 修复 channel notify

### 5.1 使用静态语义 mask

在 `ccu_kernel.h` 定义：

```cpp
enum BroadcastNotifyMask : uint32_t {
    NOTIFY_SEED_DONE       = 1U << 0,
    NOTIFY_PHASE2_START    = 1U << 1,
    NOTIFY_READ_DONE       = 1U << 2,
    NOTIFY_GLOBAL_DONE     = 1U << 3,
    NOTIFY_DIRECT_DONE_ACK = 1U << 4,
};
```

修改通知接口：

```cpp
inline void NotifyRecord(ChannelHandle channel, uint32_t mask);
inline void NotifyWait(ChannelHandle channel, uint32_t mask);
```

固定映射如下：

| 阶段 | 方向 | Mask |
| --- | --- | ---: |
| Seed 完成 | peer → root | bit 0 |
| 启动 AllGather | root → peer | bit 1 |
| AllGather 完成 | peer → root | bit 2 |
| Broadcast 完成 | root → peer | bit 3 |
| Direct 完成/确认 | 双向握手 | bit 4 |

这种静态语义分配不会像 invocation phase 轮换一样增加动态控制流。

### 5.2 最终 ACK 的处理

建议分阶段处理：

1. 初始调试阶段保留最终 ACK，方便确认通知全部消费；
2. 静态 mask、跨 Die barrier、连续动态 root 测试全部通过后，再评估是否删除；
3. 如果 Direct 已经实现完整 done/ack 双向握手，可以固定复用 Direct bit，因为上一调用结束时通知生命周期已经闭合。

---

## 6. 恢复异步语义

完成跨 Die barrier 后，main thread 的最后一个 `Wait DONE` 已经依赖 slave thread 的最后工作。

用户 stream 上的依赖变为：

```text
前序 stream 任务
→ main CCU 工作
→ 等待 slave CCU 完成
→ 后续 stream 任务
```

此时可以删除：

```cpp
aclrtSynchronizeStream(stream);
```

但推荐分步操作：

1. 增加 main/slave barrier；
2. 暂时保留 `aclrtSynchronizeStream()`；
3. 连续执行 100 次验证协议；
4. 引入静态 notify mask；
5. 删除 stream synchronize；
6. 重新执行相同矩阵。

这样可以区分跨 Die barrier 错误与 notify 生命周期错误。

---

## 7. Direct 路径的最终处理

Direct 不依赖全局 Scatter+AllGather 阶段，更容易并行化。

两个 Die 可以并行处理各自的 peer 子集：

```text
mainThread  → Die 0 Direct kernel
slaveThread → Die 1 Direct kernel
```

末尾执行一次 slave-to-main completion：

```text
main : START record → Direct Die0 → DONE wait
slave: START wait   → Direct Die1 → DONE record
```

Direct kernel 内部继续完成 root-peer 的 done/ack。这样可以实现：

- 两个 Die 并行传输；
- 不需要每 chunk `aclrtSynchronizeStream()`；
- 小包可能比多 phase Pull 更快。

最终算法选择建议通过实测确定：

```cpp
if (singleDie) {
    // 使用单 Die 的 Direct/Pull 策略
} else if (size <= measuredDirectThreshold) {
    // 双 thread 并行 Direct
} else {
    // 双 thread、phase barrier Pull
}
```

阈值应在 Release 模式下通过 4/12/16 卡 A/B 测试确定，不应凭经验硬编码。

---

## 8. 不要直接照搬官方“按 Die 切数据”

官方 NHR 往往让不同 Die 处理不同数据范围，但当前实现的资源模型是：

```text
每个 peer 只选择一条最佳 channel
→ 两个 Die 持有不同的 peer 子集
```

如果没有确认每个 Die 都具有完成独立传播所需的 channel 图，不要直接改成：

```text
Die 0 处理前半数据
Die 1 处理后半数据
```

近期最稳妥的实现仍然是：

```text
peer 按 Die 拆分
+ 每个 Pull phase 建立跨 Die thread barrier
```

它可能比官方完整 NHR 多一些 kernel launch，但更适合先恢复正确性和异步语义。

---

## 9. 验证脚本修复

### 9.1 不要在 runner 后立即发送 exit

当前流程应从：

```text
启动 HVM
发送 mock
发送 runner
发送 exit
```

改为：

```text
启动 HVM，并保持 stdin 打开
发送 mock
发送 runner
等待 runner 结束
等待 checker drain
发送 exit
等待 HVM 退出
```

可以使用 FIFO：

```bash
fifo="$workdir/hvm.stdin"
mkfifo "$fifo"

hccl-vm <"$fifo" >"$log_file" 2>&1 &
hvm_pid=$!

exec 3>"$fifo"

printf '%s\n' "$mock_cmd" >&3
printf '%s\n' "$runner_cmd" >&3

wait_for_runner_terminal "$log_file"
wait_for_checker_drained "$log_file"

printf '%s\n' "exit" >&3
exec 3>&-

wait "$hvm_pid"
```

实际脚本还需要使用 `trap` 清理 FIFO 和子进程，并给每个等待阶段设置 timeout。

### 9.2 最好增加 checker drain 标志

不要仅通过“日志几秒没有变化”推断完成。最好在 checker 请求队列末尾增加一个 sentinel：

```text
CHECKER_DRAIN_REQUEST
```

checker 必须在前面的检查请求处理完成后输出：

```text
CHECKER_DRAINED submitted=23 completed=23 failed=0
```

脚本看到该标志后才能发送 `exit`。

### 9.3 正确的通过条件

验证脚本的通过条件应为：

```text
runner terminal status == success
sync timeout == 0
hccl_op_base execute failed == 0
Checker Failed == 0
checker submitted == checker completed
checker drained marker exists
```

不应再使用：

```text
Checker Success 数量 == runner -n
```

如果暂时无法修改 checker，次优方案是：

1. 解析 checker 接收到的最大 op ID；
2. 等待该 op ID 出现 Success 或 Failed；
3. 再等待一个短暂稳定窗口；
4. 最后发送 `exit`。

但这种方式仍然不如显式 drain 可靠。

---

## 10. 推荐的分阶段提交方案

### Commit 1：资源结构

- 增加 slave thread；
- main/slave 各申请一个 thread notify；
- 升级 resource layout 和 tag；
- 暂不修改算法行为。

验收条件：注册成功，无 `CCU_E_UNAVAIL`，单 Die 用例不回退。

### Commit 2：跨 Die phase barrier

- 增加 `LaunchPhaseAcrossDies()`；
- 每个 Pull phase 在两个 thread 上执行；
- 暂时保留 multi-Die Direct fallback 和 stream sync；
- 通过独立测试开关启用新 Pull。

验收条件：12 卡、root=7、512 KiB 连续 20/100 次无 timeout。

### Commit 3：静态 channel notify mask

- 拆分四个 Pull mask；
- 删除调用次数 phase 轮换；
- 测试动态 root。

验收条件：root 交替连续 100/1000 次。

### Commit 4：删除同步回退

- 删除每 chunk `aclrtSynchronizeStream()`；
- 允许多 Die Pull；
- 保留环境变量强制选择算法，方便 A/B：

```text
HCCL_BROADCAST_ALGO=auto
HCCL_BROADCAST_ALGO=direct
HCCL_BROADCAST_ALGO=pull
```

### Commit 5：修复 checker

- 增加 drain 协议；
- 删除 `Checker Success == -n`；
- 输出每个 case 的结构化汇总。

### Commit 6：性能优化

- 合并可合并的 phase；
- 实现双 Die Direct 并行；
- 测量并确定 Direct/Pull 阈值。

---

## 11. 建议验证矩阵

| 维度 | 用例 |
| --- | --- |
| Rank 数 | 4、12、16 |
| 固定 root | 0、7、最后一个 rank |
| 动态 root | `0,7,0,7...`、顺序轮换、固定种子的随机 root |
| 连续次数 | 20、100；关键 case 跑 1000 |
| 数据大小 | 0/1 字节、对齐边界前后、chunk 边界前后、512 KiB、1 MiB、大包 |
| 算法 | 强制 Direct、强制 Pull、Auto |
| 构建类型 | Debug、Release |
| 同步方式 | 暂留 stream sync、删除 stream sync |
| 拓扑 | 单 Die、双 Die |

每个 case 建议记录：

```text
activeDieMask
mainDie/slaveDie
rootDie
chunkCount
algorithm
runner result
checker submitted/completed/failed
sync timeout
execute failed
latency/bandwidth
```

动态 root 是必须测试的场景，因为它最容易暴露 notify 方向交换和残留事件。

---

## 12. 官方实现参考方式

远程仓库足够确认架构方向。要让 AI 写出能在比赛环境编译的补丁，建议把官方实现拉到本地，因为比赛环境的 HCCL API、thread notify 参数和资源结构可能与上游 `master` 不完全一致。

建议放在竞赛仓库外，避免生成嵌套 Git 仓库：

```bash
git clone --depth 1 \
  https://gitcode.com/cann/hccl.git \
  /mnt/d/code/hccl-official
```

如果知道比赛环境对应的 CANN/HCCL 版本，应 checkout 对应 tag 或 commit，而不是直接使用最新 master。

重点搜索：

```bash
rg -n \
  "PreSyncInterThreads|PostSyncInterThreads|HcclThreadAcquireWithStream|HcclThreadAcquire|ThreadNotify" \
  /mnt/d/code/hccl-official/src \
  /mnt/d/code/hccl-official/inc
```

重点参考文件：

- [官方 NHR host 模板](https://gitcode.com/cann/hccl/blob/master/src/ops/broadcast/template/ccu/ccu_temp_broadcast_nhr_1D_mem2mem.cc)
- [官方 NHR CCU kernel](https://gitcode.com/cann/hccl/blob/master/src/ops/broadcast/template/ccu/kernel/ccu_kernel_broadcast_nhr1d_mem2mem.cc)
- [官方 Broadcast selector](https://gitcode.com/cann/hccl/blob/master/src/ops/broadcast/selector/broadcast_auto_selector.cc)

推荐本地目录布局：

```text
/mnt/d/code/HCCL-Competition
/mnt/d/code/hccl-official
```

之后让 AI 同时检查：

```text
HCCL-Competition/Hccl_Broadcast_Final
hccl-official/src/ops/broadcast
hccl-official 中 thread notify 的声明和实现
artifacts/run_hvm_case.sh
关键成功/失败日志
```

---

## 13. 最终验收标准

只有同时满足以下条件，才能认为竞赛版本闭环：

### 正确性

- 4/12/16 卡全部通过；
- 固定 root 和动态 root 全部通过；
- 连续 100/1000 次无偶发超时；
- 无 `sync timeout`；
- 无 `execute failed`；
- 无 `Checker Failed`；
- checker 明确完成 drain。

### 异步语义

- 正常路径不调用 `aclrtSynchronizeStream()`；
- Broadcast 返回后，正确性由 stream 依赖保证；
- slave Die 工作已经被 main stream 的 completion 覆盖。

### 性能

- 双 Die Direct 可以并行执行；
- 大包能够启用修复后的 Pull；
- Auto 策略的阈值来自 Release 实测；
- 相比 Direct + 每 chunk stream synchronize 有明显改善。

### 工程性

- 资源 layout/tag 已升级；
- 可通过环境变量强制 Direct/Pull；
- Debug 日志能定位 rank、root、Die、phase 和 chunk；
- 验证脚本不会因为 checker 提前退出产生误判。

---

## 14. 一句话总结

当前不应继续通过增加跨 rank ACK、改变 Die 启动顺序或按调用次数轮换 notify bit 来降低死锁概率；应当在资源和执行层引入官方多 Die 模板采用的 main/slave CCU thread，并在每个 Pull phase 建立真正的设备侧跨 Die屏障。在此基础上使用静态语义 notify mask、删除 host stream synchronize，并通过显式 checker drain 完成最终验收。
