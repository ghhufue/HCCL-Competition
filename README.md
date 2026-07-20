# HCCL Broadcast Competition

本仓库用于实现 HCCL 自定义 `Broadcast` 集合通信算子。初赛已经结束，相关题目、代码和实现分析现作为历史资料保留；当前开发目标是决赛题目。

决赛延续初赛的 Broadcast 语义和基本开发流程，但更换了评测拓扑，并将通信引擎从 `AICPU + TS` 改为 `CCU`。因此，初赛的算法设计和验证经验仍有参考价值，但资源申请、Kernel 编排和拓扑优化需要按决赛模板重新实现。

## 项目阶段

| 项目 | 初赛（已结束） | 决赛（当前） |
| --- | --- | --- |
| 代码目录 | [`Hccl_Broadcast_Preliminary/`](Hccl_Broadcast_Preliminary/) | [`Hccl_Broadcast_Final/`](Hccl_Broadcast_Final/) |
| 题目目录 | [`docs/Preliminaries/`](docs/Preliminaries/) | [`docs/Finals/`](docs/Finals/) |
| 通信引擎 | `AICPU + TS` | `CCU` |
| 评测拓扑 | 固定 `2 Server × 8 NPU` | 基于 `4 Server × 8 NPU` 集群的 `2×8`、`4×1`、`8+4` 三种子拓扑 |
| 数据类型 | `float32` | `float32` |
| 数据量 | `4B`、`512KB`、`512MB`、`400MB + 4B` | `512KB`、`512MB`、`400MB + 4B` |
| 主要实现位置 | Host 资源申请与 AICPU 数据面编排 | Host 资源申请、CCU Kernel 注册/下发与 CCU 数据面编排 |

## 题目材料

### 决赛

- [决赛原始题面](docs/Finals/problem.txt)
- [2×8 拓扑图](docs/Finals/fin_topo_1.png)
- [4×1 拓扑图](docs/Finals/fin_topo_2.png)
- [8+4 拓扑图](docs/Finals/fin_topo_3.png)

决赛核心要求：

- 实现接口：`HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)`。
- 所有 rank 的 `count`、`dataType` 和 `root` 相同，`root` 不一定是 rank 0。
- 每个对端只能申请一个 Channel；同一对端申请多个 Channel 不会获得额外链路带宽。
- 限定使用 `CCU` 通信引擎。
- 功能用例覆盖三种子拓扑、三个数据量以及题面指定的 root；性能用例覆盖三种子拓扑和三个数据量。
- 只有全部功能用例通过后，才会参与性能评分。

### 初赛归档

- [初赛整理题面](docs/Preliminaries/problem.md)
- [初赛原始题面](docs/Preliminaries/problem.txt)
- [初赛当前实现分析](docs/Preliminaries/current-implementation/README.md)

初赛的实现代码完整保存在 [`Hccl_Broadcast_Preliminary/`](Hccl_Broadcast_Preliminary/)，目录迁移前后代码内容保持不变。

## 决赛实现位置

当前应在 [`Hccl_Broadcast_Final/`](Hccl_Broadcast_Final/) 中开发。模板允许修改的六个文件为：

- [`include/custom.h`](Hccl_Broadcast_Final/include/custom.h)：定义 Host 与 CCU 侧共享的资源结构、Kernel 参数和序列化字段。
- [`op_host/broadcast.cc`](Hccl_Broadcast_Final/op_host/broadcast.cc)：实现 `HcclBroadcast` 入口，解析 rank/拓扑信息，申请 CCU Thread、Channel 和通信 Buffer，并注册 CCU Kernel。
- [`op_host/exec_op.h`](Hccl_Broadcast_Final/op_host/exec_op.h)：声明 Host 侧算法下发接口。
- [`op_host/exec_op.cc`](Hccl_Broadcast_Final/op_host/exec_op.cc)：反序列化资源，根据数据量切分任务并通过 `HcommCcuKernelLaunch()` 下发一个或多个 CCU Kernel。
- [`op_kernel_ccu/ccu_kernel.h`](Hccl_Broadcast_Final/op_kernel_ccu/ccu_kernel.h)：声明 CCU Kernel 及其参数。
- [`op_kernel_ccu/ccu_kernel.cc`](Hccl_Broadcast_Final/op_kernel_ccu/ccu_kernel.cc)：实现前同步、Broadcast 数据面传输和后同步。

模板当前保留了资源申请、Kernel 注册/下发和数据面传输的 `TODO`，需要根据三种决赛拓扑补全。实现时尤其要注意：

- 单次通信最大数据量为 `256 MiB`。
- 默认 HCCL Buffer 大小为 `400 MiB`。
- `512 MiB` 和 `400 MiB + 4B` 无法作为一次通信直接完成，必须正确分片并处理尾块。
- 三种子拓扑的 rank 数和链路结构不同，不能把初赛固定 16 rank 的映射直接写死到决赛算法中。
- 可以按需要注册和下发多个 CCU Kernel，以覆盖不同数据量或拓扑策略。

## 构建决赛模板

构建依赖 CANN Toolkit。先加载 CANN 环境，再进入决赛目录：

```bash
source /home/workspace/hvm/Ascend/cann-9.1.0/set_env.sh
cd /mnt/d/code/HCCL-Competition/Hccl_Broadcast_Final
bash build.sh
```

Debug 构建：

```bash
bash build.sh --debug
```

格式化允许修改的 C/C++ 文件：

```bash
bash build.sh --format
```

初赛归档如需重新构建，只需将工作目录换为 `Hccl_Broadcast_Preliminary/`，其构建方式不变。

## 验证流程

初赛和决赛的整体验证流程相似：

1. 加载 CANN 和 HCCL-VM 环境。
2. 构建当前阶段的自定义算子。
3. 将生成的 Host/Kernel 产物安装或注入测试环境。
4. 启动与题目一致的 Ascend 950 虚拟集群。
5. 选择目标通信域并确认 `rankNum`、rank 映射和拓扑类型。
6. 使用 `broadcast_test` 逐一执行功能用例。
7. 功能全部通过后，再增加预热和迭代次数采集性能结果。

当前 WSL 验证环境示例：

```text
CANN Toolkit: /home/workspace/hvm/Ascend/cann-9.1.0
HCCL-VM:       /home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install
broadcast_test:/home/workspace/hvm/Ascend/cann-9.1.0/tools/hccl_test/bin/broadcast_test
```

通用测试命令形式：

```bash
mpirun --allow-run-as-root --oversubscribe -np <rankNum> \
  /home/workspace/hvm/Ascend/cann-9.1.0/tools/hccl_test/bin/broadcast_test \
  -b <bytes> -e <bytes> -d fp32 -r <root> -w 5 -n 20 -c 1
```

决赛常用数据量：

```text
512KB       -> 524288
512MB       -> 536870912
400MB + 4B  -> 419430404
```

验证决赛时需要为 `2×8`、`4×1`、`8+4` 分别选择对应的通信域配置，并保证 `mpirun -np` 与该通信域的 `rankNum` 一致。初赛使用的固定 `mock-comm 128` / `-np 16` 参数不能直接套用到全部决赛拓扑。

HCCL-VM 可用于功能、通信流程和模拟性能验证，但不等同于官方真实评测环境；最终性能仍以比赛平台结果为准。

## 仓库结构

```text
HCCL-Competition/
├── Hccl_Broadcast_Final/        # 当前决赛模板与后续实现
├── Hccl_Broadcast_Preliminary/  # 已结束的初赛代码归档
├── docs/
│   ├── Finals/                  # 决赛题面与三种拓扑图
│   └── Preliminaries/           # 初赛题面、提示和实现分析
├── scripts/                     # 本地构建/验证辅助脚本
└── artifacts/                   # 本地验证结果（默认不提交）
```
