# HCCL Broadcast 初赛赛题说明

## 1. 赛题概述

本题要求实现集合通信算子 `Broadcast` 的操作接口，将通信域内 `root` 节点的数据广播到其他所有 rank。

算子语义：

- 对于 `root` rank，`buf` 是源数据缓冲区。
- 对于非 `root` rank，`buf` 是接收数据缓冲区。
- 调用完成后，所有 rank 的 `buf` 内容应与 `root` rank 的源数据一致。

本题不是普通算法题，而是 HCCL 自定义集合通信算子开发题。选手需要基于给定拓扑，实现控制面的通信资源申请，以及 AICPU 侧数据面的通信算法编排。

## 2. 评测拓扑

评测环境为 `2 * 8` 卡 Ascend 950 仿真环境，共 16 个 NPU。

拓扑特征：

- 由 2 个 Server 组成。
- 每个 Server 内包含 8 个 NPU。
- Server 内 8 个 NPU 之间为 Full-Mesh 互联。
- Server 间通过 Clos 网络互通。
- 每个 NPU 连接 Clos 网络的带宽大致是 Server 内单条直连链路带宽的 8 倍。

这意味着 Broadcast 算法不仅要正确，还要考虑跨 Server 与 Server 内传播路径的带宽利用。

## 3. 接口定义

需要实现的接口为：

```cpp
HcclResult HcclBroadcast(
    void *buf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    HcclComm comm,
    aclrtStream stream);
```

参数说明：

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `buf` | 输入/输出 | 数据 buffer 地址。对 `root` rank 是源数据 buffer；对非 `root` rank 是接收 buffer。 |
| `count` | 输入 | 参与 Broadcast 的数据元素个数。例如 1 个 `float32` 时，`count = 1`。 |
| `dataType` | 输入 | Broadcast 数据类型，类型为 `HcclDataType`。 |
| `root` | 输入 | 作为 Broadcast 源节点的 rank id。 |
| `comm` | 输入 | 集合通信操作所在的通信域。 |
| `stream` | 输入 | 当前 rank 使用的 stream。 |

返回值：

- 成功返回 `HCCL_SUCCESS`。
- 失败返回其他 `HcclResult` 错误码。

## 4. 算子约束

- 所有 rank 的 `count`、`dataType`、`root` 均相同。
- 全局只有一个 `root` 节点。
- `root` 不一定是 rank 0。
- 数据类型覆盖 `float32`。
- 限定使用 `AICPU + TS` 通信引擎。

## 5. 测试数据范围

输入数据 size 覆盖：

- `4B`
- `512KB`
- `512MB`
- `400MB + 4B`

由于数据类型为 `float32`，对应 `count` 可按字节数除以 `sizeof(float)` 理解。

## 6. 评分方式

赛题从功能和性能两个方面评分。

### 6.1 功能分

功能用例共 8 个：

- `root rank = 0`
- `root rank = 7`
- 数据量分别为：
  - `4B`
  - `512KB`
  - `512MB`
  - `400MB + 4B`

每个用例 15 分，共 120 分。

功能测试通过即可得分，不通过得 0 分。

### 6.2 性能分

性能用例共 3 个：

- `root rank = 0`
- 数据量分别为：
  - `512KB`
  - `512MB`
  - `400MB + 4B`

每个用例 10 分，共 30 分。

性能分按带宽用量排名计分，带宽最高得满分，后续按排名递减。

需要先保证功能用例全部通过，才能参与性能评分。

## 7. 可修改文件

题目要求仅可修改指定文件。README 中建议主要修改前三类文件：

- `include/custom.h`
- `op_host/broadcast.cc`
- `op_kernel_aicpu/exec_op.cc`
- `op_kernel_aicpu/exec_op.h`
- `op_host/launch_aicpu_kernel.cc`
- `op_host/launch_aicpu_kernel.h`
- `op_kernel_aicpu/aicpu_kernel.cc`

其中最核心的是：

- `broadcast.cc`：Host 侧入口与控制面资源申请。
- `custom.h`：自定义资源结构体、序列化字段。
- `exec_op.cc`：AICPU 侧通信算法任务编排。

## 8. 模板代码结构

项目结构：

```text
hccl_broadcast_problem_223_template/
├── CMakeLists.txt
├── build.sh
├── include/
│   ├── hccl.h
│   ├── common.h
│   ├── custom.h
│   ├── log.h
│   └── binary_stream.h
├── op_host/
│   ├── broadcast.cc
│   ├── launch_aicpu_kernel.cc
│   ├── launch_aicpu_kernel.h
│   └── CMakeLists.txt
└── op_kernel_aicpu/
    ├── aicpu_kernel.cc
    ├── exec_op.cc
    ├── exec_op.h
    └── CMakeLists.txt
```

模板已经提供：

- `HcclBroadcast` 入口函数框架。
- 参数封装为 `OpParam`。
- 获取 `myRank` 与 `rankSize`。
- Host/Device 同步 thread 的申请与导出。
- HCCL buffer 获取。
- AICPU kernel 加载与下发。
- AICPU 侧 `HcclAICPUKernel` 入口。
- `AlgResourceCtx` 的序列化与反序列化。
- `HcommBatchModeStart` / `HcommBatchModeEnd` 调用框架。

模板尚未实现：

- 通信 thread 数量设计。
- channel/link 资源申请。
- rank 间 Broadcast 通信路径。
- 小包、大包、非整块数据的处理策略。
- AICPU 侧 `Hcomm` 通信原语编排。

## 9. 主要实现任务

### 9.1 Host 侧资源申请

位置：`op_host/broadcast.cc`

需要根据算法设计申请资源，包括：

- AICPU_TS thread。
- 每个 thread 上所需 notify 数量。
- rank 间 channel。
- 本端和远端 HCCL 通信 buffer。
- 必要的拓扑/link 信息。

模板中已经提示可使用类似接口：

- `HcclRankGraphGetLinks()`
- `HcclChannelDescInit()`
- `HcclChannelAcquire()`

实际可用接口以 CANN/HCCL/HCOMM 头文件为准。

### 9.2 AICPU 侧算法编排

位置：`op_kernel_aicpu/exec_op.cc`

需要在 `ExecOp` 中完成 Broadcast 数据面的调度。

需要处理：

- 当前 rank 是否为 `root`。
- 当前 rank 从哪个 rank 接收数据。
- 当前 rank 向哪些 rank 发送数据。
- 数据量较小时的启动开销。
- 数据量较大时的分块、流水、并发传输。
- `400MB + 4B` 这类非整块尾部数据。

### 9.3 拓扑感知算法

由于评测拓扑为 2 个 Server，每个 Server 8 卡，可以考虑：

- 先做跨 Server 传播，再做 Server 内传播。
- 或者 root 同时向多个目标扩散，减少广播树深度。
- 对大数据使用分块流水，避免单一路径成为瓶颈。
- 对小数据降低调度和同步开销。

性能分关注带宽利用，因此单纯串行从 root 发送给其他 15 个 rank 虽然容易正确，但性能可能较差。

## 10. 编译环境要求

模板构建脚本要求先配置 CANN 环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

然后执行：

```bash
cd hccl_broadcast_problem_223_template
bash build.sh
```

编译依赖：

- Linux shell 环境。
- CMake 3.16 或以上。
- CANN Toolkit。
- CANN HCC/HCCL/HCOMM 相关头文件与库。
- AICPU device 侧交叉编译工具链。

模板默认从环境变量 `ASCEND_HOME_PATH` 获取 CANN 路径。

## 11. 验证环境要求

仅编译通过不代表功能正确。

完整功能和性能验证需要：

- 可运行 HCCL 集合通信任务的 Ascend 环境。
- 与题目相同或相近的 16 卡拓扑。
- 支持 AICPU + TS 通信引擎。
- 能加载自定义 host/device 动态库。
- 能运行多 rank 测试程序。

如果只有普通开发机或没有 Ascend NPU，只能做：

- 编译检查。
- 静态代码检查。
- 资源申请逻辑审查。
- 算法流程推演。

最终性能结果仍以官方评测环境为准。

## 12. 一句话总结

本题考察的是：在指定 Ascend 2 Server、16 NPU 拓扑下，使用 AICPU+TS 通信引擎补全 HCCL Broadcast 自定义算子，实现正确的数据广播，并尽可能提高大数据场景下的有效带宽。
