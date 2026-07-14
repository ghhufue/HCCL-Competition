# HCCL Broadcast Competition

这个仓库用于完成 HCCL Broadcast 初赛题。题目要求在给定模板上实现
`HcclBroadcast`，让通信域内所有 rank 都得到 root rank 的数据，并尽量在
Ascend 950 的 `2 server x 8 NPU` 拓扑上取得更高有效带宽。

## 题目在哪里

题目材料整理在 `docs/` 目录：

- `docs/problem.md`：根据题面和模板整理出的完整题目说明。
- `docs/problem.txt`：原始题面文本。
- `docs/prompt.md`：补充记录材料。

核心要求：

- 实现接口：`HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)`。
- 数据类型：主要覆盖 `float32`。
- 测试数据量：`4B`、`512KB`、`512MB`、`400MB + 4B`。
- 功能测试 root：`0` 和 `7`。
- 性能测试 root：`0`。
- 目标拓扑：2 台 server，每台 8 个 Ascend 950 NPU，server 内 Full-Mesh，server 间 Clos。
- 通信引擎限制：`AICPU + TS`。

## 实现在哪里

代码模板在：

```text
hccl_broadcast_problem_template/
```

主要实现入口：

- `hccl_broadcast_problem_template/op_host/broadcast.cc`
  - Host 侧 `HcclBroadcast` 入口。
  - 负责参数整理、rank 信息获取、通信资源申请、AICPU kernel 下发。

- `hccl_broadcast_problem_template/include/custom.h`
  - 自定义资源结构体。
  - 需要放置 Host 侧和 AICPU 侧都要理解的序列化字段。

- `hccl_broadcast_problem_template/op_kernel_aicpu/exec_op.cc`
  - AICPU 侧实际通信算法编排。
  - 需要调用 HCOMM/HCCL 原语完成 Broadcast 数据流。

辅助文件：

- `hccl_broadcast_problem_template/op_kernel_aicpu/exec_op.h`
- `hccl_broadcast_problem_template/op_kernel_aicpu/aicpu_kernel.cc`
- `hccl_broadcast_problem_template/op_host/launch_aicpu_kernel.cc`
- `hccl_broadcast_problem_template/op_host/launch_aicpu_kernel.h`

通常优先改 `custom.h`、`broadcast.cc`、`exec_op.cc`，除非确实需要扩展辅助逻辑。

## 构建

模板构建依赖 CANN Toolkit 环境变量，构建前需要先 source CANN 环境：

```bash
source /home/workspace/hvm/Ascend/cann-9.1.0/set_env.sh
cd /mnt/d/code/HCCL-Competition/hccl_broadcast_problem_template
bash build.sh
```

Debug 构建：

```bash
bash build.sh --debug
```

格式化：

```bash
bash build.sh --format
```

## 测试环境

当前验证环境安装在 WSL 里面，路径为：

```text
/home/workspace/hvm
```

已安装的关键组件：

- CANN Toolkit：`/home/workspace/hvm/Ascend/cann-9.1.0`
- HCCL-VM：`/home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install`
- hccl_test：`/home/workspace/hvm/Ascend/cann-9.1.0/tools/hccl_test/bin`

进入环境：

```bash
wsl
source /home/workspace/hvm/Ascend/cann-9.1.0/set_env.sh
cd /home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install
source script/hccl_config.sh
```

启动比赛风格的 Ascend950 虚拟集群：

```bash
cd /home/workspace/hvm/hcomm/test/hccl_vm/hccl_vm_install/bin
./hccl-vm start ascend950_cluster_4_server_competition.yaml
```

选择 2 server x 8 rank 的通信域：

```bash
hccl-vm mock-comm 128
```

`128.yaml` 对应的通信域是 1 个 pod、2 台 server、16 个 rank：

```text
server 0: rank 0..7
server 1: rank 0..7
```

## HCCL-VM 验证命令

Broadcast 测试程序：

```text
/home/workspace/hvm/Ascend/cann-9.1.0/tools/hccl_test/bin/broadcast_test
```

示例：测试 16 rank、root 0、512KB、fp32：

```bash
mpirun --allow-run-as-root --oversubscribe -np 16 \
  /home/workspace/hvm/Ascend/cann-9.1.0/tools/hccl_test/bin/broadcast_test \
  -b 524288 -e 524288 -d fp32 -r 0 -w 5 -n 20 -c 1
```

常用 size：

```text
4B          -> -b 4 -e 4
512KB       -> -b 524288 -e 524288
512MB       -> -b 536870912 -e 536870912
400MB + 4B  -> -b 419430404 -e 419430404
```

root 切换：

```text
root 0 -> -r 0
root 7 -> -r 7
```

注意：`mpirun -np` 需要和 `mock-comm` 选择的 `rankNum` 一致。例如
`mock-comm 128` 是 16 rank，所以使用 `-np 16`。

## 测试结果说明

HCCL-VM 可以在没有真实 NPU 的 WSL 环境中做功能验证、通信流程验证和模拟性能参考。
它不等同于官方真实 Ascend950 测评环境，最终性能分数仍以比赛平台为准。

另外，直接运行 `broadcast_test` 会加载当前 CANN/HCCL 环境里的实现。要验证本仓库的
自定义实现，需要先构建模板产物，并确保 HCCL-VM/hccl_test 加载到本仓库生成的
host/device 库和对应 AICPU 配置。
