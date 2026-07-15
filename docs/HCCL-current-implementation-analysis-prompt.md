# HCCL 现有实现梳理 Prompt

请完整阅读当前代码仓库，整理现有 HCCL Broadcast 实现，帮助我理解当前代码的真实行为。

## 一、任务边界

- 只分析代码并生成文档，不修改、重构或优化源代码。
- 所有结论必须给出实际文件、函数、结构体或关键变量作为依据。
- 不要按照理想设计补全代码中不存在的行为。
- 结论使用以下标记：
  - `[确认]`：可由代码直接确认。
  - `[推测]`：根据上下文推测，但不能完全确认。
  - `[待确认]`：当前仓库信息不足。
  - `[未使用]`：代码存在，但不在当前主路径上。

## 二、输出目录

在仓库中创建以下文档：

```text
docs/current-implementation/
├── README.md
├── 01-architecture.md
├── 02-resource-and-data-path.md
├── 03-signal-path.md
├── 04-code-map.md
├── 05-execution-trace.md
└── 06-invariants-and-questions.md
```

只允许修改 `docs/current-implementation/`，不要修改其他文件。

代码定位统一使用：

```text
文件：src/example.cpp
符号：Namespace::Class::Function
关键变量：variableName
行号：L100-L140（只作为辅助）
```

不能只给行号；不要大段复制源码。

## 三、README.md：总览和阅读导航

包含：

1. 当前分支、Commit ID、工作区是否存在未提交修改。
2. 项目实现的通信功能、算法和拓扑。
3. Host、Kernel、AI CPU、CCU分别承担什么职责。
4. Tile、Window、Slot、Thread、Notify和pipelineDepth的总体关系。
5. 使用实际函数名绘制Mermaid总体流程图。
6. 推荐的代码和文档阅读顺序。

## 四、01-architecture.md：架构和调用链

说明：

- Host入口、Kernel入口、AI CPU/CCU入口分别在哪里。
- 初始化、资源申请、Broadcast执行和资源释放流程。
- root、中间rank、叶子rank、coordinator、worker各自做什么。
- 不同rank或thread是否进入不同分支。

使用调用链表格：

| 顺序 | 文件 | 函数 | 调用者 | 主要职责 | 下一级调用 |
|---:|---|---|---|---|---|

分别给出初始化、单个Tile处理和通信结束的调用链。

## 五、02-resource-and-data-path.md：资源和数据通路

先建立资源表：

| 对象 | 代码对象 | 定义/创建位置 | 数量和大小 | 所有者 | 生命周期 | 作用 |
|---|---|---|---|---|---|---|

至少覆盖：

- rank、link、thread
- tile、window、slot
- user buffer、communication buffer
- notify、stream、task

给出当前代码中的实际计算表达式：

```text
tileSize = ...
tileCount = ...
windowId = ...
slotId = ...
bufferOffset = ...
actualTileSize = ...
```

说明这些表达式的单位是Byte、元素数还是索引。

然后选择一个普通Tile，从root用户Buffer一直跟踪到目标rank用户Buffer：

| 步骤 | Rank | Tile | 源Buffer/偏移 | 目标Buffer/偏移 | 长度 | 发起方 | 搬运接口 | 代码位置 |
|---:|---:|---:|---|---|---:|---|---|---|

必须明确：

- 发送方主动写还是接收方主动读。
- 中间rank收到数据后，转发和本地复制的顺序。
- 两个Window如何切换，Slot何时允许覆盖。
- 最后不足一个Tile的数据如何搬运。
- 4B等极小尾块是否单独产生一次完整通信。

## 六、03-signal-path.md：Notify和同步通路

扫描所有Notify申请、编号、Wait、Post、Reset、Barrier和完成判断代码。

建立Notify总表：

| 信号含义 | ID公式 | 申请位置 | Post方/位置 | Wait方/位置 | 区分维度 | 复用条件 |
|---|---|---|---|---|---|---|

“区分维度”需要检查：rank、peer、thread、direction、window、slot、tile和pipeline轮次。

要求：

1. 每个Wait都尝试找到唯一对应的Post。
2. 每个Post都说明由谁消费。
3. 区分数据到达、转发完成、本地复制完成、Slot已消费和算子完成。
4. 检查pipelineDepth大于1时是否可能错误复用Notify。
5. 使用Mermaid时序图展示一个Tile的完整信号流程，并标注实际函数名。

如果源码没有信号名称，可以使用 `[分析命名] DATA_READY`、`[分析命名] SLOT_CONSUMED`，但不能假装它是源码中的名字。

## 七、04-code-map.md：功能与代码块映射

建立总表：

| 功能 | 文件 | 函数/代码块 | 输入 | 输出 | 修改的状态 | 相关Buffer/Notify |
|---|---|---|---|---|---|---|

至少覆盖：

- rank信息和上下游选择
- Tile切分与尾块处理
- Window和Slot选择
- 数据发送、接收、转发、本地复制
- Notify申请、Wait和Post
- Thread创建与分工
- pipeline推进
- 完成判断和资源释放

然后选择主路径上最重要的5～10个函数，逐个说明：

- 调用者和被调用函数
- 参数来源及物理含义
- 关键变量和偏移含义
- 函数内部各代码块的职责
- 读取、写入了哪些Buffer
- 等待、发送了哪些Notify
- 前置条件和结束后保证
- 尾块和pipeline边界行为

不要逐行翻译C++语法，重点解释代码块在通信协议中的作用。

## 八、05-execution-trace.md：具体执行跟踪

优先使用项目中的真实测试配置；如果没有，选择4个rank、多个完整Tile和一个极小尾块。

至少跟踪：

- Tile 0
- 第一个Window切换的Tile
- 第一个Slot复用的Tile
- 流水线稳定阶段的Tile
- 最后一个尾Tile

使用表格：

| 时刻 | Rank | Thread | Tile | Window | Slot | 当前动作 | 数据位置 | 等待信号 | 发出信号 | Slot状态 |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|

严格按照当前代码执行顺序，不能按照理想算法补全。

## 九、06-invariants-and-questions.md：正确性约束

建立表格：

| 约束 | 代码保证位置 | 违反后的结果 | 当前是否明确保证 |
|---|---|---|---|

至少检查：

- Slot覆盖前是否完成所有读取。
- Notify复用前上一轮信号是否已消费。
- 不同Window、Slot、peer是否可能混用信号。
- 每个Wait是否一定存在对应Post。
- 尾Tile长度和偏移是否正确。
- worker退出前是否完成未决搬运。
- 多线程是否可能同时修改同一状态。

最后分别列出：

1. 已确认的实现行为。
2. 有代码依据的潜在风险。
3. 当前无法确认的问题，以及缺少什么信息才能确认。

当前阶段不要给出优化或修改方案。

## 十、完成前检查

生成文档后检查：

- 引用的文件、函数和变量真实存在。
- 数据通路、信号通路和执行跟踪相互一致。
- Tile、Window、Slot和Notify编号公式前后一致。
- 数据长度单位和尾块描述一致。
- 每个Wait/Post都完成配对检查。
- Mermaid图可以正常渲染。
- 推测没有写成确认事实。
- 除分析文档外，没有修改其他文件。

完成后在回复中告诉我：

1. 创建了哪些文档。
2. 推荐阅读顺序。
3. 当前实现最核心的3～5条结论。
4. 仍然无法确认的关键问题。
5. 明确确认没有修改源代码。
