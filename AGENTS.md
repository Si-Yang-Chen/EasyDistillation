# 项目执行约定

## Kunshan 是数据与计算环境

- 本地工作区主要用于代码开发、CPU 合成测试和文档维护；规范场、eigenvector、VSV/PSV/VSP/PSP 传播子、elemental、收缩结果和历史作业产物的权威副本主要位于 **Kunshan 集群**。
- 本地目录没有某个数据文件，不代表该数据不存在。不得仅凭本地搜索结果把数据、真实 smoke 或物理验证标记为不可执行或长期阻塞。
- 需要真实数据时，先在 Kunshan 上查找已有数据、manifest、历史结果目录和作业记录；记录配置号、绝对路径、shape、dtype、轴约定和 SHA-256 后再消费。
- 不猜测 Kunshan 路径、配置号或轴语义。若现有代码和记录无法确定，先在集群只读核验；只有集群入口、帐号或必要路径确实未知时才向用户询问。

## 复用与补算顺序

1. 优先复用 Kunshan 上已经存在且可审计的规范场、eigenvector、传播子和 elemental。
2. 校验已有产物是否与当前配置、边界条件、Ne/Np、时间轴、软件版本和消费公式兼容；不兼容的旧数据不能静默复用。
3. 若所需数据在 Kunshan 上经核验后确实缺失、损坏或不兼容，允许在 Kunshan 提交计算作业补齐，并继续完成后续 smoke、收缩和验证。
4. **确认缺失时允许重新生成传播子。** 这条规则取代旧交接文档中“绝不重算传播子”的笼统限制：实际原则是“已有则复用，缺失或不兼容则在集群补算”。这是对必要数据计算的项目授权，不等于允许任意扩大节点、DCU、时限、并发或重试预算。
5. 不得伪造数据、哈希、作业状态或物理结论；合成测试结果必须继续标记为合成证据。

## 集群执行规则

- Kunshan 登录节点不提供 DCU。GPU/CuPy/PyQUDA/QUDA 任务必须通过 Slurm 提交到计算节点，不能在登录节点直接运行。
- DCU 作业遵循仓库内 `.cursor/skills/dcu-slurm-submit/SKILL.md`：使用站点批准的分区、帐号、模块、MPI ABI、资源边界、结果目录、作业台账和 monitor。
- CPU 只读检查、manifest 校验、文件哈希和轻量 NumPy 测试可在登录节点运行；昂贵生成、反演和 GPU 收缩必须进入计算队列。
- 必要的数据生成和传播子补算已获项目层面许可；但生产补算不得伪装成自动测试作业。应使用 Kunshan 批准的生产提交流程；若仓库只有测试提交模板而没有生产流程，则先准备输入 manifest、资源估算、结果契约和脚本，再向用户确认具体生产资源参数。
- 每个补算或真实 smoke 作业都要使用新的结果目录，记录输入哈希、Git commit/dirty 状态、资源参数、Slurm Job ID 和轴约定。
- 作业应原子写出机器可读结果，再创建 `DONE`；提交后必须启动持久 monitor，不能只记录 Job ID 就停止工作。
- 失败时先区分代码错误、输入缺失、环境/MPI 问题和调度问题；不得通过增加 GPU、并发、时限或重试次数来掩盖失败。

## 多 Agent 远端协作与 Kunshan 上线

- Git remote、Kunshan checkout 和 `~/qedinf/experiments/*` 可能同时由其他 agent 维护。上线前必须重新执行 `git fetch --prune`，记录本地 HEAD、目标 remote ref、Kunshan checkout HEAD/branch/dirty 状态，并核对相关实验目录和正在运行的 Slurm 作业；不得沿用较早会话中的 remote 状态。
- 不得以本项目为由回退、覆盖、删除或清理其他 agent 的提交、分支、worktree、dirty 文件、实验脚本或结果。禁止对共享 checkout 使用 `git reset --hard`、`git clean`、强制 checkout 或无差别 rsync；部署应进入新的隔离源码/结果目录。
- 外部变化按“路径所有权 + 实际依赖”分类，而不是仅看提交信息：若与本项目修改文件、import/export、公共 API、schema、数据格式、构建/运行环境和消费公式均无交集，则视为主线无关变化，记录 ref/commit 后原样接受，**无需逐行代码审查或为其补跑专项测试**。
- 无关变化可以随选定的共同基线进入 Kunshan；不得为了缩小 diff 而将其剔除或回退。最终证据需注明它来自哪个 ref/commit，且本项目未对其作正确性背书。
- 若变化触及本项目文件或共享依赖（当前包括 `lattice/__init__.py`、`lattice/generator/elemental.py`、`lattice/insertion/*`、`lattice/quark_diagram.py`、propagator/preset/slab readers、Current schemas 和 Kunshan launcher），则不能按无关变化放行：先比较双方 commit/diff，保留两边语义，执行聚焦兼容测试和必要的全套测试，再生成部署快照。
- ownership 以 `docs/ledger/decisions.md` 为准：`experiments/nonlocal-current/` 属于本项目；其他实验目录的内容默认由其 owner 管理。跨 ownership 修改必须先确认 owner 状态；只读消费其他模块产物不转移其所有权。
- 每次 Kunshan 作业的 execution record 必须绑定**实际部署快照**的 commit、dirty/patch identity 和外部已接受 refs；不能只记录本地开发 HEAD。若快照含未提交文件，应保存确定性 source manifest/patch SHA-256，直至代码永久化为 Git commit。

## 当前 Current/VSV 工作的应用

- existing-VSV smoke 应先连接 Kunshan 上的真实已有 VSV，而不是因为本地没有 VSV 就停留在 synthetic fixture。
- 必须显式核验 VSV 的磁盘时间轴是 `source-relative` 还是 `source-sink`，不得根据 shape 猜测。
- 若目标配置缺少兼容 VSV，可在 Kunshan 生成该 VSV；生成完成后继续执行 directed-current artifact、逐 term 跨时收缩和真实 smoke。
- 真实 smoke 只证明文件、轴、端点、哈希和收缩工作流可复现；Ward--Takahashi、荷电归一化等物理结论仍需相应真实观测量和统计分析。

## 完成与阻塞口径

- “本地没有数据”不是完成条件，也不是阻塞理由。
- 只有在完成 Kunshan 只读查找、必要补算尝试和作业结果核验后，才能把某项数据或计算声明为缺失/失败。
- 若工作依赖正在运行的 Slurm 作业，应记录台账并由 monitor 唤醒后继续，而不是提前结束整个任务。
- 最终交付必须区分：本地代码验证、Kunshan artifact smoke、真实规范场测量、物理结论；不得把前一层的通过冒充后一层。
