# 项目执行约定

## 适用范围

本文件适用于本地开发、Kunshan 数据核验、DCU 作业和多 Agent 协作。详细永久化与上线流程归档于 [`docs/archive/2026-09-03-kunshan-permanence.md`](docs/archive/2026-09-03-kunshan-permanence.md)。当前阶段见 [`TASKBOARD.md`](TASKBOARD.md)，长期计划见 [`PLAN.md`](PLAN.md)。

## 开发原则

- 本地主要用于源码开发、CPU 合成测试和文档维护；真实 gauge、eigenvector、VSV/PSV/VSP/PSP、elemental、关联函数和历史作业产物主要在 Kunshan。
- 本地没有数据不是阻塞理由；需要真实数据时先在 Kunshan 只读查找 manifest、历史结果和作业记录。
- 不猜测路径、配置号、shape、dtype、轴或物理公式。消费前记录绝对路径、配置、轴语义和 SHA-256，并核验兼容性。
- 已有且兼容的产物优先复用；确认缺失、损坏或不兼容后，允许按批准流程在 Kunshan 补算，包括传播子，但不得任意扩大资源。
- 不伪造数据、hash、作业状态或物理结论；合成测试必须标记为合成证据。

## 任务板历史

- `TASKBOARD.md` 只记录当前阶段；已使用的 taskboard 必须保留，不得覆盖或删除。
- 在阶段切换、重大部署、生产作业提交或交接前，将当前 taskboard 复制到 `docs/archive/taskboard/`，文件名使用 `YYYY-MM-DD-<phase>.md`，并保留当时的 HEAD、测试、Job ID 和 blocker 原文。
- 历史快照只读保存；后续状态更新只修改 `TASKBOARD.md`，不得回写旧快照。新的快照必须使用新文件名。
- taskboard 历史属于交付 lineage；新增快照后同步更新 `docs/delivery-files.list`、`docs/handover-files.list`（若需要交接）和 `docs/handover-manifest.json`。

## Kunshan 规则

- 登录节点没有 DCU。GPU/CuPy/PyQUDA/QUDA 必须通过 Slurm 计算节点；登录节点只做 CPU 只读检查、manifest/hash 校验和轻量 NumPy 工作。
- DCU 作业遵循 `.cursor/skills/dcu-slurm-submit/SKILL.md`，使用批准的 partition/account/MPI ABI/资源边界、独立结果目录、作业台账和持久 monitor。
- 每个作业绑定实际 source snapshot、输入 manifest、资源参数、Job ID 和轴约定；结果原子写出，先有机器可读 result/manifest，最后创建 `DONE`。
- existing-VSV 必须显式声明 `source-relative` 或 `source-sink`；不得根据 shape 猜时间轴。
- 真实 smoke 只证明文件、轴、端点、hash 和工作流可复现，不等于 WT、荷电归一化或其他物理结论。

## 多 Agent 与上线

- 每次上线前重新 `git fetch --prune`，核对本地 remote ref、Kunshan shared checkout 的 HEAD/branch/dirty、`~/qedinf/experiments/*` owner 及活动作业。
- 本项目专用 remote branch 的 push 已获持续授权；该授权不包括 force push、覆盖其他 agent 分支、修改 shared checkout 或绕过 retention/科学门禁。
- 禁止回退、覆盖、删除或清理其他 agent 的提交、分支、worktree、dirty 文件、实验脚本和结果；共享 checkout 禁止 `reset --hard`、`git clean`、强制 checkout 和 `rsync --delete`。
- 以 ownership 和实际依赖分类外部变化：若与本项目文件、API、schema、数据格式、runtime 和消费公式均无交集，记录 ref/commit 后原样接受，无需逐行审查或专项测试；这不代表本项目为其正确性背书。
- 若触及共享 `lattice/__init__.py`、`lattice/generator/*`、`lattice/insertion/*`、`lattice/quark_diagram.py`、preset/propagator/slab readers、Current schemas 或 launcher，必须保留双方语义，做兼容审查和测试。
- 上线使用新的隔离源码/结果目录；execution record 绑定实际部署 snapshot、dirty/patch/source-manifest identity 和外部已接受 refs。

## 完成口径

最终交付必须区分：本地代码验证、Kunshan artifact smoke、真实规范场测量和物理结论。不能把前一层通过冒充后一层。
