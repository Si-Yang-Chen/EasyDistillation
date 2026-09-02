## 2026-08-28: experiments/ 目录治理
- experiments/nonlocal-current/ 归 orchestrator-lattice-flow 主管
- experiments/localized-blending/ 归 orchestrator-localized-blending 主管
- nonlocal-current 有权改 EasyDistillation 代码
- localized-blending 须确认 nonlocal-current 空闲才能改 EasyDistillation 代码

## 2026-09-03: multi-agent remote / Kunshan integration policy
- 上线前重新 fetch 并核对 origin refs、Kunshan checkout HEAD/dirty、实验目录 owner 和活动作业；旧会话状态不作为部署依据。
- 与 nonlocal-current 修改路径及其 API/schema/data/runtime 依赖均不相交的其他 agent 更改，记录来源 commit 后原样接受，无需逐行审查或专项测试；不得为了本项目回退它们。
- 触及共享 `lattice/__init__.py`、generator/elemental、insertion、quark_diagram、preset/propagator/slab I/O 或 Current schema 的更改属于交叉依赖，必须兼容审查与合并测试。
- Kunshan 部署使用新的隔离源码目录；禁止覆盖共享 checkout 的 dirty 文件，禁止 `reset --hard`、`clean`、force checkout 或无差别同步。
- execution record 绑定实际部署快照，包括共同基线 refs 与未提交 source manifest/patch hash；仅记录开发 HEAD 不足以审计 dirty 快照。
- 具体 HEAD/ref/dirty/作业状态属于动态观察，只写入 `TASKBOARD.md` 的时间戳 snapshot，不写入稳定 decision ledger。
