# 永久化与 Kunshan 上线

详细的永久化规则已归档到 [`docs/archive/2026-09-03-kunshan-permanence.md`](archive/2026-09-03-kunshan-permanence.md)。本文件只保留入口和当前原则。

## 日常开发

日常开发不需要阅读完整永久化归档。使用：

1. `AGENTS.md`：强制规则；
2. `README.md`：最小本地开发流程；
3. `TASKBOARD.md`：当前阶段；
4. `PLAN.md`：长期计划。

## 何时阅读归档

准备以下操作时再阅读归档：

- 向 Kunshan 上传源码或运行 DCU/Slurm；
- 与其他 agent 合并共享模块；
- 生成 source/environment/input manifest；
- 提交或 push 交付内容；
- 归档大型数据或结果；
- 进行 fresh-clone/release 验证。

## 永久化原则

- 源码永久身份是 Git commit 和 remote ref；dirty snapshot 必须另存 source manifest/patch hash。
- Kunshan 大型数据不进入普通 Git；保存绝对路径、configuration、shape、dtype、轴语义、SHA-256、producer 和兼容性记录。
- 每个结果使用新目录，原子写出 result/manifest，最后创建 `DONE`；Slurm 任务必须有持久 monitor。
- 外部 agent 变化若与路径、API、schema、数据格式、runtime 和消费公式均无交集，记录 ref 后原样接受；共享变化必须兼容审查。
- 禁止覆盖 shared checkout，禁止 `reset --hard`、`git clean`、force checkout 和 `rsync --delete`。
- release-ready 还要求 Git-clean、source/doc/manifest commits 已 push、fresh clone 可验证，以及 retention decision 已解决。
- retention 使用 `approved`（secondary copy + hashed restore check）或用户签署的 `not-required`；pending 必须阻止 release-ready。

## 当前入口

- 当前阶段：`TASKBOARD.md`；
- 长期路线：`PLAN.md`；
- 交付边界：`INVENTORY.md` 和 `docs/delivery-files.list`；
- exact-byte manifest：`docs/handover-manifest.json`；
- 完整规则：[`docs/archive/2026-09-03-kunshan-permanence.md`](archive/2026-09-03-kunshan-permanence.md)。
