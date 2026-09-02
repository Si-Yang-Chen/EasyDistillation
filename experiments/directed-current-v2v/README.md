# Existing VSV directed-current smoke

`contract_existing_vsv.py` 将持久化 directed-current V2V artifact 与一个**已有**完整 VSV NPY 做逐 term 跨时收缩。它用于验证文件、哈希、时间轴、端点、Ne 和收缩协议，不是 Ward--Takahashi 或荷电归一化的物理验证。

## Kunshan 数据策略

真实规范场、eigenvector、传播子和历史结果主要在 Kunshan。先在集群查找、核验并复用已有 VSV；本地没有数据不是停止理由。如果目标配置在 Kunshan 上确实没有兼容 VSV，可按根目录 `AGENTS.md` 和站点 Slurm 规则补算，完成后继续本 smoke。

## VSV 输入契约

本版本支持两种显式输入布局：

1. 单个完整 rank-6 complex NPY：

```text
(source_time, second_time, sink_spin=4, source_spin=4, sink_ne, source_ne)
```

2. 完整 source-timeslice rank-5 文件族；每个文件固定一个 source time：

```text
(second_time, sink_spin=4, source_spin=4, sink_ne, source_ne)
```

Kunshan 当前 VSV 数据使用第二种 `source-time-rank-slab` 布局。例如配置 `10000` 的 `nev1` smoke 数据有 `t000…t071` 共 72 个文件，每个 shape 为 `(72,4,4,1,1)`。其生成/消费约定将内部时间轴作为 `source-relative`。

第二时间轴不能根据 shape 猜测，必须显式声明：

- `--vsv-time-axis source-relative`：磁盘索引为 `VSV[t_source, (t_sink-t_source) % Lt]`；
- `--vsv-time-axis source-sink`：磁盘索引为 `VSV[t_source, t_sink]`。

单个 rank-5 source-timeslice 文件不在本切片支持范围内。CLI 直接只读 mmap 声明的磁盘 block，不调用 `Propagator.get()`，不执行隐藏 gamma5 重构、dagger 或 transpose。

## Kunshan source-timeslice 审计

先为完整文件族生成逐文件 SHA-256 manifest。manifest 生成器会检查所有 source time 均存在、rank/shape/dtype 一致，并拒绝覆盖已有输出：

```bash
python experiments/directed-current-v2v/build_vsv_timeslice_manifest.py \
  --pattern '/public/home/siyangchen/qedinf/data/.../10000.t{source_time:03d}.rank0000.npy' \
  --configuration 10000 \
  --temporal-extent 72 \
  --time-axis source-relative \
  --output /absolute/new/audit/10000.vsv-timeslices.manifest.json
```

## Kunshan 运行示例

正式 smoke 还必须提供 execution record，记录 Git commit/dirty、资源、执行 Job ID、输入哈希和轴声明。CLI 不只检查 JSON 自洽性，还要求 trusted launcher 环境与 record 逐项一致：

```bash
export LATTICE_EXECUTION_CLUSTER=kunshan
export LATTICE_SOURCE_GIT_COMMIT=<SOURCE_COMMIT>
export LATTICE_SOURCE_GIT_DIRTY=true   # 或 false
export LATTICE_EXECUTION_RESOURCES_JSON='{"mode":"login-node-readonly-artifact-smoke","nodes":1,"cpus":1,"dcu":0}'
export LATTICE_EXECUTION_JOB_ID=none-login-readonly
```

Slurm 作业无需设置最后一项，CLI 会优先读取真实 `SLURM_JOB_ID`。execution-record schema 是 `lattice.current.kunshan-execution-record/v1`，其 `cluster`、`git`、`resources`、`slurm_job_id` 必须与上述受信运行环境完全匹配，并以 canonical JSON（排除 `record_identity`）计算 SHA-256 identity。

先 dry-run；dry-run 会解析真实 current endpoints 并读取将访问的 VSV blocks，但不做 einsum、不写结果：

```bash
python experiments/directed-current-v2v/contract_existing_vsv.py \
  --vsv-timeslice-pattern '/public/home/siyangchen/qedinf/data/.../10000.t{source_time:03d}.rank0000.npy' \
  --vsv-timeslice-manifest /absolute/audit/10000.vsv-timeslices.manifest.json \
  --vsv-time-axis source-relative \
  --current-artifact /absolute/kunshan/path/current-artifact \
  --result-dir /absolute/kunshan/new-result-dir \
  --configuration 10000 \
  --source-time 0 --sink-time 8 --current-time 4 \
  --current-source-ne 1 --current-sink-ne 1 \
  --current-direction 3 \
  --momentum-index 0 \
  --expected-gauge-sha256 GAUGE_SHA256 \
  --expected-eigenvector-sha256 EIGENVECTOR_SHA256 \
  --dry-run
```

正式运行去掉 `--dry-run` 并增加：

```text
--execution-record /absolute/audit/10000.execution-record.json
```

单 rank-6 VSV 仍可使用 `--vsv PATH --vsv-sha256 HASH`，其他参数相同。

`--current-direction` 必须显式选择守恒流分量：`0=x, 1=y, 2=z, 3=t`。时间/跨时 smoke 使用 `3`，CLI 只收缩该方向的 forward/backward 两个 term，不会把四个 current 分量静默求和。

去掉 `--dry-run` 才执行收缩并发布结果。正式输出目录必须不存在。

若 Current artifact 记录的原始 gauge/eigenvector 文件当前未挂载，可显式加 `--no-verify-current-sources` 做降级搬运 smoke；manifest 会记录 `sources_verified=false`，此结果不能作为来源审计通过证据。

## 输出

成功时原子发布：

- `contraction-<sha256>.npy`；
- `manifest.json`：输入路径/哈希、VSV 时间轴声明、实际访问磁盘索引、配置、时间、边界、动量、Ne、逐 term provenance、收缩公式和结果哈希；
- `DONE`：结果与 manifest 的字节哈希。

分类固定为：

```text
artifact-smoke-not-physics-validation
```

## 本地 CPU 回归

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  experiments/directed-current-v2v/test_contract_existing_vsv_cpu.py
```

测试分别构造单 rank-6 以及完整 rank-5 source-timeslice 文件族，并在 `source-relative` 与 `source-sink` 两种显式语义下证明收缩与 direct NumPy reference 一致。还覆盖逐文件 hash manifest、execution record、hash/dtype/rank/非有限访问块、open-boundary dry-run、降级来源模式、已有输出和失败无 partial 目录。
