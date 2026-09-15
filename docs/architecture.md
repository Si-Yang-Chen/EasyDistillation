# EasyDistillation 总体架构文档

> 本文是仓库的总体架构（architecture）文档，汇总自 `.cache/analysis/*.md` 分析底稿（10 份，均含 `file:line` 出处）、已成稿的模块文档 [docs/modules/](docs/modules/) 与源码本身。代码引用格式为 `path:line`，行号对应当前工作区（HEAD `e19a399`）。分模块细节请参阅各模块文档；本文只回答"整体如何组织、一次计算如何流动、哪些抽象是承重墙"。

---

## 1. 一页总览

**EasyDistillation 是一个格点 QCD distillation 框架的 Python 实现**：它把"给定若干 gauge configuration，计算强子两点关联函数（进而提取质量谱）"这一任务，拆解为一条固定流水线——

```
gauge → eigenvector → elemental → perambulator → quark diagram contraction → correlator
```

具体地，它解决的问题是：

1. **数据生成**：从 gauge 场（`GaugeFieldIldg` 读入、stout smearing、SU(3) 投影）出发，求解 3D covariant Laplacian 的低模本征矢（`EigenvectorGenerator`），再组合出 distillation 的核心中间量——elemental 矩阵元 `V†φV`（`ElementalGenerator`）与夸克传播体 perambulator `τ = V†S⁻¹V`（`PerambulatorGenerator`，基于 PyQUDA/QUDA Dirac 求逆）。
2. **算符构造**：把插值算符（介子/重子，含动量、导数、gauge link 位移）按立方群 `O_h` 不可约表示（irrep）投影成可收缩的稀疏符号形式（`lattice/insertion/` 与 `lattice/symmetry/`）。
3. **图收缩**：把强子间夸克线拓扑表示为"邻接矩阵 + einsum 下标"（`QuarkDiagram`），用 `opt_einsum.contract` 数值求值，支持 current 顶点、稀疏点源采样与 multitime 时间轴。
4. **关联函数输出**：`lattice/correlator/` 输出两点函数变体（单算符、矩阵、多动量、isoscalar 含 disconnected），example 脚本进一步做有效质量/GEVP 分析。

横切关注点：numpy/cupy 双后端透明切换（`lattice/backend.py`）、MPI 配置任务分发（`lattice/dispatch.py`）、惰性磁盘 I/O（`lattice/filedata/`）、结果溯源 manifest（`lattice/result_provenance.py`）。代码规模约 3.6 万行（`lattice/` 下 `wc -l` 实测 35,653 行，其中 `lattice/symmetry/hardcoded_rep.py` 的 15,735 行为生成代码查找表）。

---

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  应用层  example/*.py                                                    │
│  gen_twopt / gen_twopt_matrix_mom / gen_two_particle_corr*(D D*) /       │
│  gen_density_peram / gen_multi_draw_diagrams / Dispatch cfglist 循环     │
├─────────────────────────────────────────────────────────────────────────┤
│  收缩与关联层                                                            │
│  correlator/one_particle.py  (twopoint 家族，distillation 直收缩)         │
│  quark_diagram.py  (QuarkDiagram/Diagram/compute_diagrams_multitime,     │
│                     quark_contract, 采样权重)   quark_draw.py (可视化)     │
│  propagators.py    (Meson/Current/Propagator/PropagatorWithCurrent 句柄)  │
│  current_elemental.py (directed current V2V 工件持久化 + term-wise 收缩)  │
│  correlator/conserved_charge.py (守恒流显式投影/显式比值)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  insertion / 对称性层                                                    │
│  insertion/  (gamma, derivative, gauge_link, mom_dict, phase,            │
│               Insertion/InsertionRow/Operator, current.py 契约体系)       │
│  symmetry/   (group_generator → gen_hardcoded_rep → hardcoded_rep 冻结表) │
│  group_projection / spatial_structure / flavor_structure / hadron        │
├─────────────────────────────────────────────────────────────────────────┤
│  生成器层  generator/                                                    │
│  eigenvector (Laplacian 低模) → elemental (calc_deriv/calc_disp,         │
│  CurrentElementalGenerator v2v/v2p/p2v/p2p) → perambulator 族            │
│  (PerambulatorGenerator / Density / Generalized, PyQUDA 求逆)            │
│  noisevector (稀释噪声源) / sparsened_point (稀疏点源) / stout_smear.cu    │
├─────────────────────────────────────────────────────────────────────────┤
│  数据源层  preset.py                                                     │
│  物理语义基类 × 文件格式后端 多继承 → 25+ 个 load(cfg) 加载器              │
│  (GaugeFieldIldg / ElementalNpy / PerambulatorNpy / PropagatorPSVNpy …)  │
├─────────────────────────────────────────────────────────────────────────┤
│  基础设施层                                                              │
│  backend.py (numpy/cupy 单例 + check_QUDA)   dispatch.py (MPI 工作队列)   │
│  filedata/ (FileData 惰性 mmap I/O: binary/ildg/ndarray/timeslice)       │
│  data.py (Operator.parts → φ)   result_provenance.py (结果 manifest)     │
│  constant.py (Nc/Ns/Nd)   base_types.py (Tag/Flavor)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

分层规则（自上而下单向依赖为主）：

- **基础设施层**不依赖任何上层；`backend.py:10` 的 `get_backend()` 是全局数组命名空间单例，`filedata/abstract.py:12` 的 `FileData` 以抽象 `__getitem__`（`lattice/filedata/abstract.py:19`）定义惰性切片协议。
- **数据源层** `lattice/preset.py` 只依赖 `lattice/filedata/`（`lattice/preset.py:4-8`），用"格式基类在前、物理角色基类在后"的多继承组合出加载器（如 `class PerambulatorNpy(NdarrayFile, Perambulator)`，`lattice/preset.py:217`；`class GaugeFieldIldg(IldgFile, GaugeField)`，`lattice/preset.py:687`）。
- **生成器层**以 preset 的角色基类（`GaugeField`/`Eigenvector`/`PointSource`）为输入端口类型，把 insertion 层的 `MomentumPhase`/`gamma`/`GaugeLink` 作为数值工具。
- **insertion/对称性层**是纯符号层（sympy + 稀疏行编码），不读盘不做收缩；current 子系统（`lattice/insertion/current.py:394` 的 `CurrentTerm`）带 schema 化契约校验。
- **收缩与关联层**消费上两层的输出：`quark_diagram.py:19-26` 从 `propagators.py` 再导出全部句柄类；`lattice/correlator/one_particle.py:5-9` 直接消费 `Operator`、`data.get_elemental_data` 与 backend。
- **应用层** example 脚本演示典型端到端流程（π 2pt、D D* 两粒子、多动量 GEVP 矩阵、图绘制）。

---

## 3. 模块依赖图

以下箭头表示"谁 import 谁"，均为源码可查事实（grep 汇总自 `.cache/analysis/*.md`）：

```
                        ┌────────────────────────── 应用层 ─────────────────────────┐
                        │  example/gen_twopt*.py   example/gen_two_particle_corr*.py │
                        └─────┬──────────────┬──────────────────┬──────────────────┘
                              │              │                  │
              lattice.correlator.one_particle │   lattice.quark_diagram           │
                              │              │                  │
        ┌─────────────────────▼───┐   ┌──────▼──────────────────▼───────────┐
        │ correlator/             │   │ quark_diagram.py  ──re-export──►    │
        │  one_particle.py ───────┼──►│ propagators.py (Particle/Meson/     │
        │   └─ data.py:get_elemental_data │   Current/Propagator/…句柄)   │
        │  dispersion_relation.py │   │ quark_draw.py ─► quark_diagram.vertex_type_from_matrix
        │  two_particles.py ─► insertion/ │                                     │
        │  conserved_charge.py (仅 numpy，独立)  │ current_elemental.py ─► insertion/current.py
        └──────────┬──────────────┘   └──────┬──────────────┬───────────────┘
                   │                         │              │
     ┌─────────────▼─────────────────────────▼───┐  ┌───────▼──────────────────────┐
     │ insertion/                                │  │ 生成器层 generator/            │
     │  __init__.py (Row/Operator/Insertion…)    │  │  elemental.py ──► insertion/  │
     │   ├─► gamma.py ─────────► symmetry/       │  │   (phase/gamma/gauge_link/     │
     │   ├─► gauge_link.py ────► symmetry/       │  │    derivative, 惰性 import)    │
     │   ├─► current.py ─► gauge_link(DirectedCurrentBasis), gamma, quark_diagram.CurrentVertexAdapter
     │  gamma/derivative/gauge_link/mom_dict/phase │  │  perambulator.py ─► preset(类型) + pyquda
     └─────────────┬─────────────────────────────┘  │  displacement_elemental.py / eigenvector.py
                   │                                │  density_/generalized_perambulator.py
     ┌─────────────▼────────────────┐               │  noisevector.py / sparsened_point.py
     │ symmetry/                    │               └───────┬──────────────────────┘
     │  gen_hardcoded_rep.py ◄──────┼──────────► hardcoded_rep.py (15735 行冻结表, lazy Proxy)
     │  group_generator.py (生成元种子，无内部依赖)    │
     │  group_projection.py ◄► quark_diagram.Diagram │
     │  spatial_structure.py / flavor_structure.py / hadron.py ─► quark_diagram(quark_contract)
     └──────────────────────────────┘
                   ▲ preset 角色基类 (GaugeField/Eigenvector/PointSource/Perambulator…) 作为类型契约
        ┌──────────┴─────────────────────────────────────────────┐
        │ preset.py (数据源层) ──► filedata/{binary,ildg,ndarray,timeslice,abstract} │
        └──────────┬─────────────────────────────────────────────┘
                   ▼
     基础设施层: backend.py (被几乎全部计算模块 import)
                dispatch.py (仅被 example 使用)   constant.py (Nc/Ns/Nd)
                result_provenance.py (仅被 test/test_ne_provenance.py 引用)
                base_types.py (被 flavor_structure/spatial_structure/quark_diagram 使用)
```

要点摘录（均有底稿出处）：

- `preset.py → filedata/`：显式 import 五个格式基类（`lattice/preset.py:4-8`）。
- `generator/* → preset`：仅 import 角色基类作类型注解，如 `lattice/generator/elemental.py:50-52`、`lattice/generator/perambulator.py` 构造签名；运行期靠 duck-typing（`load(cfg)`）。
- `generator/elemental.py → insertion/`：顶层 import `MomentumPhase`（`elemental.py:7-11`），惰性 import `derivative`/`GaugeLink`/`build_current_raw_contract`（gen-core 底稿依赖表）。
- `insertion/gamma.py → symmetry/`：`genLittleGroupIrrep`、`group_element`（`lattice/insertion/gamma.py:3-4`）；`gauge_link.py → symmetry/`（`OD_irreps`、`irrep_row_connection_dict`，`lattice/insertion/gauge_link.py:7-9`）。
- `quark_diagram.py → propagators.py`（re-export `Particle/Meson/Current/Propagator/PropagatorLocal/PropagatorWithCurrent`，`lattice/quark_diagram.py:19-26`）；`→ spatial_structure.HadronIrrepRow`（`quark_diagram.py:19`）；函数体内惰性 import `flavor_structure`（`quark_diagram.py:3248-3251`）。
- `correlator/one_particle.py → data.get_elemental_data`（`lattice/correlator/one_particle.py:7`）与 `→ insertion`（`:5-6`）；`correlator/dispersion_relation.py → one_particle.twopoint`（`:8`）。
- `correlator/conserved_charge.py` 与 `result_provenance.py` 是零内部依赖的独立工具（仅 numpy/标准库）。
- 反向消费：`lattice/hadron.py:19` 用 `quark_diagram.diagram_simplify/diagram_vertice_replace/quark_contract`；`lattice/group_projection.py:7,78` 用 `Diagram`；`lattice/insertion/current.py:772` 用 `CurrentVertexAdapter`——即 insertion/current 与 diagram 层存在一处受控的双向接触（经惰性 import 避免循环）。

---

## 4. 核心数据流：以 π 介子 2pt 为例的端到端流程

对应 example：`example/gen_twopt.py`（底稿 usage.md §2.4）；以下编号步骤给出每步的 API 与数据形状。

**Step 0 — 后端选择**。`set_backend("cupy")`（GPU 可用时），此后所有数组运算走 `backend = get_backend()` 的统一命名空间（`lattice/backend.py:10`、`lattice/backend.py:17`；cupy 不可用时 example 脚本回退 numpy）。

**Step 1 — 构造插值算符（符号层）**。
`ins = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)` 按 O_h CG 表生成 `A_1` 表示的 1 行（`lattice/insertion/__init__.py:249`，`construct()` 在 `:300-417`）；`pi_A1[0](0,0,0)` 把 `InsertionRow` 固定到动量索引 0，得 `InsertionRowMom`（`lattice/insertion/__init__.py:141`、`__call__` 在 `:227-228`，动量字符串 `f"{npx} {npy} {npz}"` 必须在 `momDict_mom9` 中）；再包成 `Operator("pi", [pi_A1[0](0,0,0)], [1])`，其 `parts` 被展平为 `[gamma_idx, [[coeff, deriv_idx, mom_idx, profile], …], gamma_idx, …]`（`lattice/insertion/__init__.py:147`，展平逻辑 `:157-176`）。

**Step 2 — 声明数据源（lazy 加载器）**。
```
e = ElementalNpy(prefix, ".mom9.npy", [4, 123, 128, 70, 70], 70)   # lattice/preset.py:717
p = PerambulatorNpy(prefix, ".peram.npy", [128, 128, 4, 4, 70, 70], 70)  # lattice/preset.py:217
```
elemental 磁盘布局 `(num_disp, num_mom, Lt, Ne, Ne)`；perambulator 布局 `[t_src, t_rel, Ns_snk, Ns_src, Ne_snk, Ne_src]`（即 distillation 的 `τ = V†S⁻¹V`，由 `test/test_perambulator.py` 的 `np.roll(…, -t, 0)` 约定确证）。`load(cfg)` 不读数据，只返回惰性 `FileData` 句柄（`lattice/filedata/abstract.py:12`，抽象 `__getitem__` 在 `:19`；首次切片才 mmap 读盘，`lattice/filedata/ndarray.py:17-46`）。

**Step 3 — elemental → φ（算符矩阵元组装）**。
`phis = get_elemental_data(operators, elemental, usedNe)`（`lattice/data.py:7-36`）：按 `Operator.parts` 成对遍历，偶数位 gamma 编号经 `gamma(n)`（`lattice/insertion/gamma.py:95`，按 n 的 4 个 bit 连乘 Dirac 矩阵）转 4×4 矩阵；奇数位元素项以 `(derivative_idx, momentum_idx)` 为键缓存切片 `elemental[derivative_idx, momentum_idx, :, :usedNe, :usedNe]`（形状 `(Lt, usedNe, usedNe)`，`lattice/data.py:27-29`），同行内多系数项线性叠加。每算符产出 `phi = (ret_gamma, ret_elemental)`，形状分别为 `(行数, 4, 4)` 与 `(行数, Lt, usedNe, usedNe)`。

**Step 4 — perambulator 逐源时刻切片**。
对每个 `t_src in timeslices`：`tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]`，形状 `(Lt_sink, 4, 4, usedNe, usedNe)`（`lattice/correlator/one_particle.py:41`）。构造 backward 项 `tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))`——`gamma(15)` 即 γ5（`gamma.output(15) == "γ5"`，`lattice/insertion/gamma.py:79-81`）；源侧 spin 结构 `gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi[0].conj(), gamma(8))`（`lattice/correlator/one_particle.py:57-63`）。

**Step 5 — 主 einsum 收缩**（`lattice/correlator/one_particle.py:64-73`）：
```
ret[idx, it] = contract("tijab,xjk,xtbc,tklcd,yli,yad->t",
    tau_bw, phi[0], backend.roll(phi[1], -t_src, 1), tau, gamma_src, phi[1][:, t_src].conj())
```
下标语义：`t` = perambulator 的 sink 时间轴；`i,j` 两个 spin 轴、`a,b` 两个 Ne 轴；`x` = 算符行（gamma 分量）；`y` = 源侧 gamma 结构行；`c,d` = elemental 源端 Ne 轴。`roll(phi[1], -t_src, 1)` 把 elemental 的 sink 时间对齐到相对时间 `t = t_sink − t_src`。

**Step 6 — 输出与后处理**。
返回 `-ret.mean(1)`（源时刻平均、约定负号），形状 `(Nop, Lt)`（`lattice/correlator/one_particle.py:74-77`）。example 脚本随后用 `arccosh((C(t+1)+C(t-1))/(2C(t)))` 打印有效质量谱。

**旁路对照（两粒子/守恒流通路）**：D D* 两粒子 2pt 走另一条路——`Meson(elemental, operator, source)` 与 `Propagator/PropagatorLocal` 句柄（`lattice/propagators.py:188`、`:742`、`:863`）被 `QuarkDiagram(adjacency_matrix, vertex_list, …)` 拓扑 + `compute_diagrams_multitime`（`lattice/quark_diagram.py:1646`）消费，`t_snk = numpy.arange(Nt)` 触发 multitime 下标（数组时间对象判定在 `lattice/quark_diagram.py:1675-1685`，须用 numpy 而非 cupy 数组）。守恒流走第三条路：`ConservedVectorCurrent.terms`（`lattice/insertion/current.py:1160`）→ `build_current_raw_contract`（`lattice/insertion/current.py:112`）→ `contract_directed_current_v2v`（`lattice/current_elemental.py`）→ `correlator.conserved_charge` 显式投影。

---

## 5. 关键抽象表

| 抽象 | 职责 | 定义位置 | 主要消费者 |
|---|---|---|---|
| **FileData 惰性加载** | 以抽象 `__getitem__` 定义"索引时才 mmap 读盘"协议；`FileMetaData(shape, dtype, extra)` 携带布局元数据；四个格式后端（binary 裸二进制、ildg ILDG/LIME、ndarray `.npy`/`.t???` 时间片、timeslice QDP）统一实现 | `lattice/filedata/abstract.py:12`（`FileData`）、`:19`（`__getitem__`）、`:5`（`FileMetaData`） | `lattice/preset.py` 全部加载器（`preset.py:4-8` import）；`lattice/data.py:7`；`lattice/correlator/one_particle.py:8`；`lattice/correlator/dispersion_relation.py:4`；约 10 个物理测试（`test/test_load.py` 等） |
| **backend 切换** | 以"模块对象作为全局后端单例"实现 numpy/cupy 透明切换；`check_QUDA` 幂等探测 PyQUDA（失败返回 False 不上抛）；`log_gpu_memory` 显存/RSS 日志 | `lattice/backend.py:10`（`get_backend`）、`:17`（`set_backend`，assert 仅 numpy/cupy）、`:39`（`check_QUDA`）、`:70`（`log_gpu_memory`） | 几乎全部计算模块（`lattice/data.py:8`、全部 `lattice/filedata/*.py`、`lattice/generator/*`、`lattice/correlator/one_particle.py:9`、`lattice/quark_diagram.py:21`）；perambulator 系硬性要求 cupy（`lattice/generator/perambulator.py` 构造内 assert） |
| **Dispatch（MPI 任务分发）** | 把 cfg 列表文本做成跨 MPI 进程工作队列：rank 0 独占锁内 pop+truncate 弹出即消费、`bcast` 广播，配合 `AtomicOpen` 原子文件与 `combine` 原子追加；依赖 mpi4py | `lattice/dispatch.py:62`（`Dispatch`）、`:34`（`AtomicOpen`）、`:76`（`__iter__`）、`:107`（`combine`） | 仅 example 脚本（`example/gen_twopt_matrix_mom.py:6,57-60`、`example/gen_two_particle_corr.py:15`、`example/gen_two_particle_corr_mom.py:17`）；`lattice/__init__.py:2` 导出；test/ 无直接测试 |
| **QuarkDiagram einsum 编译** | 把邻接矩阵 + 顶点列表编译为"连通分量 × propagator/vertex 下标"的 einsum 计划：BFS 分连通分量、按 propagator 端点状态（v/p）定 VSV/VSP/PSV/PSP 类型、current 顶点按 4^k 状态展开、`opt_einsum.contract` 求值；`Diagram`（sympy Symbol 包装）提供符号化简/群变换；prepare/bind/eval 三段式支持按 cfg/时间循环复用 | `lattice/quark_diagram.py:507`（`QuarkDiagram`）、`:366`（`validate_adjacency_matrix`）、`:724`（`analyse_v2v`）、`:1646`（`compute_diagrams_multitime`）、`:2068`（`Diagram`）、`:2971-3223`（`calc_diagram_prepare/bind/eval`） | `lattice/hadron.py:19,214-222`、`lattice/group_projection.py:7,78`、`lattice/insertion/current.py:772`（`CurrentVertexAdapter`）、`lattice/quark_draw.py:6`、example 两粒子脚本 |
| **Operator / InsertionRow** | 把"gamma 通道 × derivative/gauge-link 通道 × O_h irrep 投影 × 动量"编码为稀疏行 `Row = [gamma_idx, [[coeff, deriv_idx], …], …]`；`InsertionRow(npx,npy,npz)` 固定动量；`Operator.parts` 是收缩层的正式数据契约（偶数位 gamma 索引、奇数位 `[coeff, deriv_idx, mom_idx, profile]` 列表） | `lattice/insertion/__init__.py:147`（`Operator`，parts 展平 `:157-176`）、`:221`（`InsertionRow`）、`:141`（`InsertionRowMom`）、`:249`（`Insertion`，CG 表 `construct()` `:300-417`）、`:202`（`OperatorDisplacement`，deriv_idx 换 GaugeLink 距离） | `lattice/data.py:16-18`（按 parts 取 elemental）、`lattice/correlator/one_particle.py:5-6`、`lattice/propagators.py`（`Meson._make_cache` 按 parts 组缓存）、example 全部 2pt 脚本 |
| **schema/manifest 校验** | 三套内容寻址/契约机制：(a) 结果目录 manifest——规范化 JSON → `config_sha256` → `run_id` → 目录路径，validate 时重放 build 比对，产物 finalize 时登记哈希；(b) current raw 契约——`CurrentTerm` dataclass 逐字段校验 + raw 数组形状/Ne 元数据/cache_identity 指纹；(c) directed V2V 工件——内容寻址文件名 `v2v-{sha256}.npy`、源文件哈希绑定、TOCTOU 双读 | `lattice/result_provenance.py:120`（`build_manifest`）、`:194`（`validate_manifest`）、`:226`（`load_result_manifest`）、`:258`（`finalize_result`）；`lattice/insertion/current.py:394`（`CurrentTerm`）、`:112`（`build_current_raw_contract`）；`lattice/current_elemental.py`（save/load_directed_current_v2v） | `test/test_ne_provenance.py`（manifest 全流程）；`lattice/insertion/current.py:772` 与 `lattice/quark_diagram.py`（`CurrentVertexAdapter` 消费 assembler schema）；`test/test_current_v2v_persistence.py`、`test/test_current_v2v_contraction.py` |
| **（补充）Meson/Propagator 句柄缓存** | 把 preset 加载器包装为收缩消费的顶点/两时间传播子；`load()` 派生 identity（key + usedNe/usedNp + loader 签名含文件 stat + 数据签名含 SHA-256 + operator 签名 + dagger），identity 变化全量重建缓存；`PropagatorWithCurrent` 另提供 VSP/PSV/PSP 高模投影（low-mode subtraction） | `lattice/propagators.py:188`（`Meson`）、`:327`（`Current`）、`:742`（`Propagator`）、`:921`（`PropagatorWithCurrent`）、`:46`（`_normalize_extent`）、`:149`（`_loader_signature`） | `lattice/quark_diagram.py:19-26` re-export 后供 contraction 层与 example 使用；`test/test_ne_provenance.py:61-137` 验证 Ne/loader 变化触发重建 |

---

## 6. 目录结构表

`lattice/` 顶层（行数为 `wc -l` 实测）：

| 路径 | 行数 | 职责 | 模块文档 |
|---|---|---|---|
| `lattice/__init__.py` | 64 | 包公开门面：re-export backend/Dispatch/preset 加载器/generator/quark_diagram/current_elemental 全部公开 API | — |
| `lattice/backend.py` | 133 | numpy/cupy 后端单例、PyQUDA 探测、显存/内存日志 | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/dispatch.py` | 109 | MPI cfg 工作队列（AtomicOpen 原子锁 + pop-and-broadcast） | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/base_types.py` | 11 | `Flavor` 字面量类型与夸克线追踪 `Tag(tag, time)` | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/constant.py` | 3 | `Nc=3, Ns=4, Nd=4` | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/data.py` | 36 | `Operator.parts` + elemental → 收缩输入 φ（带 (deriv, mom) 缓存） | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/result_provenance.py` | 280 | 结果目录溯源 manifest（规范化 JSON 哈希 + git worktree 指纹 + 产物哈希） | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/filedata/abstract.py` | 26 | `FileMetaData` / `FileData`（惰性切片协议）/ `File`（抽象工厂） | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/filedata/binary.py` | 74 | 无 header 裸二进制惰性读取 | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/filedata/ildg.py` | 104 | ILDG/Lime 规范场解析（magic + XML + 数据偏移） | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/filedata/ndarray.py` | 118 | `.npy` 单文件版与 `.t???` 时间片版 | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/filedata/timeslice.py` | 129 | QDP `QDPLazyDiskMapObjFile`（偏移表 + per-record mmap） | [docs/modules/infrastructure.md](docs/modules/infrastructure.md) |
| `lattice/preset.py` | 1021 | 物理数据源预设层：12 个角色基类 × 5 种格式后端 → 25+ 加载器 | [docs/modules/preset.md](docs/modules/preset.md) |
| `lattice/generator/eigenvector.py` | 530 | 3D covariant Laplacian 低模本征矢（scipy/cupyx eigsh 与 QUDA TR-LANCZOS/Chebyshev 多路径） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/elemental.py` | 2583 | `ElementalGenerator`（calc_deriv/calc_disp）与 `CurrentElementalGenerator`（v2v/v2p/p2v/p2p/calc_all） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/displacement_elemental.py` | 215 | 对称化位移 elemental（六方向协变位移递推平均） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/perambulator.py` | 759 | PyQUDA Dirac 求逆 → VSV/PSV/PSP（calc_old 参考实现 / calc_new 向量化实现 + MPI 点归属） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/density_perambulator.py` | 176 | 密度插入广义 perambulator VSSV（逐空间点、固定 tau） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/generalized_perambulator.py` | 184 | 同上但保留全部 Lt 汇时刻（双 CUDA 流重叠） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/noisevector.py` | 110 | 本征矢空间稀释噪声源（相位/QR 酉投影 + high-mode 正交补） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/sparsened_point.py` | 114 | 每时间片独立去重的稀疏点源坐标表 | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/generator/stout_smear.cu` | 245 | 单步 stout link smearing CUDA kernel（与 Python 实现同一公式） | [docs/modules/generator.md](docs/modules/generator.md) |
| `lattice/insertion/__init__.py` | 567 | 符号插入算符组装：`Row`/`Insertion`/`Operator`/`OperatorDisplacement` + re-export current | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/insertion/gamma.py` | 349 | 16 个 Dirac gamma 矩阵位编码、介子通道命名/量子数表、O_h 群变换表 | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/insertion/derivative.py` | 132 | ∇/𝔹/𝔻/𝔼 导数组合的三进制索引编码与 irrep 行 | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/insertion/gauge_link.py` | 586 | gauge link 路径符号代数（str/list/idx 三表示 + 共轭/群变换）、`DirectedCurrentBasis` 8 方向基 | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/insertion/mom_dict.py` | 203 | 动量索引 ↔ "npx npy npz" 字符串四张静态字典 | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/insertion/phase.py` | 53 | `MomentumPhase` 平面波相位缓存（全格点与 even-odd cb2 布局） | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/insertion/current.py` | 1220 | current/density 插入契约体系：`CurrentTerm`、raw contract 校验、端点解析、自旋桥接、WT/PCAC 校验、5 个算符类 | [docs/modules/insertion.md](docs/modules/insertion.md) |
| `lattice/symmetry/`（6 文件） | 16,936 | 立方群 O_h 双群表示：`group_generator.py` 生成元种子 → `gen_hardcoded_rep.py` 在线构造 → `hardcoded_rep.py` 15,735 行冻结查找表（lazy Proxy）；`two_particle.py` 连续 J^LS 两粒子算符旁路 | [docs/modules/symmetry-and-operators.md](docs/modules/symmetry-and-operators.md) |
| `lattice/group_projection.py` | 319 | sympy 表达式按动量小群 irrep 行投影（多强子乘积算符支持） | [docs/modules/symmetry-and-operators.md](docs/modules/symmetry-and-operators.md) |
| `lattice/spatial_structure.py` | 337 | `HadronIrrep`/`HadronIrrepRow`（小群协变变换的 sympy 符号原子） | [docs/modules/symmetry-and-operators.md](docs/modules/symmetry-and-operators.md) |
| `lattice/flavor_structure.py` | 142 | `Qurak`/`Propagator`/`HadronFlavorStructure` 味结构符号 | [docs/modules/symmetry-and-operators.md](docs/modules/symmetry-and-operators.md) |
| `lattice/hadron.py` | 229 | `Hadron`（空间×味结构配对）与 `gen_correlator` 多点关联表达式生成 | [docs/modules/symmetry-and-operators.md](docs/modules/symmetry-and-operators.md) |
| `lattice/propagators.py` | 2344 | 收缩句柄层：`Meson`/`Current` 顶点、`Propagator`/`PropagatorLocal`/`PropagatorWithCurrent`（VSV/VSP/PSV/PSP + 高模投影），identity 失效缓存 | [docs/modules/propagators.md](docs/modules/propagators.md) |
| `lattice/current_elemental.py` | 721 | directed current V2V 工件持久化（内容寻址 + 源哈希绑定）与 term-wise/pair V2V 纯函数收缩 | [docs/modules/propagators.md](docs/modules/propagators.md) |
| `lattice/quark_diagram.py` | 3387 | quark 图收缩核心：邻接矩阵 + einsum 编译、current 状态/采样场景展开、`Diagram` 符号代数、`quark_contract` 味收缩、采样组合数学 | [docs/modules/diagram.md](docs/modules/diagram.md) |
| `lattice/quark_draw.py` | 473 | Feynman 风格收缩图可视化（feynman + matplotlib；import 有副作用） | [docs/modules/diagram.md](docs/modules/diagram.md) |
| `lattice/correlator/one_particle.py` | 413 | 单粒子两点函数 7 变体（twopoint/profile/indice/matrix/isoscalar/multi_mom） | [docs/modules/correlator.md](docs/modules/correlator.md) |
| `lattice/correlator/two_particles.py` | 44 | 两粒子动量壳枚举 + A/B 算符列表构造 | [docs/modules/correlator.md](docs/modules/correlator.md) |
| `lattice/correlator/dispersion_relation.py` | 32 | 动量壳平均算符构造 + 壳平均 2pt | [docs/modules/correlator.md](docs/modules/correlator.md) |
| `lattice/correlator/conserved_charge.py` | 167 | 守恒流显式标量投影与显式比值（约定中性、带 schema 审计） | [docs/modules/correlator.md](docs/modules/correlator.md) |

---

## 7. 当前已知问题清单

以下汇总自各底稿的 "candidate blockers / 未验证项" 小节（均为分析结论，非最终定论；出处同时给出源码位置与底稿文件）。行号已按当前工作区（HEAD `e19a399`）核实或修正。

### 7.1 数值正确性类（优先确认）

1. **`displacement_elemental` 与参考数据失配**。`lattice/generator/displacement_elemental.py:95-96` 的 `calc` 同时累加 `<V|D^dist|V>` 与 `<V|(D^dist)†|V>` 两项（第二项由提交 `8923174` 引入），而参考文件在 `dist=0, mom=(0,0,0)` 处为 `1.0`（当前代码给 `2.0`）；实测 `python test/test_displacement_elemental.py` 残差 `36.25`（对照 `test_elemental.py` 为 `8.7e-14`），且单项重算仍余 `21.34`——参考数据由哪个版本生成未定位。出处：[.cache/analysis/gen-core.md](../.cache/analysis/gen-core.md) §3、Unverified §1。
2. **`_stout_smear_quda` 属性名不一致**。`lattice/generator/displacement_elemental.py:195` 读取 `self._gauge_field_path`，但 `load()` 在 `:75` 写入的是 `self._gauge_field_data`——走 cupy + 无 CUDA kernel + QUDA 分支时将 `AttributeError`；numpy 后端测试（走 `_ndarray` 分支）不覆盖。同类代码在 `ElementalGenerator`/`CurrentElementalGenerator` 中属性名一致，仅此一处错位。出处：gen-core 底稿 §7。
3. **`twopoint_profile` 的 `perambulator_bw` 切片不对称**。forward 项切片带 `delta_t_slices`（`lattice/correlator/one_particle.py:101`、`:156`），而 backward 项 `tmp = perambulator_bw[t_src, :, :, :, :usedNe, :usedNe]` 未切片（`lattice/correlator/one_particle.py:107`、`:162`）——`delta_t_slices` 为真子集时两分支时间轴长度不一致，可能 einsum 尺寸报错或语义漂移。出处：[.cache/analysis/correlator.md](../.cache/analysis/correlator.md) §7。
4. **`twopoint_indice` 的 `is_diagonal` 形参未使用**。签名在 `lattice/correlator/one_particle.py:138` 声明 `is_diagonal=False`，但函数体（`:138-179`）无任何分支引用，恒按全 `Ne×Ne` 收缩——与 `twopoint_profile`（`:96-98` 生效）行为不一致。出处：correlator 底稿 §1。
5. **`density_perambulator` 的 γ5 对合疑似被重复施加**。`lattice/generator/density_perambulator.py:147` 的 `SV_f ← γ5·conj(SV_f)·γ5` 位于 `for eigen` 循环内、`if tf != self._tf:` 缩进块之外，每次 eigen 迭代重复执行；因该变换为对合（T∘T = id），第 k 行被作用 `Ne−k` 次，偶数次输出未共轭传播子。`generalized_perambulator.py:155` 把同一行放在 if 块内（单次），佐证此为缩进笔误。未运行复现（需 GPU）。出处：[.cache/analysis/gen-peram.md](../.cache/analysis/gen-peram.md) §7 blocker-2。
6. **`density_perambulator` 缓存键不含 tau**。`self._t/_tf`（`lattice/generator/density_perambulator.py:69-70`）只缓存 ti/tf，而 `[:, tau, …]` 切片发生在反演时（`:145` 附近）——同 (ti, tf) 不同 tau 连续调用会沿用旧 tau 数据。出处：gen-peram 底稿 §7 blocker-5。
7. **`PerambulatorGenerator.calc_old` 的 PSV 抽取路径失效**。`lattice/generator/perambulator.py:435` 起的 `for t_snk in range(Lt):` 循环体仅 `continue`，循环后 `t_snk` 恒为 `Lt-1`；且点归属判断用半格距（`gx*(Lx//2)`）与 `sampled_point_local_indices` 的全格距矛盾、parity 用全局坐标。`calc_new`（`:473`）已用预计算索引绕开，但 `use_vectorized=False` 时 `calc()`（`:754` 起）仍默认走 calc_old。出处：gen-peram 底稿 §2.6、§7 blocker-1。
8. **`PerambulatorGenerator` MPI 测试与现行 API 契约矛盾**。`test/test_perambulator_mpi.py:33` 用 `eigenvector=eigenvector` 关键字构造，而现行签名只有 `eigenvector_src`（`lattice/generator/perambulator.py` 构造函数）——该测试按现状会 `TypeError`。出处：gen-peram 底稿 §7 blocker-3。
9. **current + 稀疏采样的多场景加权求和疑似未闭环**。`SceneExpandedDiagram.unify_vertex_point_color_indices`（`lattice/quark_diagram.py:1570`）函数体只有 debug 打印、无任何下标写操作；`compute_diagrams_multitime` 只乘 `scene_weights[0]`（`lattice/quark_diagram.py:1926-1927`）且 `scene_diagrams` 未被逐场景求值。实际效果可能是每图只按第一场景权重缩放，与 docstring 宣称的"所有场景加权和"不一致；也可能求和在调用方完成——需按 diagram 底稿 §20 的探针确认。出处：[.cache/analysis/diagram.md](../.cache/analysis/diagram.md) §19.1。

### 7.2 契约/加载器类

10. **timeslice 加载器默认 suffix 不含 `.t???` 占位符**。`PerambulatorTimeslicesNpy` 与三个 propagator timeslice 加载器的默认 suffix 均无法命中 `NdarrayTimeslicesFileData` 的 `.t???` → `.t%03d` 替换（`lattice/filedata/ndarray.py:70-77`），默认状态下时间片切片必报 `FileNotFoundError`；必须显式传含 `.t???` 的 suffix（PSV 版已被 `test/test_propagator_psv.py` 契约测试证实）。出处：[.cache/analysis/preset.md](../.cache/analysis/preset.md) §5.3、§9。
11. **`CurrentElementalP2P` npz 分支的动量扩展条件**。`preset.py` npz 路径以 `num_disp < num_disp * num_momentum` 判断是否复制条目（`lattice/preset.py:863` 起的 `load`）——当 npz 本身已含 `num_disp × num_momentum` 条目时会二次倍乘。出处：preset 底稿 §5.5。
12. **校验强度分层不均**。binary 后端完全信任 `FileMetaData`（无 header、无校验），`.npy` 后端信任文件头而非声明 shape（`lattice/filedata/ndarray.py`），仅 ILDG 在加载器内校验 shape/precision——声明 shape 与磁盘不符不会在加载层报错。出处：preset 底稿 §9 不变量 3。

### 7.3 测试与死代码类

13. **无 assert 的脚本式测试**。`test/test_load.py`、`test/test_elemental.py`、`test/test_eigenvector.py`、`test/test_displacement_elemental.py`、`test/test_perambulator.py`、`test/test_perambulator_mpi.py` 六个文件不含任何 `def test_` 与 `assert`——运行期不一致会静默通过（收集期 import 失败才会报错）。出处：[.cache/analysis/usage.md](../.cache/analysis/usage.md) §1.3。
14. **未接线/零调用的公开 API**（grep 全仓确认）：`re_combine`（`lattice/generator/elemental.py:2582` 附近）与 `get_overlap_matrix`（`lattice/generator/eigenvector.py:455`）；`InsertionGaugeLink`/`GaugeRepRow`/`gen_insertion_dict`（`lattice/insertion/`）；`group_projection.diagonalize_Cij`；`hardcoded_rep.gauge_link` Proxy；加载器 `CurrentElementalV2V`/`ElementalBinary`/`GaugeFieldBinary`/`GaugeFieldTimeSlice`/`Jpsi2gamma*`/`OnePointNpy`。出处：gen-core/insertion/symmetry/preset 底稿各自"死代码"小节。
15. **死代码与不可达分支**：`lattice/backend.py:22` 的 assert 使其下 `raise ValueError(…torch…)` 不可达（文案误导）；`PerambulatorGenerator` 的 `_SP` 缓冲仅赋 None；`noisevector.transfer_matrix` 死字段；`Dispatch.__init__` 的 `cfg_list` 形参未使用；`quark_diagram._collect_diagrams` 的 save_dir 磁盘缓存被 `and False` 禁用；`lattice/quark_draw.py` import 即建图并执行演示段（副作用）。
16. **`test/test_gauge_links_methods.py` 数据路径硬编码集群绝对路径**（`/public/home/...`），本地不可运行；`test/scripts/test_calc_calc_disp_consistency.py:15-17` 同——calc_deriv 与 calc_disp 的一致性在仓库外验证。出处：gen-core 底稿 Unverified §4。

---

## 附：证据来源

本文全部结论的证据底稿位于 `.cache/analysis/`（infra、preset、gen-core、gen-peram、insertion、symmetry、diagram、prop-cur、correlator、usage 共 10 份，均含 file:line 出处与"事实/推断"标注）；分模块成稿见 [docs/modules/](docs/modules/) 目录。撰写时已对当前工作区（HEAD `e19a399`）逐条核实正文引用的行号，底稿与现源码行号不一致处（如 `displacement_elemental.py` 的两项累加实际在 `:95-96`）以源码为准。
