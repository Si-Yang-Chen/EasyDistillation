# lattice/correlator/ 模块文档

本模块实现 distillation 框架下的关联函数计算层：用 elemental（插值算符元）与 perambulator（夸克传播体）通过 `opt_einsum.contract` 收缩出单粒子两点关联函数（含 isoscalar disconnected 项与多动量矩阵变体），并提供动量壳枚举、壳平均算符构造以及守恒流（conserved current）V2V 值的约定中性标量投影与显式比值工具。

## ① 模块概览与依赖关系

### 文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| `lattice/correlator/one_particle.py` | 413 | 单粒子两点关联函数的 7 个收缩变体（核心） |
| `lattice/correlator/two_particles.py` | 44 | 两粒子关联的动量壳枚举与 A/B 两侧 `Operator` 列表构造（纯辅助，不做收缩） |
| `lattice/correlator/conserved_charge.py` | 167 | 守恒流 V2V 值的显式标量投影 `project_explicit_v2v_scalar` 与显式比值 `build_declared_ratio`（纯数学工具，不构造关联函数） |
| `lattice/correlator/dispersion_relation.py` | 32 | 动量壳归一算符构造 `get_mom2_oprator` 与壳平均 2pt 入口 `twopoint_mom2` |
| `lattice/correlator/__init__.py` | 13 | 仅 re-export `conserved_charge` 的 4 个符号 |

### 模块内部依赖

| 模块 | 依赖 |
|---|---|
| `one_particle.py` | `opt_einsum.contract`、`lattice.insertion`（`Operator`、`OperatorDisplacement`、`InsertionRow`）、`lattice.insertion.gamma.gamma`、`lattice.data.get_elemental_data`、`lattice.filedata.abstract.FileData`、`lattice.backend.get_backend`（`one_particle.py:1-9`） |
| `dispersion_relation.py` | `lattice.insertion`（`Operator`、`InsertionRow`）、`FileData`、`itertools.product`、`.one_particle.twopoint`（`dispersion_relation.py:1-8`） |
| `two_particles.py` | `itertools.product`、`typing`、`lattice.insertion`（`InsertionRow`、`Operator`）（`two_particles.py:1-3`） |
| `conserved_charge.py` | 仅标准库 `hashlib`/`json`/`numbers`/`typing` 与 `numpy`，不依赖本包其它模块（`conserved_charge.py:11-16`） |
| `__init__.py` | `.conserved_charge`（`__init__.py:1-8`） |

### 被谁使用

| 使用方 | 引用符号 |
|---|---|
| `test/test_meson_spectrum.py:20` | `twopoint`、`twopoint_matrix` |
| `test/test_conserved_charge_v2v.py:10-12` | `conserved_charge` 的投影/比值工具 |
| `example/gen_twopt.py:50,75` | `twopoint`、`twopoint_matrix`、`twopoint_mom2` |
| `example/gen_twopt_matrix_mom.py:7` | `twopoint_matrix_multi_mom` |
| `example/gen_two_particle_corr_mom.py:24` | `get_AB_opratorlist_row`、`get_mom2_list` |

注意：`lattice/correlator/__init__.py` 不 re-export `one_particle`/`two_particles`/`dispersion_relation` 的任何符号（`__init__.py:1-13` 只导出 conserved_charge），因此这些函数的所有使用方都必须走子模块路径，例如 `from lattice.correlator.one_particle import twopoint`。

### 守恒流的分工边界

`conserved_charge.py` 本身不构造任何关联函数。conserved current 的 insertion 与契约构造位于 `lattice/insertion/current.py`（`ConservedVectorCurrent`、`build_current_raw_contract`）与 `lattice/current_elemental.py`（`contract_directed_current_v2v`）；本模块的 `project_explicit_v2v_scalar`/`build_declared_ratio` 只对调用方组装好的 V2V 数值做显式投影与除法，并在返回值中携带带哈希审计的可观测定义（`definition`）。两者由 `test/test_conserved_charge_v2v.py:109-149` 的端到端测试串联。

## ② 公开 API 参考

### one_particle.py —— 单粒子两点关联函数

所有变体共享一套参数语义：

| 参数 | 含义 |
|---|---|
| `operators: List[Operator]` | 插值算符列表（`lattice.insertion.Operator`），长度 Nop；`twopoint_matrix` 中同一名单既作源侧也作汇侧 |
| `elemental: FileData` | elemental 数据（懒加载切片接口，见 `lattice/filedata/abstract.py:12-19`）；磁盘布局 `(num_disp, num_mom, Lt, Ne, Ne)` |
| `perambulator: FileData` | perambulator 数据；布局 `[t_src, t_snk, 4, 4, Ne, Ne]`（默认 shape `[128,128,4,4,70,70]`，`lattice/preset.py:217-227`） |
| `timeslices: Iterable[int]` | 要计算的源时刻列表，长度 Nt |
| `Lt: int` | 输出时间轴长度 |
| `usedNe: int = None` | 使用的低模个数；`None` 时切片 `[:None]` 即使用全部 Ne（`one_particle.py:41`） |
| `perambulator_bw` | 可选 backward perambulator；非 `None` 时 backward 项改用它的共轭（`one_particle.py:44-46`） |
| `is_sum_over_source_t: bool = True` | `True` 时沿源时刻轴取均值并消去该轴；`False` 时保留全部源的原始结果 |

#### `twopoint(operators, elemental, perambulator, timeslices, Lt, usedNe=None, perambulator_bw=None, is_sum_over_source_t=True)`（`one_particle.py:12-20`）

基础变体：对每个算符做 forward 粒子传播收缩。`perambulator_bw` 非 `None` 时 backward 项 `tau_bw` 用 `perambulator_bw[t_src].conj()` 替代 `tau.conj()`（`one_particle.py:44-46`）。返回 `-(Nop, Lt)`（平均后）或 `-(Nop, Nt, Lt)`（未平均）。

#### `twopoint_profile(operators, elemental, perambulator, timeslices, delta_t_slices, usedNe=None, perambulator_bw=None, is_sum_over_source_t=True, is_diagonal=False)`（`one_particle.py:80-89`）

按 sink 时刻子集计算的 profile 变体。**没有 `Lt` 形参**，内部 `Lt = len(delta_t_slices)`（`one_particle.py:87`）。perambulator 的 sink 时间整轴切片为 `perambulator[t_src, delta_t_slices, ...]`（`one_particle.py:100`）。`is_diagonal=True` 时主收缩的 Ne 指标取对角（einsum 尾段 `->ta`），输出少一维（`one_particle.py:96-98,112-121`）。

#### `twopoint_indice(operators, elemental, perambulator, timeslices, delta_t_slices, usedNe=None, perambulator_bw=None, is_sum_over_source_t=True, is_diagonal=False)`（`one_particle.py:138-146`)

参数同 `twopoint_profile`，但主收缩保留 perambulator backward 项的两个 Ne 指标（einsum 输出标签 `->abt`，`one_particle.py:166-172`），输出 Ne 轴在前、时间轴最后。**注意：形参 `is_diagonal` 在函数体内从未被引用**（`one_particle.py:138-179` 无任何分支使用，恒按全 Ne×Ne 收缩）——这是代码现状。

#### `twopoint_matrix(operators, elemental, perambulator, timeslices, Lt, usedNe=None, is_sum_over_source_t=True)`（`one_particle.py:182-188`）

对全部 `(isrc, isnk)` 算符对求关联的内层双循环变体（`one_particle.py:202-216`）。**没有 `perambulator_bw` 形参**，backward 恒取 `tau.conj()`（`one_particle.py:200`）。每源时刻打印 perambulator 吞吐率（`one_particle.py:216`）。平均沿轴 2（Nt）。

#### `twopoint_isoscalar(operators, elemental, perambulator, timeslices, Lt, usedNe=None, Nf=2, is_sum_over_source_t=True)`（`one_particle.py:223-230`）

在 connected 之外用 `tau[0]`（sink 时间轴取第 0 片）额外收缩源/汇两处 quark loop（`one_particle.py:261-262`），外积成 disconnected 项后逐源时刻 roll 对齐（`one_particle.py:265-269`），最终返回 `-connected + Nf*disconnected`（`one_particle.py:270`）。`Nf` 是 disconnected 项的味数权重。**前置校验：`Lt != Nt` 时抛 `ValueError("Disconnect must compute full timeslices!")`**（`one_particle.py:236-237`），即 disconnected 项要求覆盖所有源时刻。

#### `twopoint_isoscalar_matrix(operators, elemental, perambulator, timeslices, Lt, usedNe=None, Nf=2, is_sum_over_source_t=True)`（`one_particle.py:275-282`）

`twopoint_isoscalar` 的矩阵版：connected 对全部 `(isrc, isnk)` 对计算，disconnected 外积为 `"xi,yj->xyij"`（`one_particle.py:317-321`），同样有 `Lt != Nt` 校验（`one_particle.py:288-289`）。

#### `twopoint_matrix_multi_mom(insertions, mom_list, elemental, perambulator, timeslices, Lt, usedNe=None, insertions_coeff_list=None, is_sum_over_source_t=True, distance_list=None)`（`one_particle.py:326-336`）

多动量矩阵变体。参数差异：

| 参数 | 含义 |
|---|---|
| `insertions: List[InsertionRow]` | 插值行列表（而非 `Operator`），逐动量逐行即时构造算符（`one_particle.py:338-379`） |
| `mom_list: List` | `[(px,py,pz), ...]`，长度 Nmom |
| `insertions_coeff_list: List = None` | 各行系数，`None` 时全 1；`assert len(insertions) == len(insertions_coeff_list)`（`one_particle.py:345-346`） |
| `distance_list: List = None` | 非 `None` 时改用 `OperatorDisplacement`（位移通过把 `derivative_idx` 设为位移距离实现，要求 `derivative_idx == 0`，`lattice/insertion/__init__.py:202-218`），并 `assert len(distance_list) == len(insertions_coeff_list)`（`one_particle.py:347-348`） |

先把 `(imom, isrc, isnk)` 三重循环展开为 `op_src_list`/`op_snk_list`（共 `Nterm = Nmom*Nop*Nop` 项，`one_particle.py:381`），对两个列表分别调 `get_elemental_data` 后做与 `twopoint` 相同的主收缩，最后 `ret.reshape((Nmom, Nop, Nop, Nt, Lt))`（`one_particle.py:409`）。

### two_particles.py —— 两粒子关联辅助

| 函数 | 签名与行为 |
|---|---|
| `get_mom2_list(mom2: int) -> List[Tuple]`（`two_particles.py:7-15`） | 枚举 `px,py,pz ∈ [-3,3]³` 中满足 `px²+py²+pz² == mom2` 的全部动量元组；返回长度为壳内动量数的 list（如 `mom2=1` → 6 个） |
| `get_AB_opratorlist_row(insertion_row_A: InsertionRow, insertion_row_B: InsertionRow, mom_list: List[Tuple])`（`two_particles.py:17-26`） | 对每个 `p`：A 侧 `insertion_row_A(px,py,pz)`，B 侧 `insertion_row_B(-px,-py,-pz)`（共轭动量），各包成单行 `Operator("", [...], [1])`；返回 `(ops_A, ops_B)` 二元组 |
| `get_AB_opratorlist_rows(insertions_row_A: List[InsertionRow], insertions_row_B: List[InsertionRow], mom_list: List[Tuple], coeff: List = None)`（`two_particles.py:29-44`） | 多行版本；`coeff` 为 `None` 时全 1；`assert len(insertions_row_A) == len(insertions_row_B)`（`two_particles.py:30`）；B 侧同样取 `-p`，A、B 共用同一 `coeff`（`two_particles.py:39-41`） |

`InsertionRow.__call__(npx, npy, npz)` 把动量字符串映射为 `momentum_dict` 的整数索引：`list(self.momentum_dict.values()).index(f"{npx} {npy} {npz}")`（`lattice/insertion/__init__.py:227-228`）——因此 `px,py,pz` 必须存在于构造 `Insertion` 时的 `momentum_dict` 中，否则抛 `ValueError`。

### conserved_charge.py —— 约定中性数值工具

#### `project_explicit_v2v_scalar(value, dual_weight, *, definition)`（`conserved_charge.py:69-97`）

| 参数 | 约束 |
|---|---|
| `value` | 复数 array，4 维，前两轴必须是 `(4, 4)`，语义轴 `(sink_spin, source_spin, sink_ne, source_ne)`（`conserved_charge.py:82-83`） |
| `dual_weight` | 与 `value` 形状完全相同的复数 array（`conserved_charge.py:84-85`） |
| `definition` | 仅关键字参数；mapping，字段必须恰好为 `{schema, id, formula, approval_sha256}`，`schema` 必须为 `"lattice.current.observable-definition/v1"`，`approval_sha256` 必须是 64 位 hex（校验函数 `_definition`，`conserved_charge.py:46-67`） |

运算严格为 `np.einsum("abij,abij->", weight, value)`，**不共轭、不转置、不归一化**（`conserved_charge.py:87-88`；返回字符串 `operation` 亦如此声明）。返回 dict，含 `schema`（`SCALAR_PROJECTION_SCHEMA = "lattice.current.explicit-v2v-scalar-projection/v1"`，`conserved_charge.py:18`）、`value`（0 维标量）、`input_axes`、`operation`、`definition`（规范化后）、`definition_identity = sha256(canonical JSON)`（`conserved_charge.py:62-63,90-95`）。

#### `build_declared_ratio(numerator, denominator, *, definition, denominator_axis=None, zero_tolerance=0.0)`（`conserved_charge.py:100-160`）

| 参数 | 约束 |
|---|---|
| `numerator` / `denominator` | 实或复数值 array（`_real_or_complex_array`，`conserved_charge.py:29-39`） |
| `denominator_axis` | `None` 时要求分子分母形状完全一致（`conserved_charge.py:117-120`）；给整数 axis 时，denominator 形状必须等于分子去掉该轴后的形状，分母沿该轴 `np.expand_dims` 对齐（`conserved_charge.py:121-135`）；必须是 `Integral`（bool 被显式拒绝） |
| `zero_tolerance` | 非负有限实标量（`conserved_charge.py:107-116`）；任何 `|denominator| <= tolerance` 处抛 `ZeroDivisionError`（`conserved_charge.py:136-138`） |
| `definition` | 同上 |

返回 dict 含 `schema`（`DECLARED_RATIO_SCHEMA = "lattice.current.declared-ratio/v1"`，`conserved_charge.py:19`）、`value`（与 numerator 同形状）、`operation`、`denominator_axis`、`zero_tolerance`、`definition`、`definition_identity`（`conserved_charge.py:139-160`）。除法结果含非有限值时抛 `ValueError`。

### dispersion_relation.py —— 色散关系（dispersion relation）数据点

#### `get_mom2_oprator(insertion_row: InsertionRow, mom2: int) -> Operator`（`dispersion_relation.py:10-17`）

枚举 `px,py,pz ∈ product(range(-3,4), repeat=3)` 中满足 `px²+py²+pz² == mom2` 的动量，全部塞进**一个** `Operator(f"mom{mom2}", ret, [len(ret)**-0.5] * len(ret))`——壳内每个动量分量带系数 `1/sqrt(N_mom)`，实现壳平均；每命中一个动量打印 `add mom: {i}`（副作用）。

#### `twopoint_mom2(insertion_row: InsertionRow, mom2: int, elemental: FileData, perambulator: FileData, timeslices: Iterable[int], Lt: int, usedNe=None, func=twopoint, **kwargs)`（`dispersion_relation.py:20-32`）

流程：`operators = get_mom2_oprator(insertion_row, mom2)`，然后 `return func([operators], elemental, perambulator, timeslices, Lt, usedNe, **kwargs)`（`dispersion_relation.py:31-32`）。`func` 默认绑定为模块顶部导入的 `one_particle.twopoint`（`dispersion_relation.py:8`）；`**kwargs` 原样透传（如 `perambulator_bw`、`is_sum_over_source_t`）。默认路径下（Nop=1、`is_sum_over_source_t=True`）返回 `(Lt,)` 的壳平均 2pt；若传 `func=twopoint_matrix` 等其它变体，返回形状随 `func` 变化。

### `__init__.py`

无自有逻辑，仅 re-export `conserved_charge` 的 4 个符号并定义对应的 `__all__`（`__init__.py:1-13`）：`SCALAR_PROJECTION_SCHEMA`、`DECLARED_RATIO_SCHEMA`、`project_explicit_v2v_scalar`、`build_declared_ratio`。

## ③ 数据结构与返回数组形状约定

### 输入磁盘布局（收缩形状的锚点）

| 数据 | 布局 | 出处 |
|---|---|---|
| elemental | `(num_disp, num_mom, Lt, Ne, Ne)`；`get_elemental_data` 按 `elemental[derivative_idx, momentum_idx, :, :usedNe, :usedNe]` 切片 | `lattice/data.py:29`、`lattice/preset.py:717-729`（`ElementalNpy` 默认 shape `[4, 123, 128, 70, 70]`） |
| perambulator | `[t_src, t_snk, 4, 4, Ne, Ne]`；按 `perambulator[t_src, :, :, :, :usedNe, :usedNe]` 取源时刻整块，形状 `(Lt_snk, 4, 4, Ne, Ne)` | `one_particle.py:41`、`lattice/preset.py:217-227` |
| `Operator.parts` | `[gamma_idx, [[coeff, derivative_idx, momentum_idx, profile], ...], gamma_idx, ...]` 交替；`get_elemental_data` 对每个 `(derivative_idx, momentum_idx)` 建缓存，返回每算符的 `(ret_gamma, ret_elemental)`，其中 `ret_elemental[k] = Σ_j coeff_j · elemental[deriv_j, mom_j, :, :usedNe, :usedNe]` | `lattice/data.py:7-36`（切片与累加见 `:29,31,33`） |

`get_elemental_data` 返回的每算符 `phi` 二元组中：`phi[0]` 形状 `(行数, 4, 4)`（各行 gamma 矩阵），`phi[1]` 形状 `(行数, Lt, usedNe, usedNe)`（系数线性组合后的 elemental，`lattice/data.py:34`）。

### 各变体返回形状（返回值整体带负号）

| 函数 | `is_sum_over_source_t=False` | `is_sum_over_source_t=True` | 轴序说明 |
|---|---|---|---|
| `twopoint` | `(Nop, Nt, Lt)`（`one_particle.py:51,77`） | `(Nop, Lt)`（`one_particle.py:75`） | 轴 0=算符，轴 1=源时刻，轴 2=输出时间（相对源，见 ④） |
| `twopoint_profile` | `is_diagonal=True` → `(Nop, Nt, Lt, usedNe)`；否则 `(Nop, Nt, Lt, usedNe, usedNe)`（`one_particle.py:96-98`） | 同形但去掉 Nt 轴 | 最后轴（对角时）或最后两轴为 perambulator backward 项的 Ne 指标 |
| `twopoint_indice` | `(Nop, Nt, usedNe, usedNe, Lt)`（`one_particle.py:153`） | `(Nop, usedNe, usedNe, Lt)` | Ne 轴在前、时间轴最后；输出 `a`/`b` 见下 |
| `twopoint_matrix` | `(Nop, Nop, Nt, Lt)`（`one_particle.py:195`） | `(Nop, Nop, Lt)`（`one_particle.py:218-220`，平均沿轴 2） | 轴 0=isrc（源侧算符，提供 `phi_src[1][:, t_src].conj()`），轴 1=isnk（汇侧算符） |
| `twopoint_isoscalar` | `(Nop, Nt, Lt)` | `(Nop, Lt)` | `-connected + Nf*disconnected`（`one_particle.py:270`） |
| `twopoint_isoscalar_matrix` | `(Nop, Nop, Nt, Lt)` | `(Nop, Nop, Lt)` | 同上（`one_particle.py:322-325`） |
| `twopoint_matrix_multi_mom` | `(Nmom, Nop, Nop, Nt, Lt)`（`one_particle.py:409`） | `(Nmom, Nop, Nop, Lt)`（`one_particle.py:411-413`） | 动量 → 源算符 → 汇算符 → 时间 |
| `twopoint_mom2`（默认 `func=twopoint`） | `(Nop, Nt, Lt)` | `(Lt,)` | 单算符壳平均 |

`twopoint_indice` 输出 Ne 轴的语义可由 einsum 标签追溯确定：主收缩 `"tijab,xjk,xtbc,tklcd,yli,yad->..."` 中，backward 项 `tau_bw` 的指标 `a` 与源侧 elemental（`yad` 项，即 `phi[1][:, t_src].conj()`）配对，指标 `b` 与汇侧 elemental（`xtbc` 项，即 roll 后的 `phi[1]`）配对（`one_particle.py:64-73`）；`->abt` 输出沿用了同一套标签，故 `a` = 与源侧 elemental 配对的 Ne 指标，`b` = 与汇侧 elemental 配对的 Ne 指标，`t` = perambulator sink 时间轴。

`twopoint_profile`/`twopoint_indice` 未使用任何 `Lt` 形参，其输出时间长度由 `len(delta_t_slices)` 决定（`one_particle.py:87-89,141-142`）。

`conserved_charge` 返回形状：`project_explicit_v2v_scalar["value"]` 为 0 维标量；`build_declared_ratio["value"]` 与 numerator 同形状。两个函数均不修改输入数组。

## ④ 算法与流程

### 单粒子通路（one_particle / dispersion_relation 共享骨架）

以 `twopoint` 为例（`one_particle.py:38-77`）：

1. **elemental 展开**：`phis = get_elemental_data(operators, elemental, usedNe)`——对每个 `Operator` 的 `parts` 按 `[gamma 索引, [coeff, derivative_idx, momentum_idx, profile] 交替]` 展开，gamma 索引经 `gamma(n)` 变成 4×4 矩阵；elemental 按 `(derivative_idx, momentum_idx)` 缓存切片并做系数线性组合（`lattice/data.py:7-36`）。
2. **取 perambulator**：逐源时刻 `t_src`，`tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]`，形状 `(Lt_snk, 4, 4, Ne, Ne)`（`one_particle.py:41`）。
3. **构造 backward 项**：`tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))`；`gamma(15)` 即 γ5（`gamma(n)` 按 n 的 4 个 bit 连乘生成元矩阵，`lattice/insertion/gamma.py:95-105`；`output(15) == "γ5"`，`lattice/insertion/gamma.py:79-81`）。有 `perambulator_bw` 时改用 `perambulator_bw[t_src].conj()`（`one_particle.py:44-46`）。
4. **源侧 γ 结构**：`gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi[0].conj(), gamma(8))`（`one_particle.py:63`）；`gamma(8)` 即 γ4（`output(8) == "γ4"`，`lattice/insertion/gamma.py:82-83`）。
5. **主收缩**（forward 粒子传播）：

   ```python
   ret[idx, it] = contract("tijab,xjk,xtbc,tklcd,yli,yad->t",
       tau_bw,                              # backward perambulator
       phi[0],                              # 汇侧 gamma 结构 (行, 4, 4)
       backend.roll(phi[1], -t_src, 1),     # 汇侧 elemental, 沿 sink 时间轴 roll
       tau,                                 # forward perambulator
       gamma_src,                           # 源侧 gamma 结构
       phi[1][:, t_src].conj(),             # 源侧 elemental（源时刻切片, 共轭）
   )
   ```

   标签映射：`t` = perambulator 的 sink 时间轴；`i,j,k,l` = spin 轴；`a,b,c,d` = Ne 轴；`x` = 汇侧算符行索引；`y` = 源侧 gamma 结构行索引。
6. **时间对齐**：`backend.roll(phi[1], -t_src, 1)`。按 `np.roll` 语义 `result[i] = a[(i - shift) % n]`，`shift = -t_src` 时输出第 `t` 个元素取自 elemental 的 sink 时刻 `t + t_src`，因此输出时间轴为相对源时刻 `t = t_sink − t_src`；这与 `is_sum_over_source_t=True` 沿源轴求平均以消去源轴的用法一致。
7. **收尾**：返回值整体乘 `-1`；`is_sum_over_source_t=True` 时沿源时刻轴取 `mean`（`one_particle.py:74-77`）。

#### 变体差异

| 变体 | 与骨架的差异 |
|---|---|
| `twopoint_matrix` | 内层 `(isrc, isnk)` 双循环；源侧固定 `phi_src[1][:, t_src].conj()`，汇侧用 `phi_snk`（`one_particle.py:202-216`）；平均沿轴 2 |
| `twopoint_matrix_multi_mom` | 先按 `(imom, isrc, isnk)` 把 `insertions[i](px,py,pz)` 打包为 `Operator`（或 `OperatorDisplacement`）展开到 `op_src_list`/`op_snk_list`（`one_particle.py:338-379`），对两个列表分别调 `get_elemental_data`（`one_particle.py:385-386`），最后 reshape 为 `(Nmom, Nop, Nop, Nt, Lt)`（`one_particle.py:409`） |
| `twopoint_isoscalar(_matrix)` | 额外用 `tau[0]` 收缩 `loop_src = contract("ijab,yji,yab", tau[0], gamma_src, phi[1][:, t_src].conj())` 与 `loop_snk = contract("ijab,xji,xba", tau[0], phi[0], phi[1][:, t_src])`（`one_particle.py:261-262,302-303`）；`disconnected = contract("xi,xj->xij", ...)`（matrix 版 `"xi,yj->xyij"`），逐源时刻沿时间轴 roll `-t_src` 对齐（`one_particle.py:265-269,317-321`）；返回 `-connected + Nf*disconnected`（`one_particle.py:270,322-325`） |
| `twopoint_profile` / `twopoint_indice` | perambulator 的 sink 时间整轴改为 `perambulator[t_src, delta_t_slices, ...]`（`one_particle.py:100,143`），elemental 汇侧同步取 `[:, delta_t_slices]`（`one_particle.py:123,129,169`） |

### 两粒子辅助通路（two_particles.py）

纯枚举与符号翻转，无数值收缩：`get_mom2_list` 枚举动量壳 → `get_AB_opratorlist_row(s)` 为每个动量构造 A 侧 `+p`、B 侧 `-p` 的单行/多行 `Operator`，随后由使用方（如 `example/gen_two_particle_corr_mom.py:135-160`）交给 `Meson`/`compute_diagrams_multitime` 管线。

### 守恒流通路（conserved_charge.py）

纯校验 → 纯运算 → 返回带审计元数据的 dict：`ConservedVectorCurrent`（`lattice/insertion/current.py`）与 `contract_directed_current_v2v`（`lattice/current_elemental.py`）在调用方组装出 endpoint-aware 的 V2V 复数值 → `project_explicit_v2v_scalar` 用调用方提供的对偶张量 `dual_weight` 做 `sum(dual_weight * value)`（无隐藏 γ 结构/共轭）→ 需要比值时用 `build_declared_ratio`（无隐藏拟合或归一化）。与单粒子通路的本质区别：一切约定外置为带 SHA-256 身份（`definition_identity`）的 `definition` 审计字段。

### 色散关系通路（dispersion_relation.py）

壳枚举（`range(-3,4)`）→ 系数归一 `1/sqrt(N_mom)` → 单算符委托 `twopoint`。

## ⑤ 不变量与校验

### 由实现保证的硬性质

| 不变量 | 出处 |
|---|---|
| 所有 one_particle 变体的返回值整体带负号（`return -ret` 族） | `one_particle.py:75,77,131,179,220,270,325,411,413` |
| `usedNe=None` ⇒ 切片 `[:None]` 使用全部 Ne | `one_particle.py:41` |
| `is_sum_over_source_t=True` 沿源时刻轴取均值并消去该轴 | `one_particle.py:74-75,218-220,411-413` |
| `twopoint_isoscalar(_matrix)` 强制 `Lt == Nt`，否则 `ValueError("Disconnect must compute full timeslices!")` | `one_particle.py:236-237,288-289` |
| `conserved_charge` 不做隐藏共轭/转置/归一化（einsum 无 `.conj()`，`operation` 字段显式声明） | `conserved_charge.py:87-89` |
| `conserved_charge` 所有输入要求 finite（`np.isfinite` 全检）、复数 dtype 强制（投影） | `conserved_charge.py:24,26,38` |
| `definition` 的 identity 对字段规范化（`approval_sha256` 小写化）后做 SHA-256，同义定义得到同一 identity | `conserved_charge.py:56-63` |
| `Operator` 构造要求 rows 数 == 系数数 | `lattice/insertion/__init__.py:153-156` |
| `OperatorDisplacement` 要求 `derivative_idx == 0` 且 rows 数 == distances 数（位移实现为覆写 `derivative_idx`） | `lattice/insertion/__init__.py:207-218` |

### 边界条件（代码现状，含未防护点）

| 边界 | 现状 |
|---|---|
| `twopoint_profile`/`twopoint_indice` 的 `perambulator_bw` 分支中，`tmp = perambulator_bw[t_src, :, :, :, :usedNe, :usedNe]` **未**取 `delta_t_slices` 切片（`one_particle.py:107,162`），与 forward 侧 `perambulator[t_src, delta_t_slices, ...]`（`one_particle.py:100,143`）不对称：当 `delta_t_slices` 为真子集时，forward 的 `tau` 与 backward 的 `tau_bw` 时间轴长度不一致，主收缩会因 einsum 尺寸不匹配报错。这是代码现状，无显式防护或断言。 | 代码现状 |
| `twopoint_indice` 的形参 `is_diagonal` 从未被使用，传入任何值行为相同 | 代码现状（`one_particle.py:138-179`） |
| 主收缩要求 `usedNe ≤ Ne` 且 elemental/perambulator 时间轴长度与 `Lt` 一致；无显式断言，越界由切片/收缩报错 | 代码现状 |
| `two_particles.get_mom2_list` 枚举范围硬编码 `range(-3,4)`；`mom2 > 27` 时返回空表（不报错） | `two_particles.py:9` |
| `dispersion_relation.get_mom2_oprator` 对空动量壳无防护：`mom2` 无满足动量时 `Operator` 的 rows/coeffs 为空列表，构造断言仍通过（`len([])==len([])`），但后续 `get_elemental_data` 会产出空 gamma/elemental 并在收缩处出错 | 代码现状 |
| `dispersion_relation.py:12` 与 `two_particles.py:9` 对同一动量壳枚举做了**重复实现**，两处范围需人工同步 | 代码现状 |
| `twopoint_matrix`/`twopoint_isoscalar(_matrix)`/`twopoint_matrix_multi_mom` 每源时刻打印 perambulator 吞吐率（副作用） | `one_particle.py:216,258,311,399` |
| `conserved_charge` 的 `(sink_spin, source_spin, sink_ne, source_ne)` 轴约定与 one_particle 收缩中 perambulator `(t,i,j,a,b)` 轴序之间的衔接代码不在 `lattice/correlator/` 内（位于 current.py / current_elemental.py 一侧），本模块不做校验 | 代码现状 |
| `conserved_charge` 文件 docstring 声明"不定义 hadron 算符/spin projector/味权重/电荷归一化/Ward–Takahashi identity，调用方必须自备完整对偶张量与已批准公式"（`conserved_charge.py:1-7`）；与代码一致：文件内确无这些物理量 | 代码现状（存在性） |

## ⑥ 相关测试

| 测试文件 | 覆盖内容 |
|---|---|
| `test/test_meson_spectrum.py`（`pytestmark = pytest.mark.gpu`，`set_backend("cupy")`，`test_meson_spectrum.py:6-8`） | 用 `preset.ElementalNpy(shape=[13,6,Lt,Ne,Ne], Ne=20)` + `preset.PerambulatorNpy(shape=[Lt,Lt,4,4,Ne,Ne])`，数据 `test/weak_field.*.npy`（`test_meson_spectrum.py:30-42`）；格点 `4,4,4,8`（`:19`）。`test_twopoint_single`/`test_twopoint_multi_op` 断言输出 shape `(Nop, Lt)`（`:47-53,68-74`）；`test_twopoint_matrix` 断言 `(2, 2, Lt)`（`:87-95`）。**仅断言形状与 `.real` 可取，不断言数值** |
| `test/test_conserved_charge_v2v.py` | 无隐藏共轭 + 输入不变性（`test_conserved_charge_v2v.py:27-44`）；形状/复 dtype/`definition` 字段错误路径（`:45-56`）；`build_declared_ratio` 的 `denominator_axis` 对齐、同形、零除、shape mismatch（`:57-108`）；端到端串联：用 `ConservedVectorCurrent().terms[6:8]` + `build_current_raw_contract` + `contract_directed_current_v2v` 组装 endpoint-aware 的 temporal current V2V 值，再以单位对偶张量投影，断言等于取单分量，并断言 accessor（incoming/outgoing）调用次序（`:109-149`） |

**无直接测试覆盖的符号**（grep `test/` 确认，correlator 导入仅上述两处）：`twopoint_profile`、`twopoint_indice`、`twopoint_isoscalar(_matrix)`、`twopoint_matrix_multi_mom`、`two_particles.py` 全部、`dispersion_relation.py` 全部——这些只有 example 使用（`example/gen_twopt.py:75,77`、`example/gen_twopt_matrix_mom.py:7,72`、`example/gen_two_particle_corr_mom.py:24,135,138`）。`lattice/correlator/__init__.py` 亦无独立测试（`test_conserved_charge_v2v.py` 走 `from lattice.correlator.conserved_charge import ...` 的子模块直连路径）。

最小验证命令（复现主收缩与形状断言）：

```bash
python -m pytest test/test_meson_spectrum.py -m gpu -q
```
