# 端到端工作流（example/ 典型流程指南）

> 本文从 `example/` 的 9 个脚本提炼出 EasyDistillation 的典型端到端工作流——从数据加载、算符构造、关联函数计算到 quark 图绘制与 perambulator 生成，每条工作流给出步骤分解、对应示例脚本、涉及的 API 与前置数据/环境要求。

---

## ① 模块概览（含依赖关系）

example/ 的 9 个脚本共同串联了以下模块，依赖方向自上而下：

| 模块 | 在工作流中的角色 | 被谁依赖 |
|---|---|---|
| `lattice.backend` | 统一 numpy/cupy 后端选择（`set_backend`/`get_backend`），PyQuda 探测（`check_QUDA`） | 所有模块 |
| `lattice.preset` | 各类磁盘数据（gauge、eigenvector、elemental、perambulator）的惰性加载器 | correlator、quark_diagram、generator |
| `lattice.insertion` | interpolating operator 构造（`Insertion`/`Operator`、`GammaName`/`DerivativeName`/`ProjectionName`、`momDict_mom9`） | correlator、quark_diagram |
| `lattice.correlator.one_particle` | 单粒子 2pt：`twopoint`、`twopoint_matrix`、`twopoint_matrix_multi_mom` | example 脚本直接调用 |
| `lattice.correlator.dispersion_relation` | 动量壳求和 2pt：`twopoint_mom2` | example 脚本直接调用 |
| `lattice.correlator.two_particles` | 两粒子算符动量枚举：`get_mom2_list`、`get_AB_opratorlist_row` | example 脚本直接调用 |
| `lattice.propagators` | quark 图的"线"与"顶点"对象：`Meson`、`Propagator`、`PropagatorLocal` | quark_diagram |
| `lattice.quark_diagram` | quark 线收缩：`QuarkDiagram`、`compute_diagrams(_multitime)` | example 脚本直接调用 |
| `lattice.quark_draw` | quark 图绘制：`draw_single_diagram`、`draw_multi_diagrams` | example 脚本直接调用 |
| `lattice.symmetry.two_particle` | 两粒子 Cartesian 算符基：`two_particle_Cartesian_basis` | example 脚本直接调用 |
| `lattice.generator.density_perambulator` | density perambulator 生成（PyQuda/cupy 硬依赖）：`DensityPerambulatorGenerator` | example 脚本直接调用 |
| `lattice.dispatch` | MPI 环境下的 cfg 分发：`Dispatch` | example 脚本直接调用 |

概括而言，全部数值工作流汇聚于两个符号：`lattice/quark_diagram.py:1646` 的 `compute_diagrams_multitime`（quark 图收缩）与 `lattice/propagators.py:188` 的 `Meson`（elemental × interpolating operator 组成的顶点）。

---

## ② 公开 API 参考

仅列出 example/ 工作流实际用到的公开符号；签名均逐字符核对过源码。

### 2.1 后端与运行环境

| 符号 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `set_backend` | `set_backend(backend: Literal["numpy", "cupy"])`（`lattice/backend.py:17`） | None | 选定全库的 ndarray 实现 |
| `get_backend` | `get_backend()`（`lattice/backend.py:10`） | 后端模块 | 后续可直接 `backend.zeros/roll/arccosh` 等 |
| `check_QUDA` | `check_QUDA(grid_size=None, backend="cupy", resource_path=None)`（`lattice/backend.py:39`） | bool | 尝试 `pyquda.init` 并要求版本 ≥0.9.0，失败返回 False |

### 2.2 cfg 分发

| 符号 | 签名 | 说明 |
|---|---|---|
| `Dispatch` | `Dispatch(filename: str, suffix: str = None, cfg_list: Iterable = None)`（`lattice/dispatch.py:62`） | 构造时以 `AtomicOpen` 创建原子锁文件 `f"{filename}.{rand(suffix)}.tmp"` 并读入 cfglist（`lattice/dispatch.py:64-74`）；`__iter__` 在 MPI 各 rank 间分发 cfg 行，可直接 `for cfg in dispatcher` 使用 |

### 2.3 preset 数据加载器（`lattice/preset.py`）

加载器均为惰性对象：构造时只登记 `prefix`/`suffix`/磁盘 `shape`/`dtype`，`load(key)` 才拼接 `f"{prefix}{key}{suffix}"` 读文件。

| 类 | 构造签名 | 出处 | 磁盘 dtype | 轴序（shape 含义） |
|---|---|---|---|---|
| `GaugeFieldIldg` | `(prefix, suffix, shape=[128, 16**3, 4, 3, 3])` | `lattice/preset.py:687` | `>c16` | Lt, Vol, Nd, Nc, Nc（ILDG/LIME gauge） |
| `EigenvectorTimeSlice` | `(prefix, suffix, shape=[128, 70, 16**3, 3], totNe=70)` | `lattice/preset.py:164` | `>c8` | Lt, Ne, Vol, Nc（逐时间片 LapEig） |
| `PerambulatorBinary` | `(prefix, suffix, shape=[128, 128, 4, 4, 70, 70], totNe=70)` | `lattice/preset.py:200` | `<c16` | Lt, Lt, Ns, Ns, Ne, Ne（二进制 perambulator） |
| `PerambulatorNpy` | `(prefix, suffix, shape=[128, 128, 4, 4, 70, 70], totNe=70)` | `lattice/preset.py:217` | `<c8` | 同上（.npy perambulator） |
| `ElementalNpy` | `(prefix, suffix, shape=[4, 123, 128, 70, 70], totNe=70)` | `lattice/preset.py:717` | `<c8` | num_disp, num_mom, Lt, Ne, Ne（legacy meson elemental，docstring 明确该轴序） |

### 2.4 interpolating operator 构造（`lattice/insertion`）

| 符号 | 签名 | 出处 | 说明 |
|---|---|---|---|
| `Insertion` | `(gamma: GammaName, derivative: DerivativeName, projection: ProjectionName, momentum_dict: Dict[int, str], profile=None)` | `lattice/insertion/__init__.py:249` | 按量子数构造一整组 insertion 行；`insertion[idx]` 返回第 idx 行（`lattice/insertion/__init__.py:270`） |
| `InsertionRow.__call__` | `(npx, npy, npz) -> InsertionRowMom` | `lattice/insertion/__init__.py:227` | 固定到具体动量 `(npx, npy, npz)`；动量必须在 `momentum_dict` 的 values 中（内部 `list(...).index(...)`，不存在则 ValueError） |
| `Operator` | `(name: str, insertion_rows: List[InsertionRowMom], coefficients: List[float])` | `lattice/insertion/__init__.py:147` | 多行线性组合即一个 interpolating operator；断言行数与系数个数一致 |
| `GammaName` | 枚举类（含 `PI`、`PI_2`、`RHO`、`B1`、`A1` 等） | `lattice/insertion/gamma.py:198` | γ 结构选择 |
| `DerivativeName` | 枚举类（含 `IDEN`、`NABLA` 等） | `lattice/insertion/derivative.py:127` | 导数结构选择 |
| `ProjectionName` | 枚举类（`A1`、`T1` 等） | `lattice/insertion/__init__.py:29` | 立方群 irrep 选择 |
| `momDict_mom9` | `Dict[int, str]` | `lattice/insertion/mom_dict.py:35` | |p|² ≤ 9 的 123 个三维动量字典，value 形如 `"0 0 -3"` |

### 2.5 quark 图对象与收缩（`lattice/propagators.py`、`lattice/quark_diagram.py`）

| 符号 | 签名 | 出处 | 说明 |
|---|---|---|---|
| `Meson` | `Meson(elemental, operator, source: bool)` | `lattice/propagators.py:188` | 介子顶点；`source=True` 为源端，`False` 为汇端；`load(key, usedNe=None)`（`lattice/propagators.py:239`） |
| `Propagator` | `Propagator(perambulator, Lt)` | `lattice/propagators.py:742` | 带非局域 quark 线；`load(key, usedNe=None)`（`lattice/propagators.py:783`） |
| `PropagatorLocal` | `PropagatorLocal(perambulator, Lt)` | `lattice/propagators.py:863` | 局域 quark 线；`load(key, usedNe=None)`（`lattice/propagators.py:886`） |
| `QuarkDiagram` | `QuarkDiagram(adjacency_matrix, vertex_list=None, L=None, usedNp=None, debug=False, validate=True)` | `lattice/quark_diagram.py:507` | 邻接矩阵的非零值即 `propagator_list` 下标；`validate=True` 时校验 quark-link 不变量（`lattice/quark_diagram.py:550-551`） |
| `compute_diagrams_multitime` | `(diagrams: List[QuarkDiagram], time_list, vertex_list: List[Meson], propagator_list: List[Propagator], multitime_shape=False, debug=False)` | `lattice/quark_diagram.py:1646` | 批量计算图值；`time_list` 中 `t_snk` 必须是 numpy 数组（见 ⑤） |
| `compute_diagrams` | 同上族的非 multitime 版本 | `lattice/quark_diagram.py:1938` | 源/汇时间均取单点 |

### 2.6 关联函数（`lattice/correlator`）

| 符号 | 签名 | 出处 | 返回 |
|---|---|---|---|
| `twopoint` | `(operators: List[Operator], elemental: FileData, perambulator: FileData, timeslices: Iterable[int], Lt: int, usedNe: int = None, perambulator_bw=None, is_sum_over_source_t=True)` | `lattice/correlator/one_particle.py:12` | `is_sum_over_source_t=True` 时对源时间平均，返回 `(Nop, Lt)`；否则 `(Nop, Nt, Lt)`（`lattice/correlator/one_particle.py:79-82`） |
| `twopoint_matrix` | `(operators, elemental, perambulator, timeslices, Lt, usedNe=None, is_sum_over_source_t=True)` | `lattice/correlator/one_particle.py:182` | 平均后 `(Nop, Nop, Lt)`（`lattice/correlator/one_particle.py:218-222`） |
| `twopoint_matrix_multi_mom` | `(insertions: List[InsertionRow], mom_list: List, elemental, perambulator, timeslices, Lt, usedNe=None, insertions_coeff_list=None, is_sum_over_source_t=True, distance_list=None)` | `lattice/correlator/one_particle.py:326` | `(Nmom·Nop·Nop, Nt, Lt)`，内部 `ret = backend.zeros((Nterm, Nt, Lt))`、`Nterm = Nmom * Nop * Nop`（`lattice/correlator/one_particle.py:379-380`） |
| `twopoint_mom2` | `(insertion_row: InsertionRow, mom2: int, elemental, perambulator, timeslices, Lt, usedNe=None, func=twopoint, **kwargs)` | `lattice/correlator/dispersion_relation.py:20` | 对 |p|²==mom2 的所有动量求和后委托 `func`（默认 `twopoint`）；见 ⑤ 的注释不一致提醒 |
| `get_mom2_list` | `(mom2: int) -> List[Tuple]` | `lattice/correlator/two_particles.py:7` | 枚举 |p|²==mom2 的全部三维动量（如 mom2=1 给出 6 个） |
| `get_AB_opratorlist_row` | `(insertion_row_A: InsertionRow, insertion_row_B: InsertionRow, mom_list: List[Tuple])` | `lattice/correlator/two_particles.py:17` | `(ops_A, ops_B)`：A 侧固定 `insertion_row_A(p)`，B 侧取 `insertion_row_B(-p)`（`lattice/correlator/two_particles.py:23-25`） |

### 2.7 两粒子算符基与绘图

| 符号 | 签名 | 出处 | 说明 |
|---|---|---|---|
| `two_particle_Cartesian_basis` | `(op1, op2, mom2, J, L, Spin)` | `lattice/symmetry/two_particle.py:108` | 返回 O(3)→O_h 立方群的 Cartesian 算符基（sympy 表达式列表）；`J>1` 抛 `NotImplementedError`（`lattice/symmetry/two_particle.py:121-122`）；`op1`/`op2` 支持 `"V"`/`"P"`/`"V1"`/`"V2"` |
| `draw_single_diagram` | `(adjacency_matrix, vertex_attribute_list, line_color_list, save_path=None, quark_types=None)` | `lattice/quark_draw.py:320` | 画单张 quark 图；顶点属性 dict 含 `pos`（src/snk）、`type`、`name` |
| `draw_multi_diagrams` | `(adjacency_matrix_list, vertex_attribute_list, line_color_list, save_path=None)` | `lattice/quark_draw.py:299` | 批量画多张图；`save_path` 为与图等长的列表 |
| `DensityPerambulatorGenerator.__init__` | `(latt_size, gauge_field, eigenvector, mass, tol, maxiter, xi_0=1.0, nu=1.0, clover_coeff_t=0.0, clover_coeff_r=1.0, t_boundary=1, multigrid=None, gamma_list=[0..15], momentum_list=[(0,0,0)])` | `lattice/generator/density_perambulator.py:13-33` | 构造时断言 `backend.__name__ == "cupy"` 并要求 `check_QUDA()` 通过（`lattice/generator/density_perambulator.py:35-38`） |
| `DensityPerambulatorGenerator.load` | `(key: str)` | `lattice/generator/density_perambulator.py:76` | 将 gauge 读入 QUDA dirac 并加载 eigenvector |
| `DensityPerambulatorGenerator.calc` | `(ti: int, tf: int, tau: int)` | `lattice/generator/density_perambulator.py:83` | 返回 `VSSV.transpose(2,3,4,5,6,7,8,0,1)`，即 `(len(gamma_list), len(momentum_list), Lz, Ly, Lx, Ns, Ns, Ne, Ne)`（`lattice/generator/density_perambulator.py:176`） |

---

## ③ 数据结构与数组形状约定

以 example 的 16³×128 弱场组态、Ne=70 为例（`example/gen_twopt.py:15-46`）：

| 数据 | 磁盘 shape | 含义 | 典型示例 |
|---|---|---|---|
| elemental | `(num_disp, num_mom, Lt, Ne, Ne)` | interpolation matrix O_{i,j} | `[4, 123, 128, 70, 70]`（`example/gen_twopt.py:35`） |
| perambulator | `(Lt, Lt, Ns, Ns, Ne, Ne)` | 源/汇时间 × spin × 本征模 | `[128, 128, 4, 4, 70, 70]`（`example/gen_twopt.py:42`） |
| eigenvector（逐时间片） | `(Lt, Ne, Vol, Nc)` | Laplace eigenvector | `[Lt, Ne, Lz*Ly*Lx, Nc]`（`example/gen_density_peram.py:20-24`） |
| gauge（ILDG） | `(Lt, Lz, Ly, Lx, Nd, Nc, Nc)` | 规范链接 | `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]`（`example/gen_density_peram.py:17-19`） |
| 2pt（单粒子） | `(Nop, Lt)` | 对源时间平均后 | `twopoint([op_pi, op_pi2], ...)`（`example/gen_twopt.py:45`） |
| 2pt 矩阵 | `(Nop, Nop, Lt)` | `(isrc, isnk)` 元素 | `twopoint_matrix(...)`（`example/gen_twopt.py:52`） |
| 多动量 2pt 矩阵 | `(Nmom·Nop·Nop, Nt, Lt)` | 逐动量展开的矩阵 | `twopoint_matrix_multi_mom(...)`（`lattice/correlator/one_particle.py:380`） |
| 两粒子 2pt（多动量） | `(Nsrc_mom, Nsnk_mom, Ndiagram, Nt)` | 动量对 × 图 | `twopt_tosave = backend.zeros((1*len(mom_list), 1*len(mom_list), 6, Nt))`（`example/gen_two_particle_corr_mom.py:141`） |
| density perambulator | `(len(gamma_list), len(momentum_list), Lz, Ly, Lx, Ns, Ns, Ne, Ne)` | γ/动量/空间/spin/本征模 | `calc` 返回值转置（`lattice/generator/density_perambulator.py:176`） |

注意：elemental 磁盘轴序在两套数据集中写法不同——`[4, 123, 128, 70, 70]`（num_disp 在前，`example/gen_twopt.py:35`）与 `[128, 13, 123, 70, 70]`（Lt 在前，`example/gen_two_particle_corr.py:44`），对应两个不同的集群数据集，使用时必须按实际 `.npy` 的 shape 填写，不能混用。

---

## ④ 算法与流程

### 4.1 单粒子两点函数与 effective mass

**示例**：`example/gen_twopt.py`。**前置要求**：cupy 可用（不可用时脚本自动回退 numpy，`example/gen_twopt.py:1-11`）；集群上的 elemental/perambulator `.npy` 数据路径。

1. **后端选择**：尝试 `set_backend("cupy")` 并探测 CUDA 设备数，失败回退 `set_backend("numpy")`（`example/gen_twopt.py:1-11`）。
2. **构造 operator**：`Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)` 取第 0 行，`InsertionRow(0,0,0)` 固定零动量，包成 `Operator("pi", [...], [1])`；第二个 operator `pi2` 以系数 `[3, 1]` 混合 π 与 B1×∇ 行（`example/gen_twopt.py:19-25`）。
3. **加载数据**：`ElementalNpy`/`PerambulatorNpy` 构造惰性加载器（`example/gen_twopt.py:31-42`），`load("2000")` 读入单 cfg 的 `e`、`p`（`example/gen_twopt.py:44-46`）。
4. **计算 2pt**：`twopoint([op_pi, op_pi2], e, p, list(range(128)), 128)` 返回 `(Nop, Lt)`（`example/gen_twopt.py:55`）；取实部后用 `arccosh((C(t+1)+C(t-1))/(2C(t)))` 得 effective mass（`example/gen_twopt.py:56-58`）。
5. **计算 2×2 矩阵**：`twopoint_matrix` 同参数（`example/gen_twopt.py:60`），取 `[0,0]`/`[1,1]` 元素沿时间轴 roll 求 effective mass（`example/gen_twopt.py:61-70`）。
6. **动量壳求和**：`twopoint_mom2(pi_A1[0], 2, e, p, list(range(128)), 128)` 对 |p|²=2 的所有动量求和（`example/gen_twopt.py:74-77`）。

### 4.2 带动量的 2pt 矩阵（GEVP 输入）

**示例**：`example/gen_twopt_matrix_mom.py`。**前置要求**：同 4.1，另需 cfglist 文件（见 ⑤）。

1. 构造两个 `Insertion`：h_c 用 `(GammaName.B1, IDEN, T1)`，J/ψ 用 `(GammaName.RHO, IDEN, T1)`，各取第 `[2]` 行（`example/gen_twopt_matrix_mom.py:27-31`）。
2. 手工列出 7 个动量的 `momlist`（|p|²=1..8 的代表点；这些动量均已验证属于 `momDict_mom9` 的 values，`example/gen_twopt_matrix_mom.py:33-41`）。
3. 加载 elemental 与 charm `PerambulatorNpy`（`example/gen_twopt_matrix_mom.py:43-54`）。
4. 逐 cfg：`Dispatch("./cfglist.689.txt", "2pt")` 迭代分发（`example/gen_twopt_matrix_mom.py:57,60`）；已存在输出文件则跳过（断点续算，:62-63）；调用 `twopoint_matrix_multi_mom([ins_hc[2], ins_jpsi[2]], momlist, e, p, list(range(128)), 128)` 并 `backend.save(save_path, twopt)` 落盘（`example/gen_twopt_matrix_mom.py:72-73`）。该矩阵即为后续 GEVP 的输入。

### 4.3 两粒子关联（D D̄*/D̄*D̄*，无动量投影与带动量投影两版）

**示例**：`example/gen_two_particle_corr.py`（零动量）、`example/gen_two_particle_corr_mom.py`（动量壳 mom2=1）。**前置要求**：cupy/集群数据；`cfglist.700.txt` / `cfglist.689.txt` 须存在于运行目录；charm perambulator。

1. **构造 insertion/operator**：D（π×IDEN）、D*（ρ×IDEN，取 T1 第 `[2]` 行）、χc1（A1×IDEN）（`example/gen_two_particle_corr.py:23-33`）。
2. **构造图**：4 个 `QuarkDiagram(..., validate=False)` 邻接矩阵（:51-85）共享一个 4-slot 顶点列表 `[D_src, Ds_src, D_snk, Ds_snk]`——每个矩阵只是收缩 *terms*，空槽位由同批其它 term 补全，故须关闭 quark-link 校验（`example/gen_two_particle_corr.py:40-44` 注释明确说明）。带动量版另含 χc1→DD*、χc1→J/ψη 两个 4 顶点图（`example/gen_two_particle_corr_mom.py:100-117`）。
3. **组装线与顶点**：`Propagator(perambulator_light, Lt)`（轻 quark 非局域线）、`Propagator(perambulator_charm, Lt)`（charm 线）、`PropagatorLocal(perambulator_light, Lt)`（局域线）；四个 `Meson(elemental, op, True/False)` 作源/汇顶点（`example/gen_two_particle_corr.py:103-109`）；邻接矩阵取值 1/2/3 即分别索引到 charm/light/local 线。
4. **带动量版**：`for i_mom2 in range(mom_max)[1:2]` 取 mom2=1 壳，`get_mom2_list(1)` 枚举 6 个动量，`get_AB_opratorlist_row(ins_D[0], ins_Dstar[2], mom_list)` 生成 A/B 两侧 operator 列表（B 侧自动取 -p）（`example/gen_two_particle_corr_mom.py:133-138`）；对每个动量对预建 4 个 `Meson` 列表以只初始化一个缓存池（`example/gen_two_particle_corr_mom.py:149-157`，注释明确强调）。
5. **时间循环**：对每个 `t_src`，`compute_diagrams_multitime([4 或 6 图], [t_src, t_src, t_snk, t_snk], [D_src, Ds_src, D_snk, Ds_snk], [None, line_charm, line_light, line_local_light])`，其中 `t_snk = numpy.arange(Nt)`（必须 numpy，见 ⑤；`example/gen_two_particle_corr.py:124`、`example/gen_two_particle_corr_mom.py:143-146`），返回 `(Ndiagram, Nt)`；`twopt += backend.roll(tmp, -t_src, 1)` 把源时间 roll 到统一原点后累加，最后 `/= 128` 平均并打印 effective mass（`example/gen_two_particle_corr.py:127-140`）。
6. **汇总存储**：带动量版将结果累入 `twopt_tosave[i_src, i_snk, 0:6]`，shape `(Nsrc_mom, Nsnk_mom, 6, Nt)`（`example/gen_two_particle_corr_mom.py:141, 188`；`backend.save` 当前被注释，:194）。

### 4.4 quark 图计算与绘制

**计算（连通/不连通/三顶点图）**——示例 `example/gen_twopt_diagram.py`：

1. 设置 `CUPY_ACCELERATORS=cub,cutensor` 与 `CUTENSOR_PATH`（集群路径），cupy 回退逻辑同前（`example/gen_twopt_diagram.py:1-13`）。
2. 构造 π、π₂（`GammaName.PI_2`）、ρ 的 operator（`example/gen_twopt_diagram.py:26-36`）。
3. 图：`connected=[[0,1],[1,0]]`（双 quark 线）、`disconnected=[[2,0],[0,2]]`（两条局域线）、`rho2pipi=[[0,1,0],[0,0,2],[1,0,0]]`（ρ→ππ 三顶点图）；三者均未传 `validate`，默认校验开启（`example/gen_twopt_diagram.py:60-72`）。
4. 逐 cfg 以 `usedNe=50` 截断本征模加载（`Meson.load`/`Propagator.load` 均接受 `usedNe`（`example/gen_twopt_diagram.py:94-97`），对每个 `t_src` 收缩后 roll 累加、平均（`example/gen_twopt_diagram.py:99-109`）。
5. 连通/不连通重组：`twopt[1] = -twopt[0] + 2*twopt[1]`（`example/gen_twopt_diagram.py:110`，连通/不连通贡献的物理组合惯例）；随后第二段计算 ρ→ππ 三顶点图（`example/gen_twopt_diagram.py:115-130`）。

**绘制**——示例 `example/gen_multi_draw_diagrams.py`：

1. 以 6×6 邻接矩阵列表定义 7 张 D/D*/χc1 相关 quark 图（`example/gen_multi_draw_diagrams.py:1-48`；其中 `graph_DsD_DsD_3` 标注 `# Warning: missing!!!`，对应 quark 线缺失的占位图）。
2. 顶点属性以 dict 列表描述（`pos`/`type`/`name`，name 用 LaTeX 数学串如 `R"$\bar{D}^*$"`）；`draw_single_diagram` 画单张（`example/gen_multi_draw_diagrams.py:74`），`draw_multi_diagrams` 批量画（:87；`save_path` 参数当前被注释，:98）。
3. 环境要求：`draw_single_diagram` 渲染 LaTeX 顶点名需要本机安装 LaTeX；保存文件需取消 `save_path` 注释。

### 4.5 density perambulator 生成

**示例**：`example/gen_density_peram.py`。**前置要求（硬性）**：GPU + cupy + PyQuda ≥0.9.0（`check_QUDA()` 不通过则脚本直接 `raise ImportError`，`example/gen_density_peram.py:1-4`）；集群上的 gauge `.lime` 与逐时间片 Laplace eigenvector。

1. `check_QUDA()` 探测 PyQuda（`example/gen_density_peram.py:1-4`）；`set_backend("cupy")`（`example/gen_density_peram.py:9`，`DensityPerambulatorGenerator` 内部亦断言 cupy，`lattice/generator/density_perambulator.py:38`）。
2. 构造 `GaugeFieldIldg` 与 `EigenvectorTimeSlice` 加载器（`example/gen_density_peram.py:19-25`）。
3. 构造 `DensityPerambulatorGenerator`，传入 clover Wilson 参数（`mass`/`xi_0`/`nu`/`clover_coeff_t`/`clover_coeff_r`/`t_boundary=-1`/`multigrid`）、`gamma_list=[1,2,4,8]` 与 `momentum_list=[(0,0,0)]`（`example/gen_density_peram.py:28-42`）。
4. 设 QUDA verbosity：`perambulator.dirac.invert_param.verbosity = enum_quda.QUDA_SUMMARIZE`（`example/gen_density_peram.py:44`）。
5. 逐 cfg：`perambulator.load(cfg)`（gauge 进 QUDA dirac + 加载 eigenvector，`lattice/generator/density_perambulator.py:76-81`）→ `perambulator.calc(0, 64, 32)`（源 t=0、终 t=64、τ=32；返回 shape 见 ②.7）（`example/gen_density_peram.py:54`）。
6. 清理：`perambulator.dirac.destroy()`（`example/gen_density_peram.py:59`）。保存代码当前被注释（`example/gen_density_peram.py:57`）。

### 4.6 辅助脚本

| 脚本 | 用途 | 关键 API |
|---|---|---|
| `example/gen_two_particle_operators.py` | 打印三组两粒子 Cartesian 算符基（S-wave PV、P-wave PV、P-wave VV） | `two_particle_Cartesian_basis`（纯符号计算，无数据依赖） |
| `example/hardcoding_OhD.py` | Oh 群表重生成参考片段（docstring 声明：全部代码被注释，运行会重新生成 `hardcoded_rep.py` 预置的表） | `genMatrixGroupOhD`/`genIrrepOhD`/`genLittleGroupIrrep`/`gen_connection`/`reductionToLittleGroup`（`lattice/symmetry/gen_hardcoded_rep.py:23,76,182,239,422`） |

---

## ⑤ 不变量与校验

| 不变量 | 内容 | 出处 |
|---|---|---|
| quark-link 校验 | `QuarkDiagram` 默认 `validate=True`：校验每条 quark 线有归属（顶点为孤立/介子/重子端点）；仅机械收缩 *terms* 传 `False`（example 中 4+6 个 term 图均如此） | `lattice/quark_diagram.py:550-551`；`example/gen_two_particle_corr.py:41-44` |
| `t_snk` 必须用 numpy | `compute_diagrams_multitime` 以 `isinstance(time, (int, np.integer))` 判定 multitime 时间轴，cupy 的 int 数组不满足该判定，故 `t_snk = numpy.arange(Nt)` 是硬要求 | `lattice/quark_diagram.py:1693`；`example/gen_two_particle_corr_mom.py:143-146` 注释明确提醒 |
| `usedNe` 归一 | `Meson.load`/`Propagator.load` 的 `usedNe` 越界/非 int/负值被拒（含 `True`，因 bool ⊂ int） | `lattice/propagators.py:46-60, 239-244, 783-788` |
| `PropagatorLocal` 时间约束 | `PropagatorLocal.get(t_source, t_sink)` 断言 `t_source == t_sink` | `lattice/propagators.py:913-917` |
| 动量合法性 | `InsertionRow(px, py, pz)` 要求动量串 `"px py pz"` 存在于 `momentum_dict` values 中，否则 `list.index` 抛 ValueError | `lattice/insertion/__init__.py:227-229` |
| operator 行/系数配对 | `Operator` 断言 `len(insertion_rows) == len(coefficients)` | `lattice/insertion/__init__.py:154-157` |
| density perambulator 环境 | `check_QUDA()` 通过 + `set_backend("cupy")` 是 `DensityPerambulatorGenerator` 的硬前置 | `lattice/generator/density_perambulator.py:35-38` |
| `J>1` 未实现 | `two_particle_Cartesian_basis` 对 J>1 抛 `NotImplementedError` | `lattice/symmetry/two_particle.py:121-122` |
| cfglist 文件 | `Dispatch.__init__` 构造期即 `open(filename)` 读 cfglist；`cfglist.689.txt`/`cfglist.700.txt` 不在仓库内，运行相应 example 前须自行放置（每行一个 cfg 名） | `lattice/dispatch.py:64-74` |

**已知注释与代码不一致（使用时以代码为准）**：

- `example/gen_two_particle_corr_mom.py:26` 注释 "shells mom2 = 0, 1, 2, 3"，而 :134 `for i_mom2 in range(mom_max)[1:2]` 实际只取 mom2=1 壳。
- `example/gen_density_peram.py:46-57` 的 `out_prefix`/`out_suffix` 与 `backend.save` 已备好但保存调用被注释，当前只计算不落盘。

---

## ⑥ 相关测试

| 测试文件 | 覆盖的工作流环节 | markers / 数据依赖 |
|---|---|---|
| `test/test_meson_spectrum.py` | `twopoint`/`twopoint_matrix` 端到端（用 `test/weak_field.elemental.npy` + `test/weak_field.perambulator.npy`，shape `[13,6,Lt,Ne,Ne]` elemental 与 `[Lt,Lt,4,4,Ne,Ne]` perambulator） | `gpu`；需 `test/weak_field.*` 数据 |
| `test/test_quark_diagram_contraction.py` | `QuarkDiagram` 收缩管线：`compute_diagrams(_multitime)` 签名、einsum shape、multitime 输出 rolling | 无 markers；Mock + 小数组 |
| `test/test_current_contraction.py` | `QuarkDiagram` current 顶点 + `compute_diagrams(_multitime)` | `integration`；手工构造图 |
| `test/test_quark_draw.py` | `draw_single_diagram` 等绘图 API | 无 markers；无数据依赖 |
| `test/test_ne_provenance.py` | `usedNe` 归一、缓存身份（(key, usedNe, loader 签名, 数据签名, operator 签名, dagger)） | 无 markers |
| `test/test_perambulator.py` / `test_perambulator_mpi.py` / `test_perambulator_phase3.py` | perambulator 生成管线（PyQuda，单卡/MPI/新旧路径一致性） | `mpi`；需 `test/weak_field.lime` 等 |
| `test/test_flavor_structure.py` / `test_quark_contract.py` | quark/介子/重子 flavor 收缩（`quark_contract`） | 无 markers |
| `test/test_sampling_weight.py` | 采样权重 w(k)=C(L³,k)/C(usedNp,k) 与 highmode 投影公式 | 无 markers |

说明：`two_particles.py`、`dispersion_relation.py`、`symmetry/two_particle.py` 与 `dispatch.py` 目前没有专门的测试文件；无 GPU/MPI 环境下可直接运行的最小回归子集为 `pytest test/test_gamma.py test/test_sampling_weight.py test/test_vertex_type.py test/test_sparsened_point.py test/test_symmetry.py -q`。
