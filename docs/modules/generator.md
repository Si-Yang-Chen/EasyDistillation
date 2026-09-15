# `lattice/generator/` —— distillation 数据生成模块

> 定位：`lattice/generator/` 承担 distillation 流水线的"数据生成"职责——从 gauge configuration 出发，依次产出 eigenvector（3D covariant Laplacian 低模）、elemental 矩阵元（本征矢 × smeared gauge link）、noisevector（随机噪声源）、sparsened point（稀疏点源坐标表）与 perambulator 族产物（基于 PyQUDA/QUDA Dirac 求逆的 `VSV/PSV/PSP/VSSV`），供 `lattice/insertion/`、`lattice/correlator/` 等下游收缩消费。本文所有行为描述均以源码为准，出处以 `path:line` 给出。

---

## 1. 模块概览

### 1.1 文件与符号一览

| 文件 | 公开符号 | 职责 |
|---|---|---|
| `lattice/generator/elemental.py` | `ElementalGenerator`（:38）、`CurrentElementalGenerator`（:996）、`re_combine`（:2582） | 本征矢 + stout smeared gauge link → elemental 矩阵元（derivative 展开或 GaugeLink 位移路径）；v/p 基间 current 顶点块；位移基→插入基线性重组 |
| `lattice/generator/displacement_elemental.py` | `DisplacementElementalGenerator`（:11） | 六方向平均协变位移的对称化 elemental（按位移阶数） |
| `lattice/generator/eigenvector.py` | `EigenvectorGenerator`（:29）、`get_overlap_matrix`（:455） | 逐时间片求解 3D Laplacian 最低 `Ne` 个本征对 |
| `lattice/generator/noisevector.py` | `NoisevectorGenerator`（:11） | 本征矢低模按 dilution 方案随机组合成 noise vector，可选 high-mode 补充 |
| `lattice/generator/sparsened_point.py` | `generate_sparsened_points`（:9） | 每时间片独立生成互不重复的随机空间点坐标表 |
| `lattice/generator/perambulator.py` | `sampled_point_local_indices`（:11）、`PerambulatorGenerator`（:43） | PyQUDA Dirac 求逆 → perambulator `VSV`/`PSV`/`PSP` |
| `lattice/generator/density_perambulator.py` | `DensityPerambulatorGenerator`（:12） | 双源时刻 + gamma 插入 + 动量相位的逐空间点 `VSSV`（固定汇时刻 tau） |
| `lattice/generator/generalized_perambulator.py` | `GeneralizedPerambulatorGenerator`（:13） | 同上收缩，但保留全部汇时刻（时间分辨 `VSSV`） |
| `lattice/generator/stout_smear.cu` | `stout_smear<double>` CUDA kernel（:159） | 单步 stout link smearing 的 GPU 实现，与 Python `_stout_smear_ndarray` 同一公式 |
| `lattice/generator/__init__.py` | 聚合导出（:1-8） | 导出上述全部 9 个符号 |

### 1.2 依赖关系

模块内部 import（`..` 相对导入）：

| 模块 | 依赖的仓库内模块 |
|---|---|
| `elemental.py` | `..constant`、`..backend.get_backend`、`..preset`（`GaugeField/Eigenvector/PointSource`）、`..insertion.phase.MomentumPhase`（顶层，:7-11）；函数体内懒加载 `..insertion.derivative.derivative`、`..insertion.gauge_link`（`GaugeLink/DirectedCurrentBasis`）、`..insertion.current.build_current_raw_contract`、`..backend.check_QUDA`、`pyquda_utils.io` |
| `displacement_elemental.py` | `..constant`、`..backend.get_backend`、`..preset`、`..insertion.phase.MomentumPhase`；懒加载 `..backend.check_QUDA`、`pyquda_utils.io` |
| `eigenvector.py` | `..constant`、`..backend`（`get_backend/check_QUDA`）、`..preset.GaugeField`；懒加载 scipy/cupyx `sparse.linalg`、`pyquda.pyquda`、`pyquda.enum_quda`、`pyquda_utils.core` |
| `noisevector.py` | `..constant`、`..backend`、`..preset.Eigenvector` |
| `sparsened_point.py` | 仅 numpy |
| `perambulator.py` / `density_perambulator.py` / `generalized_perambulator.py` | `..constant`、`..backend`、`..preset`、`opt_einsum.contract`；函数体内懒加载 `pyquda_utils.core/io`（外部安装包，提供 `LatticeInfo`、`getDirac`、`LatticeFermion`、`MultiLatticeFermion`、`invert`、`cb2`、`io.readQIOGauge`、`source.source` 等） |

外部依赖：`opt_einsum`（全部收缩）、`pyquda`/`pyquda_utils`（三个 perambulator 模块与 stout smearing 的 QUDA 路径，不在本仓库内）。

### 1.3 被谁消费（仓库内调用点，grep 确认）

| 符号 | 消费方 |
|---|---|
| `ElementalGenerator` | `lattice/__init__.py` 导出；`test/test_elemental.py`、`test/scripts/` 下多个对比脚本 |
| `CurrentElementalGenerator` | `lattice/__init__.py` 导出；`lattice/insertion/current.py:1023` 附近 `compute_elemental` 的 duck-type 检查要求 generator 提供 `calc_all(t)`；`test/test_temporal_current_elemental.py` |
| `DisplacementElementalGenerator` | `lattice/__init__.py` 导出；`test/test_displacement_elemental.py` |
| `EigenvectorGenerator` | `lattice/__init__.py` 导出；`test/test_eigenvector.py` |
| `NoisevectorGenerator` | 仅 `lattice/generator/__init__.py:2` 与 `lattice/__init__.py` 导出；仓库内无其他调用点、无测试 |
| `generate_sparsened_points` | 仅 `lattice/generator/__init__.py:8` 导出；`test/test_sparsened_point.py` 直接 import |
| `PerambulatorGenerator` | `test/test_perambulator.py`、`test/test_perambulator_mpi.py`、`test/test_perambulator_phase3.py` |
| `DensityPerambulatorGenerator` | 仅导出；`example/gen_density_peram.py` |
| `GeneralizedPerambulatorGenerator` | 仅导出；无测试、无示例 |
| `re_combine` / `get_overlap_matrix` | 仓库内（排除 archive）无任何调用点，仅定义 |

---

## 2. 公开 API 参考

### 2.1 `ElementalGenerator`（elemental.py:38）

在单条 gauge configuration 上，把 Laplace eigenvector 与经 stout smearing 的空间 gauge link 组合成 elemental 矩阵元。两种模式：`calc_deriv` 按 derivative 单项式展开，`calc_disp` 按 `GaugeLink` 位移路径编码。

```python
def __init__(
    self,
    latt_size: List[int],
    gauge_field: GaugeField,
    eigenvector: Eigenvector,
    num_nabla: int = 0,
    momentum_list: List[Tuple[int]] = [(0, 0, 0)],
    dilution: Tuple = None,
    is_blending: bool = False,
    usedNe: int = None,
    usedNp: int = None,
    calc_mode: Literal["calc_deriv", "calc_disp"] = "calc_deriv",
    debug: bool = False,
) -> None
```
（elemental.py:47-66）

| 参数 | 含义 |
|---|---|
| `latt_size` | `[Lx, Ly, Lz, Lt]` |
| `gauge_field` / `eigenvector` | 数据句柄（`lattice/preset.py` 的 lazy loader 容器） |
| `num_nabla` | derivative/nabla 的最大阶数，决定 `num_derivative` 与 `num_disp` |
| `momentum_list` | 动量列表，逐项形如 `(px, py, pz)` |
| `dilution` / `is_blending` | blending（随机稀释混合）无偏估计系数的构造输入，见 §4.1.4 |
| `usedNe` | 实际使用的本征矢数，缺省取 `eigenvector.Ne`（:96） |
| `usedNp` | 仅赋值保留兼容，类内未使用（:97-99 注释 "Not used in ElementalGenerator, but kept for compatibility"） |
| `calc_mode` | `"calc_deriv"` → `calc(t)` 分派到 `calc_deriv(t)`；`"calc_disp"` → `calc_disp(t)`（:722-724） |
| `debug` | 打印规模信息等调试输出（如 :736-737） |

关键属性：`Ne = eigenvector.Ne`；`num_derivative = (3 ** (num_nabla + 1) - 1) // 2`（:89，即 `{dx,dy,dz}` 上总阶数 ≤ `num_nabla` 的单项式个数，`num_nabla=2` 时为 13）；`num_disp = list(GaugeLink.nmax_generator(num_nabla))[-1]`（:90，合法 gauge 路径编码总数，n=0:1、n=1:7、n=2:37）；`derivative_list`（:91，由 `lattice/insertion/derivative.py` 的 `derivative(n)` 把 n 的 base-3 数字展开为方向元组）；`_momentum_phase = MomentumPhase(latt_size)`（:129）。

| 方法 | 签名 | 说明 |
|---|---|---|
| `load` | `load(self, key: str)`（:174） | 读 gauge（转置为 `(Nd, Lt, Lz, Ly, Lx, Nc, Nc)` 后取 `[:Nd-1]` 存入 `self._U`）、记录 gauge 文件路径 `self._gauge_field_path`（供 QUDA smearing 重读）、读 eigenvector |
| `project_SU3` | `project_SU3(self)`（:179） | 迭代 `U ← (U + (U⁻¹)ᴴ)/2` 直到 `max|U − (U⁻¹)ᴴ| ≤ 1e-15` 且 `max|U·Uᴴ − I| ≤ 1e-15`，把读入的 gauge 投影回 SU(3) |
| `stout_smear` | `stout_smear(self, nstep, rho)`（:360） | 分派顺序：numpy → `_stout_smear_ndarray`；cupy → 有 CUDA kernel 用 kernel，否则 QUDA 可用走 QUDA，否则 `_ndarray` |
| `calc` | `calc(self, t: int)`（:722） | 按 `calc_mode` 分派；返回复用缓冲区 `self._VPV` / `self._disp_buffer` 本身（非拷贝） |
| `calc_deriv` | `calc_deriv(self, t: int)`（:668） | 见 §4.1.1；返回 `(num_derivative, num_momentum, usedNe, usedNe)` complex128 |
| `calc_disp` | `calc_disp(self, t: int)`（:727） | 见 §4.1.2；返回 `(num_disp, num_momentum, usedNe, usedNe)` complex128 |
| `_gauge_links_product` | `(self, gauge_list, t=None)`（:375） | 沿路径在全格点累积 gauge link 乘积；`t=None` 返回 `(Lt, Lz, Ly, Lx, Nc, Nc)`，`t=int` 返回 `(Lz, Ly, Lx, Nc, Nc)`；空路径返回 `None` |
| `_apply_gauge_links_to_points` | `(self, point_left, gauge_list, t=None)`（:459） | 点基版本的链接乘积应用；本类内无调用者，与 `CurrentElementalGenerator` 共用实现 |
| `_nD` / `_disp` | `(self, V, U, deriv)`（:646） / `(self, V, U, d)`（:657） | 沿方向序列的复合协变位移算子（前向/后向链接） |
| `re_combine_auto` 等 | 见 §2.3 | 位移基→插入基线性重组（类方法） |

### 2.2 `CurrentElementalGenerator`（elemental.py:996）

在 eigenvector 基（v）与稀疏 point 源基（p）之间生成 current/位移顶点的四类 elemental 块，供 `lattice/insertion/current.py` 的 `compute_elemental` 消费。

```python
def __init__(
    self,
    latt_size: List[int],
    gauge_field: GaugeField,
    eigenvector: Eigenvector,
    point: "PointSource",
    num_nabla: int = 0,
    momentum_list: List[Tuple[int]] = [(0, 0, 0)],
    usedNe: int = None,
    usedNp: int = None,
    debug: bool = False,
) -> None
```
（elemental.py:1023-1043）

与 `ElementalGenerator` 的区别：无 `calc_mode`/`dilution`/`is_blending`；多一个 `point: PointSource`（持有 `Np` 与坐标表）。`usedNe/usedNp` 经 `_bounded_count`（elemental.py:19-35）校验：类型必须是 int（bool 被拒）、须满足 `0 <= used <= available`，否则 `TypeError`/`ValueError`（:1074）。

| 方法 | 位置 | 返回形状 |
|---|---|---|
| `load(key)` | :1086 | 同时读 gauge（保留全部 4 个方向到 `self._current_U`）、eigenvector（截断到 `[:usedNe]`）、point（截断到 `[:usedNp]`）；随后 `self.gauge_field.data = None` 释放缓存 |
| `clear_loaded_data()` | :1097 | 释放已加载数据；cupy 下显式 synchronize + `gc.collect()` + memory pool `free_all_blocks()` |
| `project_SU3()` | :1148 | 同 `ElementalGenerator`，结果同时写回 `self._current_U[:Nd-1]` |
| `stout_smear(nstep, rho)` | :1274 | 同 `ElementalGenerator` 分派，末尾把 `self._U` 写回 `self._current_U[:Nd-1]` |
| `calc_directed_current_raw(boundary="periodic")` | :1292 | `{"v2v": (8, Lt, num_momentum, usedNe, usedNe), "contract": build_current_raw_contract(...)}`；8 个方向由 `DirectedCurrentBasis.DIRECTIONS` 给出（±x/±y/±z/±t） |
| `calc_v2p(t)` | :1663 | `(num_disp, usedNe, usedNp, Nc)`，无动量轴（point 块不投影动量） |
| `calc_p2v(t)` | :1772 | `(num_disp, usedNp, Nc, usedNe)`，无动量轴 |
| `calc_p2p(t)` | :1872 | **列表**（每 `disp_idx` 一项）：零位移项 `{"type": "identity"}`；其余 `{"type": "sparse", "indices": (K,2) int32, "values": (K,Nc,Nc)}`；空对时 `K=0` |
| `calc_v2v(t)` | :2025 | `(num_disp, num_momentum, usedNe, usedNe)`；算法与 `ElementalGenerator.calc_disp` 相同，但不复用预分配 buffer，每次 `backend.zeros` |
| `calc_all(t)` | :2138 | `{"v2v": (num_disp, num_momentum, usedNe, usedNe), "v2p": (num_disp, usedNe, usedNp, Nc), "p2v": (num_disp, usedNp, Nc, usedNe), "p2p": list[dict]}`；单次 `gauge_link_product` 复用同时产出四块 |
| `_gauge_links_product` / `_apply_gauge_links_to_points` | :1392 / :1476 | 与 `ElementalGenerator` 同名方法逐字相同（拷贝） |
| `_re_combine_dense/_sparse` / `re_combine_auto` | :2430 / :2445 / :2490 | 同 §2.3，类级 benchmark 状态与 `ElementalGenerator` 相互独立 |

`lattice/insertion/current.py:1023-1040` 中 `compute_elemental` 强制要求 generator 提供 `calc_all(t)`（否则 `TypeError("compute_elemental requires a CurrentElementalGenerator")`），且返回必须是 Mapping 并包含 `elemental_key`。

### 2.3 `re_combine` 系列（elemental.py）

位移基 → 插入基的线性重组：`insertion_list[i]` 是 `[(coeff, disp_idx), ...]` 的列表，第 i 个插入算符由位移基元素的线性组合给出。

| 符号 | 签名 | 实现 |
|---|---|---|
| `ElementalGenerator._re_combine_dense` | `classmethod (cls, array, insertion_list, axis, out=None)`（:844） | 构造稠密系数矩阵后 `tensordot(coeff, array, axes=([1],[axis]))` + `moveaxis` |
| `ElementalGenerator._re_combine_sparse` | 同上（:859） | 系数矩阵改用 scipy/cupyx `coo_matrix→csr` 与 reshape 后的 2D 数组相乘 |
| `ElementalGenerator.re_combine_auto` | 同上（:904） | 首次调用时两种实现都跑一遍，`allclose(rtol=1e-6, atol=1e-10)` 交叉验证（失败抛 `ValueError("re_combine validation failed: ...")`），随后各计时取平均选快者；状态存于类属性 `_recombine_*`（:39-45） |
| `re_combine`（模块级） | `re_combine(array, insertion_list, axis, out=None)`（:2582-2583） | `ElementalGenerator.re_combine_auto` 的薄包装 |

`CurrentElementalGenerator` 内有逐字相同的一套类方法（:2430-2490）。两套实现与类级状态互不干扰。

### 2.4 `DisplacementElementalGenerator`（displacement_elemental.py:11）

计算对称化位移 elemental：对每个阶数 `dist ≤ distance`，取沿 ±x/±y/±z 六方向重复 `dist` 次协变位移后的 eigenvector 与原 eigenvector 的逐动量重叠矩阵。

```python
def __init__(
    self,
    latt_size: List[int],
    gauge_field: GaugeField,
    eigenvector: Eigenvector,
    distance: int = 0,
    momentum_list: List[Tuple[int]] = [(0, 0, 0)],
) -> None
```
（displacement_elemental.py:12-21）

| 方法 | 位置 | 说明 |
|---|---|---|
| `_D(V, U, distance)` | :53 | 见 §4.2；**依赖 `self.Vd` 跨调用残留状态递推** |
| `load(key)` | :73 | 读 gauge（`[:Nd-1]`）、记录文件路径到 `self._gauge_field_data`（:75）、读 eigenvector |
| `calc(t)` | :78 | 见 §4.2；返回 `self._VPV`，形状 `(distance+1, num_momentum, Ne, Ne)` complex128 |
| `project_SU3()` | :99 | 同 `ElementalGenerator` |
| `_stout_smear_ndarray` / `_stout_smear_cuda_kernel` / `_stout_smear_quda` / `stout_smear` | :111 / :180 / :191 / :203 | 与 `ElementalGenerator` 同一算法的拷贝；分派顺序也相同（numpy → ndarray；cupy → kernel → QUDA → ndarray） |

注意：`_VPV` 的轴 0 是"位移阶数"而非 GaugeLink 路径编号（displacement_elemental.py:38）——这是与 `ElementalGenerator.calc_disp` 的 `num_disp` 轴的本质区别。

### 2.5 `EigenvectorGenerator`（eigenvector.py:29）与 `get_overlap_matrix`

从（stout smeared 的）3D gauge 场在每个时间片上求解 3D covariant Laplacian 的最低 `Ne` 个本征对。

```python
def __init__(self, latt_size: List[int], gauge_field: GaugeField, Ne: int, tol: float) -> None
```
（eigenvector.py:30-34）

| 方法 | 位置 | 说明 |
|---|---|---|
| `load(key)` | :57 | 读 gauge（转置 copy）、打印读盘吞吐、释放 `gauge_field.data`、记录 `self._gauge_field_path` |
| `project_SU3()` | :65 | 同 `ElementalGenerator`，仅对空间链接 `[:Nd-1]` 操作并写回 |
| `stout_smear(nstep, rho)` | :249 | 分派顺序与 `ElementalGenerator` **不同**：cupy 下优先 QUDA（`check_QUDA()` 先于 kernel 判断，:257-260），其次 CUDA kernel，最后 cupy ndarray；numpy 下 ndarray。每次都 print 所选路径 |
| `laplacian_cupy_numpy(t, apply_renorm_phase)` | :268 | scipy/cupyx `LinearOperator` 包装 `_Laplacian` 后 `eigsh(A, Ne, which="SA", tol)`；按 `argsort(evals)` 排序；`apply_renorm_phase=True` 时把每个 eigenvector 乘 `exp(-i·angle(evecs[:,0]))` 固定首分量相位；返回 `(evecs[Ne,Lz,Ly,Lx,Nc], evals)` |
| `laplacian_quda(t, apply_renorm_phase, polynomial_degree, lambda_cut)` | :297 | QUDA `QUDA_LAPLACE_DSLASH`（`mass=-1`、`kappa=1/(2(Nd-1))`、`laplace3D=3`）+ TR-LANCZOS（`QUDA_SPECTRUM_SR_EIG`）；`polynomial_degree>0 and lambda_cut>0` 时启用 Chebyshev 多项式加速；`evals *= 2(Nd-1)` 换算回 Laplacian 本征值 |
| `laplacian_quda_scipy(t, apply_renorm_phase)` | :374 | QUDA 的 `pure_gauge.laplace(...)` 只做 matvec，本征求解交给 cupyx `eigsh`；输出经 `lexico` 还原词典序 |
| `calc(t, apply_renorm_phase=True, polynomial_degree=0, lambda_cut=0.0)` | :428-434 | 分派：cupy+QUDA 且 Chebyshev 参数有效 → `laplacian_quda`；cupy+QUDA → `laplacian_quda_scipy`；cupy/numpy → `laplacian_cupy_numpy`；其他后端 → `NotImplementedError`（:429 附近注释写明不用 QUDA 自带 eigensolver） |

模块级函数：

```python
def get_overlap_matrix(eigenvector_data, point_data, usedNe=None, usedNp=None, debug=False)
```
（eigenvector.py:455-461）

返回 `overlap[Lt, Ne, Np, Nc]`，其中 `M[t,e,p,c] = eigenvector_data[t,e,z_p,y_p,x_p,c]`（纯高级索引抽取，无 gauge link）；`usedNe/usedNp` 为 `None` 时取全体。仓库内无调用点。

### 2.6 `NoisevectorGenerator`（noisevector.py:11）

把 eigenvector 低模空间按 dilution 方案随机线性组合成 noise vector，可选追加与低模正交的 high-mode 随机向量。

```python
def __init__(
    self,
    eigenvector: Eigenvector,
    dilution: Tuple,
    highmode: int = 0,
    full_noisev: bool = False,
)
```
（noisevector.py:12-17）

| 参数/属性 | 含义 |
|---|---|
| `dilution = (eigv_list, noisev_list)` | `eigv_list` 为每个 block 消耗的本征模数列表；`noisev_list` 为 int（各 block 同值）或列表（noisevector.py:19-23） |
| 构造校验 | `len(noisev_list) == len(eigv_list)` 与逐块 `noisev_list <= eigv_list`（assert，:27-28） |
| `transfer_matrix` | 构造中赋 `None`（:21），之后从未被写回/读取（死属性；`calc` 内的同名变量均为局部变量） |

方法：

- `load(key)`（:32-33）：`self._eigenvector_data = self.eigenvector.load(key)`。
- `calc(seed=None)`（:35）：`backend.random.seed(seed)` 后按 block 生成噪声向量，返回形状 `(Lt, Nn, Lz, Ly, Lx, Nc)`、dtype `<c16` 的数组，其中 `Nn = sum(eigv_list) + highmode`（`full_noisev=True`）或 `sum(noisev_list) + highmode`（:40-44）。分块算法见 §4.3。

### 2.7 `generate_sparsened_points`（sparsened_point.py:9）

```python
def generate_sparsened_points(
    latt_size: List[int], num_points: int, seed: Optional[int] = None
) -> np.ndarray
```

- 参数：`latt_size = [Lx, Ly, Lz, Lt]`（必须恰好 4 个元素，否则 `ValueError`，:65-66）；`num_points` 须满足 `0 < num_points ≤ Lx·Ly·Lz`（:69-77）；`seed` 非 `None` 时 `np.random.seed(seed)`（:80-81，使用全局 NumPy legacy 随机状态，可复现但会污染全局状态）。
- 返回：`coords (num_points, Lt, 3)`，dtype `int32`；`coords[p, t, :] = [x, y, z]`；每个时间片内部点两两不同（while 循环 + `set` 去重，:85 起），不同时间片之间无约束。

### 2.8 `PerambulatorGenerator`（perambulator.py:43）与 `sampled_point_local_indices`

以 LapH/distillation 的 eigenvector（和可选点源/点汇采样表）为源，用 PyQUDA/QUDA 的 Dirac 求逆生成 perambulator 族产物 `VSV`（本征矢→本征矢）、`PSV`（本征矢→采样点）、`PSP`（点→点）。

```python
def __init__(
    self,
    latt_size: List[int],
    gauge_field: GaugeField,
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    nu: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    t_boundary: Literal[1, -1] = 1,
    multigrid: List[List[int]] = None,
    contract_prec: str = "<c16",
    eigenvector_src: Eigenvector = None,
    usedNe_src: int = None,
    eigenvector_snk: Eigenvector = None,
    usedNe_snk: int = None,
    point_src: PointSource = None,
    usedNp_src: int = None,
    point_snk: PointSource = None,
    usedNp_snk: int = None,
    MRHS: bool = False,
    use_vectorized: bool = True,
    same_eigenvector: bool = True,
) -> None
```
（perambulator.py:91-115）

| 参数 | 含义 |
|---|---|
| `mass/tol/maxiter/xi_0/nu/clover_coeff_t/clover_coeff_r/multigrid` | 传给 `core.getDirac`（:178-181）构造 anisotropic clover Dirac 算子；`xi_0/nu` 同时作为 `LatticeInfo` 的 anisotropy（:120-122） |
| `t_boundary` | 时间方向边界条件（±1） |
| `contract_prec` | 全部缓冲区 dtype，默认 `<c16`（little-endian complex128） |
| `eigenvector_src/snk`、`usedNe_src/snk` | 源/汇本征矢句柄与截断；缺省取 `eigenvector.Ne`；显式值与 `Ne` 不一致时源端 `print` 警告不报错（:138-147），汇端无警告（:148-152） |
| `point_src/snk`、`usedNp_src/snk` | 点源/点汇句柄与截断，缺省取 `Np`（:165-171） |
| `MRHS` | 多右端项开关：`True` 时用 `MultiLatticeFermion` 一次装 Ns 个右端（每 spin 一个）求逆 |
| `use_vectorized` | `calc(t_src)` 分派开关：`True` → `calc_new`，`False` → `calc_old`（:754-759） |
| `same_eigenvector` | `True` 时强制 `eigenvector_snk = eigenvector_src`（:133-135） |

注意两点代码现状：(1) 构造中 PyQUDA 检查被注释掉（:117-118），与两个变体模块的 `if not check_QUDA(): raise ImportError` 不同；(2) 类 docstring（:40-89）写的是 `eigenvector`/`usedNe`，与实际签名的 `eigenvector_src`/`usedNe_src`/`usedNe_snk` 存在漂移，以签名为准（`test/test_perambulator_mpi.py:33` 仍用旧关键字 `eigenvector=`，见 §5）。

模块级函数：

```python
def sampled_point_local_indices(point_data, point_count, latt_info)
```
（perambulator.py:11）

输入 `point_data` 形状 `[Np, Lt_global, 3]`（`point_data[p, t] = (x, y, z)`）、使用点数 `point_count`、本 rank 的 `core.LatticeInfo`（`size=(Lx,Ly,Lz,Lt)` 与 `grid_coord`）。输出 6 元组 `(point_indices, local_times, parity, z_local, y_local, x_local // 2)`：用**全格距**掩码 `gx*Lx <= x_global < (gx+1)*Lx` 筛选本 rank 拥有的点（:38-42），换算本地坐标后取 parity = `(t_local + x_local + y_local + z_local) % 2`（:53），`x_local // 2` 为 checkerboard 压缩后的 x 索引。是 MPI 多卡下"每 rank 只处理自己拥有的采样点"的几何工具，输出直接作为高级索引用于 PSV/PSP 的批量抽取。

| 方法 | 位置 | 说明 |
|---|---|---|
| `load(key)` | :219 | ① `gauge_field_smear = io.readQIOGauge(...)`（此时**尚未 stout**）；② 切到 numpy 后端，逐 (e, t) 把本 rank 子格的 eigenvector 拷到 host 并存**共轭**（`eigenvector_src_data_dagger`，:231-254；`same_eigenvector=True` 时汇端复用源端数组）；③ 读 `point_source_data/point_sink_data` 并用 §2.8 函数填充 `self._point_sink_indices`（:284-288）；④ 设备端存 even-odd 化的 V†：`self._eigenvector_data_dagger = backend.asarray(core.cb2(...))`（:294-296） |
| `stout_smear(nstep, rho, dir_ignore=3)` | :318 | numpy 下 `raise NotImplementedError`（:320-323）；cupy 下调用 `gauge_field_smear.stoutSmear(nstep, rho, dir_ignore)`（:308-316）；未 `load()` 先 smear 抛 `ValueError`（:311-314） |
| `calc_old(t_src)` | :328 | 逐 spin 串行参考实现，见 §4.4；返回 `(VSV, PSV)`（持久缓冲区本身，非拷贝，:471） |
| `calc_new(t_src, products=("VSV", "PSV"))` | :473 | 向量化实现，见 §4.4；`products` 白名单校验（转大写 frozenset，超出 `{"VSV","PSV"}` 或为空 → `ValueError`，:524-531）；返回 `(VSV or None, PSV or None)` |
| `calc_eigen_sources(t_src, products=("VSV","PSV"))` | :659-661 | 一行薄封装，直接 `return self.calc_new(t_src, products=products)` |
| `calc_point_sources(t_src)` | :663 | PSP 生成，见 §4.4；返回 `_PSP` 缓冲区（:751） |
| `calc(t_src)` | :754 | `use_vectorized=True` → `calc_new`，否则 `calc_old` |

性能指标 `self.last_metrics`（:176-177）由 `calc_new`/`calc_point_sources` 填充：`product / source_time / eigenvectors / inversion_seconds / vsv_seconds / psv_seconds / elapsed_seconds / device_used_bytes / peak_device_used_bytes`。

### 2.9 `DensityPerambulatorGenerator`（density_perambulator.py:12）

对两个源时刻 `ti`、`tf` 各自的本征矢源求 Dirac 解（限制到共同汇时刻 `tau`），插入 gamma 结构 `Gamma` 与动量相位 `e^{ip.x}`，收缩出逐空间点的广义 perambulator `VSSV`。

```python
def __init__(
    self,
    latt_size: List[int],
    gauge_field: GaugeField,
    eigenvector: Eigenvector,
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    nu: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    t_boundary: Literal[1, -1] = 1,
    multigrid: List[List[int]] = None,
    gamma_list: List[int] = [i for i in range(Ns * Ns)],   # 默认全部 16 个 gamma
    momentum_list: List[Tuple[int]] = [(0, 0, 0)],
) -> None
```
（density_perambulator.py:13-29）

- 构造即硬检查 `if not check_QUDA(): raise ImportError`（:30-31）并 `assert backend == cupy`（:37）。`gamma_list` 元素为 gamma bitmask（`lattice/insertion/gamma.py:95-105` 的 `gamma(n)` 约定，`n ∈ [0,15]`）。
- `load(key)`（:72-75）：`dirac.loadGauge(io.readQIOGauge(...))` + 读 eigenvector；**不做任何 smearing**（类体第一行 TODO 注释 "Add parameters to do smearing before the inversion"，:12）。
- `calc(ti: int, tf: int, tau: int)`（:78）：流程见 §4.5，返回 `VSSV.transpose(2, 3, 4, 5, 6, 7, 8, 0, 1)`（:176），输出形状 `(N_gamma, N_mom, Lz, Ly, Lx, Ns, Ns, Ne_f, Ne_i)`。

### 2.10 `GeneralizedPerambulatorGenerator`（generalized_perambulator.py:13）

与 density 版相同的物理收缩（双源 + gamma5 共轭 + `Gamma` + 动量相位），但解保留全部 `Lt` 个汇时刻，输出时间分辨的 `VSSV`。

- `__init__` 签名与 density 版**完全相同**（:14-30），同样有 `check_QUDA` 硬检查（:31-32）与 cupy assert（:38）；`load()` 同构（:71-74）；同样不支持 smearing（:13 的 TODO 注释）。
- `calc(ti: int, tf: int)`（:77）：**没有 `tau` 参数**；流程见 §4.5，返回 `VSSV.transpose(2, 3, 4, 5, 6, 0, 1)`（:184），输出形状 `(N_gamma, N_mom, Lt, Ns, Ns, Ne_f, Ne_i)`。

### 2.11 `stout_smear.cu`（CUDA kernel）

单步 stout link smearing 的 GPU 实现，与 Python `_stout_smear_ndarray` 同一公式：

- `Matrix<T,Nc>` 3×3 复矩阵设备类（stout_smear.cu:13-120）：加/减/乘、trace、adjoint、`antihermitian`（:144-156）。
- `__global__ void stout_smear<T>(complex<T> *U_out, const complex<T> *U_in, const T rho, int Lx, Ly, Lz, Lt)`（stout_smear.cu:159 起）。索引宏 `get_x` 按 x-fastest 折算线性地址（:3-5），对应 Python 侧 `(mu, t, z, y, x, color, color)` 行主布局。
- 每线程处理一个格点 × 一个空间方向 μ；对 ν≠μ 求前向/后向两个 staple 之和（:174-197），`Q = antiherm(rho · Σstaple · U_μᴴ)`（:199），用不变量 `c0 = tr(Q³).real/3`、`c1 = tr(Q²).real/2` 解析计算 `exp(iQ) = f0·I + f1·Q + f2·Q²`（:201-243），输出 `U_out = e^{iQ}·U`。

Python 侧只实例化 `stout_smear<double>`（`name_expressions=("stout_smear<double>",)`，elemental.py:76-79），`rho` 以 double 传入；`nstep` 由 Python 循环实现（每次 kernel 前 `U_in = U.copy()`）。启动配置 `grid = (Lx·Ly·Lz, Nd-1, 1)`、`block = (Lt, 1, 1)`（elemental.py:346-348 等）。

---

## 3. 数据结构与数组形状约定

### 3.1 贯穿性约定

| 对象 | 布局 | 依据 |
|---|---|---|
| gauge field（generator 内部 `self._U`） | `(Nd, Lt, Lz, Ly, Lx, Nc, Nc)` 经 `transpose(4,0,1,2,3,5,6)` 而来，随后多只取空间方向 `[:Nd-1]` 即 `(3, Lt, Lz, Ly, Lx, Nc, Nc)` | elemental.py:175、displacement_elemental.py:74、eigenvector.py:58 |
| eigenvector | `(Lt, Ne, Lz, Ly, Lx, Nc)`，轴序 `[t_global, eigen, z, y, x, color]` | elemental.py:677-678（`V[e] = eigenvector[t, e]`）、test/test_perambulator.py:23-24 |
| point | `(Np, Lt, 3)`，分量顺序 `[x, y, z]` | sparsened_point.py 返回约定；elemental.py:1794-1799 中 `point_shifted[..., 2/1/0]` 分别作为 z/y/x 使用 |
| 动量相位 | `MomentumPhase(latt_size).get(momentum)` → `(Lz, Ly, Lx)` 的 `exp(i(px·x·2π/Lx + py·y·2π/Ly + pz·z·2π/Lz))`，带 per-momentum 缓存 | lattice/insertion/phase.py:10-21, 55-61 |
| even-odd 相位 | `MomentumPhase.get_cb2(np)` → `(2, Lt, Lz, Ly, Lx//2)`，按 `(it+iz+iy)%2` 拆 parity | lattice/insertion/phase.py:20-53 |
| `gamma(n)` | bitmask 构造 Dirac 矩阵：`n & 0b0001/0b0010/0b0100/0b1000` → gamma_0/1/2/3 连乘；`gamma(15)` 即 gamma5 | lattice/insertion/gamma.py:95-110 |
| 张量收缩 | 统一 `opt_einsum.contract` | 各模块 import |

### 3.2 generator 缓冲区/输出形状

| 类 / 缓冲区 | 形状 | dtype | 说明 |
|---|---|---|---|
| `ElementalGenerator._V` | `(usedNe, Lz, Ly, Lx, Nc)` | `<c8` | calc_deriv 模式的本征矢工作数组（elemental.py:103） |
| `ElementalGenerator._VPV` | `(num_derivative, num_momentum, usedNe, usedNe)` | `<c16` | `calc_deriv` 输出/复用缓冲（elemental.py:104-108）；调用侧常 `transpose(1,2,0,3,4)` 存为 `(num_deriv, num_mom, Lt, Ne, Ne)` |
| `ElementalGenerator._disp_buffer` | `(num_disp, num_momentum, usedNe, usedNe)` | `<c16` | `calc_disp` 输出/复用缓冲（elemental.py:112-114） |
| `ElementalGenerator.stocastic_coeff` | `(Ne, Ne)` | backend 默认 | blending 系数矩阵（:141） |
| `CurrentElementalGenerator` 各输出 | 见 §2.2 表 | `<c16` | v2v/v2p/p2v/p2p |
| `DisplacementElementalGenerator._VPV` | `(distance+1, num_momentum, Ne, Ne)` | `<c16` | 轴 0 为位移阶数（displacement_elemental.py:38） |
| `DisplacementElementalGenerator.Vd` | `(2*(Nd-1), Ne, Lz, Ly, Lx, Nc)` | `<c16` | 6 个方向槽的递推工作数组，跨 `dist` 调用**有意保留**（:51） |
| `EigenvectorGenerator.calc` 输出 | `(evecs[Ne, Lz, Ly, Lx, Nc], evals)` | — | 逐时间片 |
| `NoisevectorGenerator.calc` 输出 | `(Lt, Nn, Lz, Ly, Lx, Nc)` | `<c16` | `Nn = sum(eigv/noisev_list) + highmode`（noisevector.py:40-44） |
| `generate_sparsened_points` 输出 | `(num_points, Lt, 3)` | `int32` | `coords[p,t,:] = [x,y,z]` |
| `PerambulatorGenerator._SV` | `(2, Lt, Lz, Ly, Lx//2, Ns, Ns, Nc)` | `contract_prec` | Dirac 解 S·V，even-odd 布局，轴 `[parity, t, z, y, x/2, s_snk, s_src, c]`（perambulator.py:185-187） |
| `PerambulatorGenerator._VSV` | `(Lt, Ns, Ns, Ne_snk, Ne_src)` | `contract_prec` | perambulator tau（:188-192）；盘上约定 `[t_src, t_rel, s_snk, s_src, e_snk, e_src]`（test/test_perambulator.py:49-56 收集方式确立） |
| `PerambulatorGenerator._PSV` | `(Lt, Ns, Ns, Np_snk, Nc, Ne_src)` | `contract_prec` | 本征矢→采样点（:193-198） |
| `PerambulatorGenerator._PSP` | `(Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc)` | `contract_prec` | 点→点，轴 `[t_local, s_snk, s_src, p_snk, c_snk, p_src, c_src]`（:201-207） |
| `PerambulatorGenerator._SP` | 恒 `None` | — | 死代码：只有赋 `None` 的分支（:199-200, 215-217） |
| `PerambulatorGenerator._eigenvector_data_dagger` | cb2 后 `[Ne, parity, t, z, y, x//2, Nc]` | — | `core.cb2` 把 parity 轴插在原 axis 1 位置（:294-296 及 :400-401、:427-430 的下标用法） |
| `DensityPerambulatorGenerator._SV_i/_SV_f` | `(Ne, 2, Lz, Ly, Lx//2, Ns, Ns, Nc)` | `<c16` | ti/tf 源的解，只保留 tau 切片（:64-65） |
| `DensityPerambulatorGenerator._VSSV`（host） | `(Ne, Ne, N_gamma, N_mom, Lz, Ly, Lx, Ns, Ns)` | `<c16` | 重组回词典序的最终缓冲，轴 `[e_f, e_i, Gamma, mom, z, y, x, s_f, s_i]`（:67） |
| `GeneralizedPerambulatorGenerator._SV_i/_SV_f` | `(2, Lt, Lz, Ly, Lx//2, Ns, Ns, Nc)` | `<c16` | **单个 eigen** 的全时刻解，设备端复用（:60-61） |
| `GeneralizedPerambulatorGenerator._h_SV_i/_h_SV_f` | `(Ne, 2, Lt, Lz, Ly, Lx//2, Ns, Ns, Nc)`（pinned） | `<c16` | 每 eigen 的 host 副本，双 CUDA 流重叠传输（:62-63） |
| `GeneralizedPerambulatorGenerator._VSSV` | `(Ne, Ne, N_gamma, N_mom, Lt, Ns, Ns)`（pinned） | `<c16` | 轴 `[e_f, e_i, Gamma, mom, t, s_f, s_i]`（:65） |

### 3.3 PSV/VSP/PSP 盘上容器（`lattice/preset.py`，供上下文参考）

| 容器 | 盘上形状 | 语义 |
|---|---|---|
| `PropagatorPSV` | `[Lt(t_src), Lt(t_rel), Ns_snk, Ns_src, Np, Nc, Ne]` | `PSV[t_src, t_rel, s', s, p, c, e] = <eta_{x_p}, c | S | xi_e>`（preset.py:62-76） |
| `PropagatorVSP` | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` | 与 PSV 方向相反（preset.py:89） |
| `PropagatorPSP` | `[Lt, Lt, Ns, Ns, Np_snk, Nc_snk, Np_src, Nc_src]` | preset.py:118 |

VSP 不由 generator 单独生成：`calc_new` 的 docstring 明言 "VSP is reconstructed from PSV at contraction time"（perambulator.py:502 附近）；`lattice/propagators.py:977-998` 的 `psv_dagger` 缓存即 PSV→VSP 的顺序重排。

---

## 4. 算法与流程

### 4.1 `ElementalGenerator`

#### 4.1.1 `calc_deriv(t)`（elemental.py:668-720）

1. `V[e] = eigenvector[t, e]` 装入 `self._V`（:677-678）。
2. 对 `derivative_list` 中第 `idx` 个 derivative（长度 L 的方向序列），对 `pick_nabla in range(2**L)` 做子集枚举：按二进制位把 L 个方向分给 `pick_right`/`pick_left`，`coeff = (-1) ** len(pick_right)`（:686-697）。文件中保留一段被注释掉的等价二项式实现（`coeff = (-1)**k · C(L,k)` 左右各取 k 个连续方向，:680-685），佐证 2^L 子集枚举与二项式（莱布尼茨）展开的对应关系。
3. `right = self._nD(V, U[:, t], pick_right)`、`left = self._nD(V, U[:, t], pick_left[::-1])`（:698-699）；left 的方向序列做了反转（`[::-1]`）。
4. 对每个动量：`VPV[idx, mom] += coeff · Σ_x phase(x) · conj(left) · right`（einsum `"zyx,ezyxc,fzyxc->ef"`，:702-708）；blending 时额外乘 `stocastic_coeff[:usedNe, :usedNe]`（:705-719）。

`_nD(V, U, deriv)`（:646-655）对 deriv 中每个方向 d 依次做协变位移：前向 `UVf[x] = U_d(x)·V(x+d̂)`（`roll(V, -1, 3-d)`）；后向 `V(x) ← roll(U_d(x)ᴴ·V(x), +1, 3-d)`。即沿方向序列的复合协变位移算子。

#### 4.1.2 `calc_disp(t)`（elemental.py:727-841）

1. 对每个 `disp_idx`：`gauge_link = GaugeLink(disp_idx)`，取 `gauge_list`（0..2 前向、3..5 后向的链接方向序列）与净位移 `disp`（每个 <3 的方向 +1、≥3 的方向 −1）（:737-741）。
2. `shift_V[x] = V(x + disp)`：对 V 沿三个轴分别 `roll(-disp[axis], 3-axis)`（:743-750）。
3. `gauge_link_product = self._gauge_links_product(gauge_list, t=t)`，形状 `(Lz, Ly, Lx, Nc, Nc)`（:759-760）。
4. 空路径（`None`，即恒等）：`result[idx, mom] = Σ_x phase(x)·conj(V)·V`（:776-784）。
5. 否则对每个动量：`disp_phase = exp(Σ_i momentum[i] · (disp[i]/2) · 2πi/L_i)`（:788-799），然后
   `result[idx, mom] = disp_phase · Σ_x phase(x) · conj(V(x)) · G(x) · V(x+disp)`（einsum `"zyx,ezyxa,zyxac,fzyxc->ef"`，:800-806）。因子 `disp[i]/2` 是位移算子的半移（half-shift）约定，代码中显式出现。

`_gauge_links_product(gauge_list, t)`（:375-457）按顺序累积：前向 d<3 时 `product = product @ U_d`（einsum `"tzyxab,tzyxbc->tzyxac"`）并维护 `self._U_shift = roll(self._U, -1, 4-d)`；后向 d≥3 时先 `self._U_shift = roll(self._U, +1, 4-(d-3))`，再左乘 `_U_shift[d-3]ᴴ`。注意：前向分支的累积用的是 `self._U[d]`（未滚动），`_U_shift` 只在后向分支被使用（:386-398）——对多链接非共线路径是否与逐步 `_disp`（配合 roll 后的 V）完全一致，仓库内以 `test/scripts/test_disp_vs_product.py` 的对比脚本作为作者验证手段，本文档将其记为未在本地复验的现状（见 §5.2）。

#### 4.1.3 `re_combine` 系列

见 §2.3 表：dense 用 `tensordot`，sparse 用 CSR 稀疏乘法，`re_combine_auto` 首次调用交叉验证（`rtol=1e-6, atol=1e-10`，失败抛 `ValueError`）后按计时选路。仓库内无调用点。

#### 4.1.4 blending（随机稀释混合）系数

`is_blending and dilution is not None` 时构造块对角系数矩阵 `stocastic_coeff (Ne, Ne)`（elemental.py:134-167）：

- 校验（assert）：`len(usedNe_list) == len(totNe_list)`、`(usedNe_list <= totNe_list).all()`、`sum(usedNe_list) == Ne`（:138-140）。
- 非对角块（nv1≠nv2）：`coeff = totNe1·totNe2 / (usedNe1·usedNe2)`（:151-153）。
- 对角块：非对角元 `coeff2 = (totNe1/usedNe1)·((totNe1-1)/(usedNe1-1))`，对角元 `coeff1 = totNe1/usedNe1`（:154-163）。
- `is_blending=True` 但无 dilution 时抛 `ValueError("Dilution tuple is not defined.")`（:170）。
- 该系数仅在 `calc_deriv` 的收缩中使用（:705-719）；`calc_disp` 路径不使用。

### 4.2 `DisplacementElementalGenerator`（displacement_elemental.py:53-97）

`_D(V, U, dist)`：

- `dist == 0`：返回 V 本身（:55-56）。
- `dist == 1`：对每个空间方向 d：`Vd[d] = U_d(x)·V(x+d̂)`（前向）、`Vd[-d-1] = roll(U_d(x)ᴴ·V(x), +1, 3-d)`（后向）；返回 `Vd.mean(0)`，即六方向平均位移（:57-69）。
- `dist ≥ 2`：在**上一次调用残留的 `self.Vd`** 上继续迭代（`Vd[d] = U_d·roll(Vd[d])`、`Vd[-d-1] = roll(U_dᴴ·Vd[-d-1])`），返回 `Vd.mean(0)`。因此 `_D` 依赖跨调用状态，必须由 `calc` 按 `dist = 0, 1, 2, ...` 顺序调用（`calc` 中正是如此，:88-89）；单独以 `dist ≥ 2` 调用会得到错误结果。

`calc(t)`（:78-97）：`V[e] = eigenvector[t,e]`；对每个 `dist` 清零 `VPV[dist]` 后取 `right = _D(V, U[:,t], dist)`、`left = V`，对每个动量**同时累加两项**：

```python
VPV[dist, imom] += contract("zyx,ezyxc,fzyxc->ef", momentum_phase.get(mom), left.conj(), right)
VPV[dist, imom] += contract("zyx,ezyxc,fzyxc->ef", momentum_phase.get(mom), right.conj(), left)
```
（displacement_elemental.py:92-93）

### 4.3 `NoisevectorGenerator.calc(seed)`（noisevector.py:35-107）

1. `backend.random.seed(seed)`（:37）。
2. 输出缓冲按 §2.6 分配（:39-45）。
3. 逐 block（`noisev_start`/`eigv_start` 连续推进，:47-48）：
   - `noisev_lenth == eigv_lenth`：直接复制该 block 的 eigenvector（恒等稀释，:52-56）。
   - `noisev_lenth == 1`：对每 (t, 模 i) 生成 `uniform(0, 2π)` 随机相位，`noisev[:, n] = Σ_i e^{iφ_{t,i}} V_i(t)`（Z_n 相位型随机组合，:58-65）。
   - 其他：对每个 t 生成复高斯随机矩阵，QR 分解取前 `noisev_lenth` 列正交基 `Q`，`noisev[t, block] = V(t) @ Q`（逐时间片独立的酉随机投影，:67-80）。`full_noisev=True` 时把 `noisev_lenth` 强制改回 `eigv_lenth`（:72），即输出与输入模数相同的完全随机旋转（仅对走此分支的 block 生效，见 §5.2）。
4. `highmode > 0` 时（:85-102）：反复生成复高斯随机场 `(Lt, Lz, Ly, Lx, Nc)`，对每 t 先投影掉与已用本征矢 `eigenvector[t, :eigv_start]` 的分量、再投影掉与已生成 highmode 噪声的分量（einsum `"zyxa,izyxa,iuvwb->uvwb"` 的逐时间片投影子，:89-97），范数 > 1e-10 才接收并归一化（:98-101），直到补满 `highmode` 个。
5. 返回 `noisev`。

### 4.4 `PerambulatorGenerator`

#### `calc_old(t_src)`（perambulator.py:328-471）

1. gauge 有更新则 `dirac.loadGauge(self.gauge_field_smear)`（:345-348）。
2. 对每个源本征矢 `eigen`：仅当本 rank 拥有 `t_src`（`gt*Lt <= t_src < (gt+1)*Lt`，:388）时，把 `eigenvector_dagger[eigen, :, t_src % Lt, ...].conj()` 写入 `LatticeFermion` 的对应 `spin` 槽（二次共轭把 load 时存的 V† 还原成 V，:413-414）；单 RHS 路径逐 spin 调 `dirac.invert(V)`，MRHS 路径用 `MultiLatticeFermion`（`dirac.invertMultiSrc`）一次装 Ns 个右端求逆（:370-384）；解存入 `_SV`，轴语义 `SV[parity, t, z, y, x/2, s_snk, s_src, c_src]`（:416-421）。
3. VSV 收缩（:427-431）：`VSV[:, :, :, :Ne_snk, eigen] = contract("ketzyxa,etzyxija->tijk", eigenvector_sink_dagger, SV_array, optimize=True)`——对 parity/t/z/y/x/c 求和得 `VSV[t, i=s_snk, j=s_src, k=Ne_snk]`，写入最后一轴（源本征矢）。汇端因子是 load 时存好的 V†，即 `tau_{ji}(t_snk, t_src) = Σ_x V_j(t_snk, x)† · S⁻¹(t_snk, x; t_src) · V_i(t_src, x)`。
4. PSV 抽取（:433-462）：**存在结构性缺陷**——`for t_snk in range(Lt)` 的循环体只有 `continue`（:435-437），循环结束后 `t_snk == Lt-1`，后续 `t_index` 恒取最后一片；且点归属判断用**半格距**（`gx*(Lx//2) <= x < (gx+1)*(Lx//2)`，:447）、parity 用全局坐标（:455-458），均与 §2.8 的 helper（全格距 + 本地坐标）不一致。详见 §5.2。
5. 返回 `(VSV, PSV)`（:471，持久缓冲区本身）。

#### `calc_new(t_src, products)`（perambulator.py:473-657）

- 入口校验 `products` 白名单（:524-531）；请求 `VSV` 但无汇本征矢、请求 `PSV` 但无点汇时 `ValueError`。
- 求逆部分与 calc_old 同构（:550-590，含 MRHS 分支）。
- VSV 收缩同 :604-608（仅当 `"VSV" in products`）。
- PSV 向量化抽取（:612-625）：完全用 §2.8 预计算的本地索引一次完成，无逐点循环：
  ```python
  PSV[local_times, :, :, point_indices, :, eigen] = SV_array[parity, local_times, z_local, y_local, x_half]
  ```
- 性能指标写入 `self.last_metrics`（:641-653）。
- 返回 `(VSV if "VSV" in products else None, PSV if "PSV" in products else None)`（:655-657）。

#### `calc_point_sources(t_src)`（perambulator.py:663-752）

1. 前置校验：需要 `point_src` 与 `point_snk`（:666-668）、`_PSP` 缓冲已分配（:669-673）；gauge 更新则 `loadGauge`。
2. `PSP.fill(0)`（:683）；对每个点源 `p ∈ [0, Np_src)`：取坐标 `point_source_data[p, t_src]` 拼 `source_position = [x, y, z, t_src]`（:691-694）；把 `Ns*Nc = 12` 个 spin-color 点源按 `rhs_count = 12 if MRHS else 1` 分批（:696），`rhs[batch_idx] = source.source(latt_info, "point", source_position, spin_color // Nc, spin_color % Nc)`（:702-705），`solutions = self.dirac.invertMultiSrcRestart(rhs, 0)`（:710）。
3. 在**采样点汇**处批量抽取（:722-731）：
   ```python
   PSP[local_times, :, spin_color // Nc, point_indices, :, point_src_idx, spin_color % Nc] = values
   ```
   其中 `values = solutions[batch_idx].data[parity, local_times, z_local, y_local, x_half]`。
4. 只有本 rank 拥有的采样汇会被写入（`_point_sink_indices` 已按 rank 过滤），其余元素保持 0——多卡结果需调用方 gather/reduce（`test/test_perambulator_mpi.py:64` 用 `gatherLattice(..., reduce_op="sum")`）。
5. 返回 `_PSP` 并更新 `last_metrics`（:738-751）。

### 4.5 `DensityPerambulatorGenerator.calc(ti, tf, tau)`（density_perambulator.py:78-176）与 `GeneralizedPerambulatorGenerator.calc(ti, tf)`（generalized_perambulator.py:77-184）

两者共享同一收缩结构，逐项对比如下：

| 步骤 | density（固定 tau） | generalized（全时刻） |
|---|---|---|
| 源准备 | `ti != self._t` 时把 `eigenvector[ti, e]` 抄入 host `data_lexico` 并手工 even-odd 重排到 `data_cb2`，parity 判据 `eo = (ti + z + y) % 2`（:84-112）。`ti/tf` 为**全局**时间，写入 `V` 前换算为本地时间并加归属判断（:94-103） | 同构（:93-123；`eo = (ti+z+y)%2`，:107/:116），同样做全局→本地换算（:95-104），缓存键为 `_ti/_tf` |
| 反演 | 逐 eigen、逐源 spin 反演，**只取 tau 切片**存入 `_SV_i/_SV_f`（:135-146）。源写入按 rank 归属门控，但 `dirac.invert()` 在**所有 rank** 执行（收缩是逐空间点、不做空间求和，见下） | 逐 eigen、逐 spin 反演**全时刻解**入 `_SV_i/_SV_f`，并异步 `.get(stream, out=h_SV[eigen])` 到 pinned host 副本（:139-156） |
| 反演 | 逐 eigen、逐源 spin 反演，**只取 tau 切片**存入 `_SV_i/_SV_f`（:135-146） | 逐 eigen、逐 spin 反演**全时刻解**入 `_SV_i/_SV_f`，并异步 `.get(stream, out=h_SV[eigen])` 到 pinned host 副本（:139-156） |
| gamma5 共轭 | 逐 eigen 行执行一次：`SV_f[eigen] = contract("ii,ezyxjic,jj->ezyxijc", gamma(15), SV_f[eigen].conj(), gamma(15))`（:148-153）——位于 `if tf != self._tf:` 块内，只作用于刚反演的那一行。`T(S) = gamma5·conj(S)·gamma5` 是对合（`T∘T = id`），故每行必须且只能作用一次 | 同一变换 `contract("ii,etzyxjic,jj->etzyxijc", ...)`，同样位于 `if tf != self._tf:` 块内（:155），只随反演执行一次（设备缓冲为单 eigen 形状，随后 `.get()` 快照） |
| 收缩 | 逐 `(eigen_f, eigen_i, gamma_idx, momentum_idx)`：`contract("ezyx,ezyxijc,jk,ezyxklc->ezyxil", momentum_phase.get_cb2(momentum)[:, tau], SV_f[eigen_f], gamma(gamma_i), SV_i[eigen_i]).get(stream, out=VSSV_cb2[...])`——动量相位只取 `[:, tau]` 切片（:154-164） | 逐 `(eigen_i, eigen_f)`：`SV_i.set(h_SV_i[...])`、`SV_f.set(h_SV_f[...])` 后 `contract("etzyx,etzyxijc,gjk,etzyxklc->gtil", momentum_phase.get_cb2(momentum), SV_f, gamma_current, SV_i, optimize=True)`——动量相位用全时刻；`gamma_current = backend.asarray([gamma(i) for i in gamma_list])` 在 eigen_f 循环内重复构建（:166-182） |
| 重组/返回 | even-odd → 词典序重组到 host `_VSSV[e_f, e_i, ...]`（按 `eo = (tau+z+y)%2` 摆回奇偶 x，:166-175），返回 transpose 后 `(N_gamma, N_mom, Lz, Ly, Lx, Ns, Ns, Ne_f, Ne_i)`（:176） | 返回 transpose 后 `(N_gamma, N_mom, Lt, Ns, Ns, Ne_f, Ne_i)`（:184） |

两者的矩阵元形式（由 einsum 下标直接读出）：

```
VSSV^{(f,i)}_{Gamma,p}(x) = Σ_内部指标 e^{ip·x} · [gamma5·S_f†·gamma5](x)_{ij} · Gamma_{jk} · S_i(x)_{kl}
```

说明：`SV_f[eigen]` 的轴序为 `[parity, z, y, x//2, s_snk, s_src, c]`，einsum 下标 `ezyxjic`
把 `j` 绑到 axis 4（`s_src`）、`i` 绑到 axis 5（`s_snk`），因此 `contract(...)->...ijc` 顺带完成了
自旋转置，实现的就是 `γ5·S_f†·γ5`（而非 `γ5·S_f*·γ5`）。`T` 的对合性来自 `γ5` 实对角且 `γ5² = I`。

density 版输出逐空间点（固定汇时刻 tau），generalized 版输出逐汇时刻。

#### 全局/本地时间约定（已修）

两个类都把**全局** `latt_size` 传给 `core.LatticeInfo`，而 PyQUDA 的 `LatticeInfo.size` 经
`getSublatticeSize` 解析为**本地子格**（`size = GL // G`）。因此 `LatticeFermion(...).data` 的
时间轴长度是 `Lt_local = GLt / Gt`，而代码中的 `ti`/`tf` 是**全局**源时刻。修复前直接把全局
`ti`/`tf` 当作本地轴下标使用（无换算、无归属判断）：

- 单卡（`Gt = 1, gt = 0`）：全局 == 本地，恰好恒等，故此前未被发现。
- 多卡：`gt >= 1` 时得到**负下标或越界**，且 `gt` 正确命中的 rank 也无法区分。

现按 `perambulator.py` 的模式处理：`gt = latt_info.grid_coord[3]`、`Lt_local = latt_info.size[3]`，
计算 `owns = gt*Lt_local <= t < (gt+1)*Lt_local` 与 `t_local = t - gt*Lt_local`。
**只对源的写入做归属门控**（`if owns_ti: V[:, ti_local, ...] = ...`）；
因为收缩 `...->ezyxil` 保留 `z,y,x`、是**逐空间点**而非空间求和，所以 `dirac.invert()` 必须在
每个 rank 上照常执行。

> 注意仍存在的限制：这两个模块的 `MomentumPhase(latt_size)`（:63 / :64）也拿的是全局元组，
> 且 `_SV_i/_SV_f`、`VSSV_cb2`、`VSSV` 均按全局 `Lx,Ly,Lz,Lt` 分配，而 `LatticeFermion` 是本地形状
> ——多卡下 `_V.data.reshape(2, Lt, Lz, Ly, Lx//2, Ns, Nc)` 会因元素数不符直接 `ValueError`。
> 故这两个模块目前**实际只能在单卡（`grid=[1,1,1,1]`）下运行**；全局/本地尺寸的整体重构超出本次修复
> 范围，此处仅记录现状。

### 4.6 Stout smearing（数学内容，三处 Python + CUDA 一致）

每步对每个空间方向 μ 构造无迹反厄米矩阵 `Q_μ(x) = antiherm[ρ·Σ_μ(x)·U_μ†(x)]`（`Σ` 为 3D staples 之和），`U_μ(x) ← e^{iQ_μ(x)}·U_μ(x)`；`e^{iQ}` 用 c0/c1 解析公式计算，`|w| > 0.05` 时 `sinc` 换成精确 `sin(w)/w`，`c0 < 0` 时利用共轭对称性修正 f 系数符号（`f0_imag, f1_real, f2_imag` 反号，elemental.py:324-327）。

四个类（`ElementalGenerator`、`CurrentElementalGenerator`、`DisplacementElementalGenerator`、`EigenvectorGenerator`）的 `_stout_smear_ndarray` / `_stout_smear_cuda_kernel` / `_stout_smear_quda` 是同一算法的复制粘贴（对比 elemental.py:255-336 与 displacement_elemental.py:111-179、eigenvector.py:141-222 逐行相同；`CurrentElementalGenerator` 的 docstring 自述 "Copied from ElementalGenerator"，elemental.py:1166 附近）。唯一实现差异：

- `EigenvectorGenerator._stout_smear_quda` 保留全部 4 个方向（`gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)`，eigenvector.py:245-247）并缓存 `self.gauge_quda`；其余类只取 `[:Nd-1]`（elemental.py:356-358）。
- QUDA 分派顺序：`ElementalGenerator`/`CurrentElementalGenerator`/`DisplacementElementalGenerator` 为 cupy 下 kernel 优先、QUDA 次之；`EigenvectorGenerator` 为 QUDA 优先（eigenvector.py:257-260）。

### 4.7 数据流总览

```
gauge config ──► EigenvectorGenerator.load / project_SU3 / stout_smear
                    └─► calc(t) 逐时间片：3D covariant Laplacian 低模 ──► eigenvector 落盘
eigenvector + gauge ──► ElementalGenerator (calc_deriv / calc_disp) ──► VPV elemental
                    └─► CurrentElementalGenerator.calc_all ──► v2v/v2p/p2v/p2p ──► insertion.current.compute_elemental
eigenvector ──► NoisevectorGenerator.calc ──► noisevector（随机源，仓库内尚未接线）
              └─► generate_sparsened_points ──► PointSource 坐标表
eigenvector + gauge (+ point) ──► PerambulatorGenerator (stout_smear → calc/calc_point_sources)
                                    ──► VSV / PSV / PSP ──► lattice/correlator/one_particle.py 等收缩
                                  └─► Density / Generalized 变体 ──► 带插入的逐点/逐时刻 VSSV
```

---

## 5. 不变量与校验

### 5.1 显式校验清单

| 位置 | 校验 |
|---|---|
| elemental.py:19-35 `_bounded_count` | `usedNe/usedNp` 须为非 bool 的 int 且 `0 <= used <= available`，否则 `TypeError`/`ValueError`（CurrentElementalGenerator :1074 使用） |
| elemental.py:138-140 | blending 断言：块数一致、`used <= tot`、`sum(used) == Ne` |
| elemental.py:170 | `is_blending=True` 且无 dilution → `ValueError("Dilution tuple is not defined.")` |
| elemental.py:1298-1320（`calc_directed_current_raw`） | `boundary ∈ {"periodic","open"}`；`_current_U` 已加载且第一维 == Nd；gauge 形状 == `(Nd, Lt, Lz, Ly, Lx, Nc, Nc)`；eigenvector 形状 == `(Lt, loaded_ne, Lz, Ly, Lx, Nc)`——全部显式 `raise ValueError` |
| elemental.py:920-931（`re_combine_auto`） | 首次 dense/sparse 交叉验证失败 → `ValueError("re_combine validation failed: ...")` |
| displacement_elemental.py:53-71 + :88-89 | `_D` 依赖 `self.Vd` 跨调用残留状态，`calc` 必须按 `dist` 递增调用 |
| sparsened_point.py:65-77 | `len(latt_size) == 4`、`0 < num_points <= Lx·Ly·Lz` |
| noisevector.py:27-28 | `len(noisev_list) == len(eigv_list)`、逐块 `noisev <= eigv`（assert） |
| perambulator.py:128-131 | `assert backend == "cupy"`（PyQUDA 路径 cupy-only） |
| perambulator.py:311-323 | 未 `load()` 先 `stout_smear` → `ValueError`；numpy 后端 → `NotImplementedError` |
| perambulator.py:388, :571 | 源时间片归属 `gt*Lt <= t_src < (gt+1)*Lt`——只有拥有 `t_src` 的 rank 写入源，其余 rank 用零源求逆；多卡求和后正确（配合 `test_perambulator_mpi.py:64` 的 `reduce_op="sum"`） |
| perambulator.py:524-531 | `calc_new` 的 `products` 白名单（`{"VSV","PSV"}`，转大写） |
| perambulator.py:666-673 | `calc_point_sources` 需要双 point 句柄与 `_PSP` 缓冲 |
| density_perambulator.py:30-31 / generalized_perambulator.py:31-32 | `if not check_QUDA(): raise ImportError`（`PerambulatorGenerator` 中对应检查被注释掉，perambulator.py:117-118——三者不一致，代码现状如此） |
| density_perambulator.py:37 / generalized_perambulator.py:38 | `assert backend == "cupy"` |
| eigenvector.py:428-452（`calc` 分派） | 不支持的后端 → `NotImplementedError` |

### 5.2 已知代码现状问题（candidate，如实记录，未做运行时复现或仅部分复现）

1. **`DisplacementElementalGenerator._stout_smear_quda` 属性名错位**：`_stout_smear_quda` 读取 `self._gauge_field_path`（displacement_elemental.py:195），但 `load()` 里赋值的属性名是 `self._gauge_field_data`（displacement_elemental.py:75，构造时初始化亦为 `_gauge_field_data`，:46）。若 cupy 后端下无 CUDA kernel 且 QUDA 可用而选中该分支，会触发 `AttributeError`。numpy 后端测试走 `_ndarray` 分支，未覆盖此路径。对照：`ElementalGenerator.load` 存 `self._gauge_field_path`（elemental.py:176）、`CurrentElementalGenerator.load` 同（elemental.py:1089），与各自的 `_stout_smear_quda` 一致，只有 displacement 类不一致。
2. **displacement elemental 与仓库参考数据失配**：实测（numpy 后端，4⁴ 小格、Ne=20、distance=8）`python test/test_displacement_elemental.py` 输出残差 `res = 36.25`，而 `test/test_elemental.py` 为 `8.7e-14`。定位实验显示：参考文件 `test/weak_field.displacement_elemental.npy` 在 `dist=0, mom=(0,0,0)` 处为 `1.0000000052`，而当前 `calc` 因累加两项（displacement_elemental.py:92-93）给出 `2.0`；git 证据（`git log -S "right.conj(), left"`）表明第二项由提交 `8923174` 引入，该文件 `calc` 在 `f8b3251` 时只有单项。只保留单项重算后全局残差仍有 21.34，说明还有其他未定位差异。该测试脚本只打印残差、无断言（test/test_displacement_elemental.py:35-45），故不会失败。是参考数据过期还是第二项为有意对称化，仅凭代码无法判定。
3. **`calc_old` 的 PSV 抽取路径失效**（perambulator.py:433-462）：`for t_snk in range(Lt)` 循环体只有 `continue`（:435-437），循环后 `t_snk == Lt-1`，`t_index` 恒取最后一片；点归属用半格距（:447）而 helper 用全格距（:38-42）；parity 用全局坐标（:455-458）而 helper 用本地坐标。`calc_new` 已用 `_point_sink_indices` 绕开，但 `use_vectorized=False` 时 `calc()` 仍走 `calc_old`（:754-759）；phase3 测试的 PSV 一致性检查因配置 `Np_snk=0` 被跳过（test_perambulator_phase3.py:127-136）。
4. ~~**density 版 gamma5 对合被重复施加**~~（**已修复**）：原 `SV_f[:] = contract("ii,kezyxjic,jj->kezyxijc", ...)` 位于 `for eigen` 循环体内、`if tf != self._tf:` 之外（缩进经 `cat -A` 核实），且作用于**整个缓冲**而非当前行。由于 `T(S) = gamma5·conj(S)·gamma5` 满足 `T∘T = id`，第 e 行被作用 `Ne - e` 次：奇偶为偶数时传出的仍是**未共轭**传播子（Ne=70 时 35/70 行错误，且同 `(ti,tf)` 二次调用会把已正确的行整体翻转回未共轭）。现已移入 `if` 并改为只作用于 `SV_f[eigen]`，数值复现从 35/70 → 70/70，VSSV 残差 7.35e+02 → 0。
5. ~~**density/generalized 用全局时间索引本地时间轴**~~（**已修复**）：`V[:, ti, ...]`（density_perambulator.py:144/:155、generalized_perambulator.py:154/:165）直接用**全局** `ti`/`tf` 引 `LatticeFermion` 的**本地**时间轴（长度 `Lt = GLt/Gt`），既无 `ti - gt*Lt` 换算也无归属判断。现已改为 `owns` 门控 + 本地时间（见 §4.5）。原描述中"写错片/越界"只适用于多卡，而多卡在该处之前就会先撞上全局/本地尺寸不一致的 `reshape ValueError`，故实际影响是：该模块仅在单卡下可运行，单卡下恰好恒等因此原本未暴露。完整的全局/本地重构未做（超出范围），现状已记录在 §4.5。
6. **density 版缓存键不含 tau**（density_perambulator.py:69-70, :135, :143, :149-152）：`_t/_tf` 只记录 (ti, tf)；以相同 (ti, tf) 但不同 tau 连续调用 `calc` 时反演被缓存跳过，而 `[:, tau, ...]` 切片发生在反演时——旧 tau 的数据被沿用，结果错误。generalized 版不依赖 tau，不受影响。
7. **`test/test_perambulator_mpi.py:33` 使用已不存在的构造参数**：该测试用 `eigenvector=eigenvector` 关键字，而当前签名只有 `eigenvector_src`（perambulator.py:104）——按现状该测试会 `TypeError`。
8. **noisevector 的 `full_noisev` 语义不闭合**（noisevector.py:42-44, :52-72）：`full_noisev=True` 时缓冲按 `sum(eigv_list)` 分配，但只有走"一般块"分支的 block 会把 `noisev_lenth` 提升为 `eigv_lenth`（:72）；`==1` 分支与恒等分支不受影响，可能出现分配的列未被写满（保持 0）。属边界语义未定义。
9. **死代码**：`PerambulatorGenerator._SP` 恒为 `None`（perambulator.py:199-200, 215-217）；`NoisevectorGenerator.transfer_matrix` 赋 `None` 后从未读写（noisevector.py:21）。
10. **`_gauge_links_product` 前向累积未滚动**：elemental.py:386-398 前向分支用 `self._U[d]`（未滚动），`self._U_shift` 仅在后向分支使用。对单链接路径无影响；对多链接非共线路径是否与逐步 `_disp` 完全一致未在本地复验（对比脚本 `test/scripts/test_disp_vs_product.py:116-160` 即为此目的而设，且其只比 disp_idx 1..2）。三处同代码实现相互一致。
11. **类 docstring 与签名漂移**：`PerambulatorGenerator` 类 docstring（perambulator.py:40-89）写 `eigenvector`/`usedNe`，实际签名为 `eigenvector_src`/`usedNe_src`/`usedNe_snk`；`calc_new` docstring 中的 "3-5x speedup" 等声明未经验证。以签名为准。
12. **`PerambulatorGenerator` 构造不做 PyQUDA 硬检查**：检查被注释掉（perambulator.py:117-118），而两个变体模块有 `raise ImportError`——三个 perambulator 类的前置校验强度不一致，代码现状如此。

---

## 6. 相关测试

| 测试文件 | 覆盖对象与要点 |
|---|---|
| `test/test_elemental.py` | `ElementalGenerator.calc_deriv`：对 `weak_field` 配置逐 t 调 `calc(t)`，收集后 `transpose(1,2,0,3,4)` 与 `ElementalNpy(..., [num_deriv, num_mom, Lt, Ne, Ne])` 参考比较；实测残差 `8.7e-14`（通过） |
| `test/test_displacement_elemental.py` | `DisplacementElementalGenerator`（distance=8，无 smearing/SU3 投影）与参考比较；**只打印残差、无断言**（:35-45）；当前残差 36.25（见 §5.2-2） |
| `test/test_eigenvector.py` | `EigenvectorGenerator`：`pytestmark = pytest.mark.gpu`，cupy 不可用时整模块 skip（:7-22）；流程 `load → stout_smear(10, 0.12) → 逐 t calc(t)`（默认参数与 Chebyshev 模式 `calc(t, True, 10, evals[:,-1].real*1.1)`，:57-70）；校验允许每个本征向量一个整体相位 `phase = ref[0]/out[0]`，`allclose(..., rtol=1e-7)` 否则 raise（:34-48）；本机（无 GPU）未运行 |
| `test/test_current_elemental.py` | 测 `lattice` 顶层的 `CurrentElementalV2P/P2V/P2P`、`OverlapMatrixNpy` 等**数据读取容器**（unittest，import 失败即 skip），不直接测 `CurrentElementalGenerator` |
| `test/test_temporal_current_elemental.py` | `CurrentElementalGenerator` + `lattice/insertion/current.py`（`build_current_raw_contract`、`assemble_current_terms`、directed raw schema 等）的 current 元数据与组装契约 |
| `test/test_sparsened_point.py` | `generate_sparsened_points` 全覆盖：shape/dtype/坐标范围、单时间片内唯一性、seed 可复现与 `seed=None` 走全局随机态、单点/满容量上界、三类 `ValueError`、非立方格点、数组独立性、`np.save/np.load` 工作流、参数化 sweep（pytest，可直接 CPU 运行） |
| `test/test_perambulator.py` | `pytest.mark.mpi` + `importorskip("pyquda")`（:6-9）；4³×8 格点、Ne=20、`t_boundary=-1`、`stout_smear(20, 0.1)`；`peramb[t] = numpy.roll(perambulator.calc(t).get(), -t, 0)` 确立盘上轴约定 `[t_src, t_rel, s_snk, s_src, e_snk, e_src]`，与参考比较 |
| `test/test_perambulator_mpi.py` | grid `[1,1,2,2]` 多卡；`gatherLattice(peramb, axes=[1,-1,-1,-1], reduce_op="sum", root=0)` 聚合（:64）；**当前使用了不存在的 `eigenvector=` 参数，会 `TypeError`**（§5.2-6） |
| `test/test_perambulator_phase3.py` | 四组：形状/dtype 兼容（VSV `(Lt,Ns,Ns,Ne,Ne)`、`np.complex128`，:63-88）、calc_old vs calc_new 数值一致性（`||VSV_old − VSV_new|| < 1e-10`，:108-124；PSV 一致性因 `Np_snk=0` skipped，:127-136）、性能基准（只打印不断言，:139-183）、确定性/时间片无关性（:186-230）；文件头注释承认参考值为 `[PLACEHOLDER]`（:11） |
| `test/test_propagator_psv.py` | PSV **读取容器**（`PropagatorPSVNpy`/`PropagatorPSVTimeslicesNpy`）的 lazy FileData 契约，不测生成端 |
| （无） | `NoisevectorGenerator`、`GeneralizedPerambulatorGenerator`、`DensityPerambulatorGenerator`（仅 `example/gen_density_peram.py`）、`re_combine`、`get_overlap_matrix` 均无测试覆盖 |

`test/scripts/` 下另有作者自用的对比脚本（如 `test_disp_vs_product.py` 对比 `_disp` 与 `_gauge_links_product`、`test_calc_calc_disp_consistency.py` 对比 calc_deriv 与 calc_disp——后者含硬编码集群路径，本地不可运行）。
