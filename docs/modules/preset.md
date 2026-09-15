# `lattice/preset.py` — 物理数据源 preset 模块

> 定位：`lattice/preset.py` 是本仓库的**物理数据源预设层**，把「某类物理量（gauge field、eigenvector、perambulator、elemental、propagator、一点/两点关联函数）」与「某种磁盘格式（QDP 裸二进制、ILDG/LIME、`.npy`、按时间切片的 `.npy`）」组合成统一暴露 `load(cfg)` 的 lazy 加载器类。

## ① 模块概览

### 模块分层

模块分两层（`lattice/preset.py:11-149`）：

1. **逻辑角色基类（mixin，12 个）**：`GaugeField`、`Eigenvector`、`Elemental`、`CurrentElemental`、`Perambulator`、`PointSource`、`OverlapMatrix`、`PropagatorPSV`、`PropagatorVSP`、`PropagatorPSP`、`OnePoint`、`TwoPoint`。它们只持有元数据（`elem: FileMetaData` 与 `Ne`/`Np` 等维度参数），不实现任何 I/O。
2. **具体加载器类（25 个）**：以「格式基类 + 角色基类」多继承组成，统一暴露 `load(key)`，文件名约定 `{prefix}{key}{suffix}`（如 `lattice/preset.py:160`、`lattice/preset.py:742`）。

### 依赖关系

模块头 import（`lattice/preset.py:1-6`）：

| 依赖 | 用途 |
|---|---|
| `lattice.filedata.abstract` | `FileData`（lazy handle 抽象）、`FileMetaData`（shape/dtype/extra 三字段记录类，`lattice/filedata/abstract.py:5-9`） |
| `lattice.filedata.binary` | `BinaryFile`：无 header 裸二进制，完全信任 `FileMetaData`（`lattice/filedata/binary.py:17-25`） |
| `lattice.filedata.ildg` | `IldgFile`：ILDG/LIME 格式，加载器内校验 shape 与精度（`lattice/filedata/ildg.py:35-36`） |
| `lattice.filedata.timeslice` | `QDPLazyDiskMapObjFile`：QDP record 文件，`extra` 轴作为 record key（`lattice/filedata/timeslice.py:36-52`） |
| `lattice.filedata.ndarray` | `NdarrayFile` / `NdarrayTimeslicesFile`：`.npy` mmap 加载；timeslice 版将切片首维解释为时间切片号并替换文件名中的 `.t???` 占位符（`lattice/filedata/ndarray.py:71-73`） |

**被谁使用**：

- `lattice/__init__.py:4-31` 把 25 个具体加载器与部分角色基类（`PointSource`、`OverlapMatrix`、`PropagatorPSV`、`PropagatorVSP`、`PropagatorPSP`）重导出，构成公开门面。
- `lattice/propagators.py:30-38` import 角色基类 `Perambulator`、`PropagatorVSP`、`PropagatorPSV`、`PropagatorPSP`、`OverlapMatrix`、`CurrentElementalV2P`、`CurrentElementalP2V`、`CurrentElementalP2P` 作为类型注解；`propagators.py` 的 `_loader_signature`（`lattice/propagators.py:149-181`）读取加载器的 `prefix/suffix/elem/Ne/Np` 属性生成缓存身份。
- `lattice/generator/*.py`（7 个文件）import `GaugeField`、`Eigenvector`、`PointSource` 作为生成器输入端口的类型注解（`lattice/generator/eigenvector.py:8`、`lattice/generator/elemental.py:11`、`lattice/generator/perambulator.py:7`、`lattice/generator/noisevector.py:8`、`lattice/generator/displacement_elemental.py:7`、`lattice/generator/density_perambulator.py:7`、`lattice/generator/generalized_perambulator.py:8`）。

### 继承体系

两种多继承方向统一为「格式基类在前、角色基类在后」，无菱形继承。初始化模式（以 `GaugeFieldTimeSlice` 为例，`lattice/preset.py:152-159`）：

```python
def __init__(self, prefix, suffix, shape=...) -> None:
    super().__init__()                                   # MRO 命中格式基类 __init__（设置 self.file/data）
    GaugeField.__init__(self, FileMetaData(shape, ">c16", 2))  # 显式调用角色基类（设置 self.elem）
    self.prefix = prefix
    self.suffix = ".stout.n20.f0.12.mod" if suffix is None else suffix
```

`load` 内的 `super().get_file_data(...)` 命中格式基类的工厂方法（`lattice/preset.py:160-162`）。

### 加载器汇总表

| 类名 | 物理数据 | 磁盘格式 | 构造参数 | 输出形状 |
|---|---|---|---|---|
| `GaugeFieldTimeSlice` (`lattice/preset.py:151`) | gauge 构型（链接变量） | QDP record（key=`(t, mu)`） | `prefix, suffix, shape=[128,4,16³,3,3]` | record 形状 `[Vol,3,3]`，cast `<c8` |
| `GaugeFieldIldg` (`lattice/preset.py:687`) | gauge 构型 | ILDG/LIME 单 record | `prefix, suffix, shape=[128,16³,4,3,3]` | `[Lt,Lz,Ly,Lx,Nd,Nc,Nc]`（实际用法），cast `<c16` |
| `GaugeFieldBinary` (`lattice/preset.py:700`) | gauge 构型 | 裸二进制（`<f8` 默认） | `prefix, suffix, shape=[128,16³,4,3,3], dtype="<f8"` | 声明 shape，无 cast |
| `EigenvectorTimeSlice` (`lattice/preset.py:164`) | LapH eigenvector 集 | QDP record（key=`(t,e)`） | `prefix, suffix, shape=[128,70,16³,3], totNe=70` | record 形状 `[Vol,Nc]`，cast `<c8` |
| `EigenvectorNpy` (`lattice/preset.py:183`) | LapH eigenvector 集 | `.npy`（mmap） | `prefix, suffix, shape=[70,128,16³,3], totNe=70` | 实际用法 `[Lt,Ne,Lz,Ly,Lx,Nc]` |
| `PerambulatorBinary` (`lattice/preset.py:200`) | distillation perambulator | 裸二进制 | `prefix, suffix, shape=[128,128,4,4,70,70], totNe=70` | `[t_src,t_snk,Ns,Ns,Ne,Ne]` |
| `PerambulatorNpy` (`lattice/preset.py:217`) | distillation perambulator | `.npy`（`<c8`） | 同上 | 同上 |
| `PerambulatorTimeslicesNpy` (`lattice/preset.py:553`) | perambulator（按 t_src 切片） | `.npy` timeslices | 同上 | 切片首维 = t_src |
| `PointSourceNpy` (`lattice/preset.py:234`) | 稀疏点源坐标 | `.npy`（`<i4`） | `prefix, suffix, shape=[128,72,3], Np=128` | `(Np, Lt, 3)` zero-based `[x,y,z]` |
| `OverlapMatrixNpy` (`lattice/preset.py:250`) | eigenvector×point 重叠矩阵 | `.npy`（`<c16`） | `prefix, suffix, shape=[72,128,216,3], Ne=128, Np=216` | `[Lt,Ne,Np,Nc]` |
| `PropagatorPSVNpy` (`lattice/preset.py:270`) | point→eigenvector 传播子 | `.npy` | `prefix, suffix, shape(必填), Np, Ne, dtype="<c16"` | `[Lt,Lt,Ns,Ns,Np,Nc,Ne]` |
| `PropagatorVSPNpy` (`lattice/preset.py:329`) | eigenvector→point 传播子 | `.npy` | 同上 | `[Lt,Lt,Ns,Ns,Ne,Np,Nc]` |
| `PropagatorVSPTimeslicesNpy` (`lattice/preset.py:388`) | VSP（按 t_src 切片） | `.npy` timeslices | 同上 | 全量 `[Lt,Lt,Ns,Ns,Ne,Np,Nc]`，每文件 6 维 |
| `PropagatorPSPNpy` (`lattice/preset.py:441`) | point→point 传播子 | `.npy` | `prefix, suffix, shape(必填), Np_snk, Np_src, dtype="<c16"` | `[Lt,Lt,Ns,Ns,Np_snk,Nc_snk,Np_src,Nc_src]` |
| `PropagatorPSPTimeslicesNpy` (`lattice/preset.py:500`) | PSP（按 t_src 切片） | `.npy` timeslices | 同上 | 全量 8 维，每文件 7 维 |
| `PropagatorPSVTimeslicesNpy` (`lattice/preset.py:574`) | PSV（按 t_src 切片） | `.npy` timeslices | 同 PSVNpy | 全量 `[Lt,Lt,Ns,Ns,Np,Nc,Ne]`，每文件 6 维 |
| `ElementalBinary` (`lattice/preset.py:657`) | meson elemental | 裸二进制（`<c16`） | `prefix, suffix, shape=[40,27,128,70,70], totNe=70` | `[num_disp,num_mom,Lt,Ne,Ne]` |
| `ElementalNpy` (`lattice/preset.py:717`) | meson elemental（Legacy） | `.npy`（`<c8`） | `prefix, suffix, shape=[4,123,128,70,70], totNe=70` | `(num_disp,num_mom,Lt,Ne,Ne)` |
| `CurrentElementalV2V` (`lattice/preset.py:741`) | current elemental V2V | `.npy`（`<c16`） | `prefix, suffix, shape(必填), Ne` | 视图 `(num_disp,num_mom,Lt,Ne,Ne)` |
| `Jpsi2gammaBinary` (`lattice/preset.py:674`) | 两点函数（Jpsi→γ） | 裸二进制（`<f8`） | `prefix, suffix, shape=[128,2,3,4,27,128]` | 声明 shape |
| `Jpsi2gammaNpy` (`lattice/preset.py:790`) | 两点函数 | `.npy` | `prefix, suffix`（无 shape 参数，`elem=None`） | 由 npy header 决定 |
| `OnePointNpy` (`lattice/preset.py:801`) | 一点函数 | `.npy` | `prefix, suffix`（无 shape 参数，`elem=None`） | 由 npy header 决定 |
| `CurrentElementalV2P` (`lattice/preset.py:813`) | current elemental V2P | `.npy`（`<c16`） | `prefix, suffix, shape(必填), Ne, Np` | `[Lt,num_disp,Ne,Np,Nc]` |
| `CurrentElementalP2V` (`lattice/preset.py:838`) | current elemental P2V | `.npy`（`<c16`） | 同上 | `[Lt,num_disp,Np,Nc,Ne]` |
| `CurrentElementalP2P` (`lattice/preset.py:863`) | current elemental P2P（稀疏） | HDF5 或 `.npz`（同步物化） | `prefix, suffix=None` | `List[dict]`（每 `(disp, momentum)` 一项） |

## ② 公开 API 参考

### 2.1 角色基类（mixin，12 个）

所有角色基类均在 `lattice/preset.py:11-149`，模式统一：`deepcopy(elem)` 后保存维度参数。

| 类 | `__init__` 签名 | 保存属性 | 语义 |
|---|---|---|---|
| `GaugeField` (`lattice/preset.py:11`) | `(elem: FileMetaData)` | `elem` | gauge 构型（链接变量） |
| `Eigenvector` (`lattice/preset.py:16`) | `(elem: FileMetaData, eigenNum: int)` | `elem, Ne` | LapH eigenvector 集，`Ne = eigenNum` |
| `Elemental` (`lattice/preset.py:22`) | `(elem: FileMetaData, eigenNum: int)` | `elem, Ne` | meson elemental，V×V 矩阵元 |
| `CurrentElemental` (`lattice/preset.py:28`) | `(elem: FileMetaData, eigenNum: int, pointNum: int = None)` | `elem, Ne, Np` | 含 point 信息的 current 顶点数据 |
| `Perambulator` (`lattice/preset.py:41`) | `(elem: FileMetaData, eigenNum: int)` | `elem, Ne` | distillation perambulator |
| `PointSource` (`lattice/preset.py:47`) | `(elem: FileMetaData, Np: int)` | `elem, Np` | 稀疏点源坐标集 |
| `OverlapMatrix` (`lattice/preset.py:53`) | `(elem: FileMetaData)` | `elem` | eigenvector×point 重叠矩阵（Ne/Np 由具体子类补充，见 `OverlapMatrixNpy`） |
| `PropagatorPSV` (`lattice/preset.py:58`) | `(elem: FileMetaData, Np: int, Ne: int)` | `elem, Np, Ne` | point→eigenvector 传播子：`S_{xa,i} = <eta_{x,a}|S|xi_i>` |
| `PropagatorVSP` (`lattice/preset.py:85`) | `(elem: FileMetaData, Np: int, Ne: int)` | `elem, Np, Ne` | eigenvector→point 传播子：`S_{i,xa} = <xi_i|S|eta_{x,a}>`，与 PSV 方向相反 |
| `PropagatorPSP` (`lattice/preset.py:114`) | `(elem: FileMetaData, Np_snk: int, Np_src: int)` | `elem, Np_snk, Np_src` | point→point 传播子：`S_{xa,yb} = <eta_{x,a}|S|eta_{y,b}>` |
| `OnePoint` (`lattice/preset.py:141`) | `(elem: FileMetaData)` | `elem` | 一点函数 |
| `TwoPoint` (`lattice/preset.py:146`) | `(elem: FileMetaData)` | `elem` | 两点函数 |

`Ne/Np/elem.shape` 不只是注释：`lattice/propagators.py:96-120` 的 `_available_extent`/`_normalize_extent` 会把它们与数据 shape 交叉校验，不一致则 `raise ValueError`；`lattice/propagators.py:150-181` 的 `_loader_signature` 也读取这些属性生成缓存身份。

### 2.2 具体加载器（25 个）

除特别说明外，所有加载器的 `load(self, key: str)` 均为 `return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)`，返回 lazy `FileData` handle（惰性，`__getitem__` 才真正读盘）。

#### Gauge 系列

**`GaugeFieldTimeSlice(QDPLazyDiskMapObjFile, GaugeField)`** — `lattice/preset.py:151-162`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 4, 16**3, 3, 3])`
- 元数据：`FileMetaData(shape, ">c16", 2)` — 大端 complex128，`extra=2` 表示前两轴 `(t, mu)` 是 QDP record key；每条 record 的数据形状 = `shape[2:] = [Vol, 3, 3]`。
- `suffix is None` 时默认 `".stout.n20.f0.12.mod"`。返回值被 QDP 加载器强制 cast 到 `"<c8"`（`lattice/filedata/timeslice.py:96`）。
- QDP 版 gauge shape 顺序 `[Lt, Nd, Vol, Nc, Nc]` 与 ILDG 版 `[Lt, Vol, Nd, Nc, Nc]`（`lattice/preset.py:689`）不同——两种格式的第 2/3 轴互换。

**`GaugeFieldIldg(IldgFile, GaugeField)`** — `lattice/preset.py:687-698`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 16**3, 4, 3, 3])`
- 元数据：`FileMetaData(shape, ">c16", 0)`，`suffix is None` 时默认 `".lime"`。整个构型是单条 `ildg-binary-data` record；加载器断言 shape×dtype 与 record 长度一致（`lattice/filedata/ildg.py:35-36`），返回 cast 到 `"<c16"`（`lattice/filedata/ildg.py:70`）。
- 实际使用中 shape 传 7 维 `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]`（chroma convention，`test/test_load.py:23-24`），默认的 5 维只是占位。

**`GaugeFieldBinary(BinaryFile, GaugeField)`** — `lattice/preset.py:700-715`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 16**3, 4, 3, 3], dtype: str = "<f8")`
- 默认后缀 `".dat"`。裸二进制、无 header；二进制路径完全信任 `FileMetaData`（`lattice/filedata/binary.py:17-25`），shape/dtype 错误不会报错。

#### Eigenvector 系列

**`EigenvectorTimeSlice(QDPLazyDiskMapObjFile, Eigenvector)`** — `lattice/preset.py:164-181`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 70, 16**3, 3], totNe: int = 70)`
- 元数据：`FileMetaData(shape, ">c8", 2)` — 大端 complex64，record key = `(t, e)`，每条 record 是 `[Vol, Nc]` 复向量。默认后缀 `".stout.n20.f0.12.laplace_eigs.3d.mod"`。返回 cast `"<c8"`。`Ne = totNe`。

**`EigenvectorNpy(NdarrayFile, Eigenvector)`** — `lattice/preset.py:183-198`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [70, 128, 16**3, 3], totNe: int = 70)`
- 元数据：`FileMetaData(shape, "<c16", 2)`，默认后缀 `".lime.npy"`。
- 测试与实际使用的 shape 是 `[Lt, Ne, Lz, Ly, Lx, Nc]`（`test/test_elemental.py:19`、`test/test_perambulator.py:27`）——磁盘轴序 t 优先，与默认值 `[70, 128, ...]`（Ne 优先）不一致；默认值只是历史占位。注意：`.npy` 路径的实际切片 shape/dtype 来自 npy header 而非 `FileMetaData`（`lattice/filedata/ndarray.py:29-45`），因此 `extra=2` 对 npy 路径无实际作用。

#### Perambulator 系列

**`PerambulatorBinary(BinaryFile, Perambulator)`** — `lattice/preset.py:200-215`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 128, 4, 4, 70, 70], totNe: int = 70)`
- 元数据：`FileMetaData(shape, "<c16", 0)`，默认后缀 `".stout.n20.f0.12.nev70.peram"`。轴序 `[t_src, t_snk, Ns, Ns, Ne, Ne]`。
- `example/gen_two_particle_corr.py:40-44` 用它加载 light/charm 两个 perambulator。

**`PerambulatorNpy(NdarrayFile, Perambulator)`** — `lattice/preset.py:217-232`
- 签名同上；元数据：`FileMetaData(shape, "<c8", 0)`（complex64 npy）。实际 shape 用法 `[Lt, Lt, Ns, Ns, Ne, Ne]`（`test/test_perambulator.py:53`）。

**`PerambulatorTimeslicesNpy(NdarrayTimeslicesFile, Perambulator)`** — `lattice/preset.py:553-572`
- 签名同上；元数据：`FileMetaData(shape, "<c8", 0)`。切片 key 首维 = t_src。
- **默认后缀 `".stout.n20.f0.12.nev70.peram"` 不含 `.t???` 占位符**——按 `NdarrayTimeslicesFileData` 的替换逻辑（`lattice/filedata/ndarray.py:71-73`），默认后缀下切片时找不到文件；必须显式传含 `.t???` 的 suffix 才能工作（机制由 `test/test_propagator_psv.py:110-118` 对同类加载器的契约测试确认）。

#### Point / Overlap 系列

**`PointSourceNpy(NdarrayFile, PointSource)`** — `lattice/preset.py:234-248`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 72, 3], Np: int = 128)`
- 元数据：`FileMetaData(shape, "<i4", 0)`（整型坐标），默认后缀 `".npy"`。
- 磁盘布局 `(Np, Lt, 3)`，zero-based `[x, y, z]`（`lattice/preset.py:240-243` 注释：加载器从 npy header 读真实 dtype，元数据仅为消费者文档）。

**`OverlapMatrixNpy(NdarrayFile, OverlapMatrix)`** — `lattice/preset.py:250-268`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [72, 128, 216, 3], Ne: int = 128, Np: int = 216)`
- 元数据：`FileMetaData(shape, "<c16", 2)`，默认后缀 `".overlap_matrix.npy"`。
- shape 注释 `[Lt, Ne, Np, Nc]`（`lattice/preset.py:254`）。**注意**：`OverlapMatrix` 基类不存 Ne/Np，此子类直接 `self.Ne = Ne; self.Np = Np`（`lattice/preset.py:262-263`）；`propagators.py` 通过 duck-typed `getattr(loader, "Ne"/"Np")` 读取。

#### Propagator 系列（6 个）

全部 `FileMetaData(shape, dtype, 0)`（整文件一条记录），dtype 默认 `"<c16"`；shape 均为**必填**（无默认）。

**`PropagatorPSVNpy(NdarrayFile, PropagatorPSV)`** — `lattice/preset.py:270-327`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int], Np: int, Ne: int, dtype: str = "<c16")`
- shape `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`；默认后缀 `".npy"`。docstring 见 `lattice/preset.py:271-302`。

**`PropagatorVSPNpy(NdarrayFile, PropagatorVSP)`** — `lattice/preset.py:329-386`
- 签名同 PSVNpy；shape `[Lt, Lt, Ns, Ns, Ne, Np, Nc]`；默认后缀 `".npy"`。

**`PropagatorVSPTimeslicesNpy(NdarrayTimeslicesFile, PropagatorVSP)`** — `lattice/preset.py:388-439`
- 签名同上；全量 shape `[Lt, Lt, Ns, Ns, Ne, Np, Nc]`，每文件 `[Lt, Ns, Ns, Ne, Np, Nc]`；默认后缀 `".npy"`（同样不含 `.t???`，见上文 timeslice 契约）。

**`PropagatorPSPNpy(NdarrayFile, PropagatorPSP)`** — `lattice/preset.py:441-498`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int], Np_snk: int, Np_src: int, dtype: str = "<c16")`
- shape `[Lt, Lt, Ns, Ns, Np_snk, Nc_snk, Np_src, Nc_src]`（8 维，docstring 示例 `[72,72,4,4,216,3,216,3]`，`lattice/preset.py:476-487`）；默认后缀 `".npy"`。

**`PropagatorPSPTimeslicesNpy(NdarrayTimeslicesFile, PropagatorPSP)`** — `lattice/preset.py:500-551`
- 签名同 PSPNpy；全量 8 维，每文件 7 维；默认后缀 `".npy"`。

**`PropagatorPSVTimeslicesNpy(NdarrayTimeslicesFile, PropagatorPSV)`** — `lattice/preset.py:574-651`
- 签名同 PSVNpy；全量 `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`，每文件 `[Lt, Ns, Ns, Np, Nc, Ne]`；默认后缀 `".npy"`。
- docstring 记录了生成约定：每文件存 `np.roll(PSV.get(), -t_src, 0)` 后的相对时间数组（`lattice/preset.py:620-627`）。

#### Elemental 系列

**`ElementalBinary(BinaryFile, Elemental)`** — `lattice/preset.py:657-672`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [40, 27, 128, 70, 70], totNe: int = 70)`
- 元数据：`FileMetaData(shape, "<c16", 0)`，默认后缀 `".stout.n20.f0.12.nev70.meson"`。轴序 `[num_disp, num_mom, Lt, Ne, Ne]`。

**`ElementalNpy(NdarrayFile, Elemental)`** — `lattice/preset.py:717-739`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [4, 123, 128, 70, 70], totNe: int = 70)`
- 元数据：`FileMetaData(shape, "<c8", 0)`，默认后缀 `".stout.n20.f0.12.nev70.meson.npy"`。
- 磁盘布局 `(num_disp, num_mom, Lt, Ne, Ne)`，与消费端轴序一致（`lattice/propagators.py:279-281` 切片 `elemental_data[derivative_idx, momentum_idx, :, :usedNe, :usedNe]`，轴 0=displacement、轴 1=momentum、轴 2=时间）。
- docstring 自述为 "Legacy"，建议改用 `CurrentElementalV2V`（`lattice/preset.py:718-723`）。
- 是 meson 通道的标准 elemental 加载器，被 `test/test_elemental.py:30`、`test/test_meson_spectrum.py:38`、`test/scripts/profile_vertex_map_timing_gap.py:165` 等广泛使用。

**`CurrentElementalV2V(NdarrayFile, Elemental)`** — `lattice/preset.py:741-788`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int], Ne: int)` — **shape 必填**。
- 元数据：`FileMetaData(shape, "<c16", 0)`，默认后缀 `"_v2v.npy"`。**注意角色基类是 `Elemental`（不是 `CurrentElemental`）**，`self.Ne` 即传入的 `Ne`。
- 磁盘布局 `(Lt, num_disp, num_mom, Ne, Ne)`（docstring，`lattice/preset.py:746-753`）——与 `ElementalNpy` 相比时间轴在最前。
- `load()` 返回的不是原始 handle，而是内嵌类 **`_View`**（`lattice/preset.py:754-768`）：其 `shape` 属性对外呈现 `(num_disp, num_mom, Lt, Ne, Ne)`，`__getitem__` 把 `(disp, mom, t, e1, e2)` 的 key 重排为 `(t, disp, mom, e1, e2)` 索引底层 mmap。
- `_View.__getitem__` 要求 `isinstance(key, tuple) and len(key) >= 5`，否则 `raise IndexError`（`lattice/preset.py:763-766`）——因此 `view[:]`、`view[0]` 等常规切片全部失败，只能按完整 5 元组索引。

#### CurrentElemental 三件套

**`CurrentElementalV2P(NdarrayFile, CurrentElemental)`** — `lattice/preset.py:813-836`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int], Ne: int, Np: int)` — shape 必填。
- 元数据：`FileMetaData(shape, "<c16", 2)`，默认后缀 `"_v2p.npy"`；shape `[Lt, num_disp, Ne, Np, Nc]`。
- 消费端印证：`lattice/propagators.py:482` 读 `p2v_full = self.p2v_data.load(self.key)[:]`（`[Lt, num_disp, Np, Nc, Ne]`）。

**`CurrentElementalP2V(NdarrayFile, CurrentElemental)`** — `lattice/preset.py:838-861`
- 签名同上；shape `[Lt, num_disp, Np, Nc, Ne]`，默认后缀 `"_p2v.npy"`。

**`CurrentElementalP2P`**（不继承任何格式基类）— `lattice/preset.py:863-1021`
- 签名：`__init__(self, prefix: str, suffix: str = None)`（`lattice/preset.py:878-883`）— 唯一带真实默认值的 `suffix`；只存 prefix/suffix，`self.file = None; self.data = None`。无 elem/Ne/Np。
- `load(self, key: str, t: int, num_momentum: int = None) -> List[dict]`（`lattice/preset.py:885`）：**唯一非 lazy、同步物化的加载器**，一次只取一个时间切片：
  1. 优先 HDF5：`{prefix}{key}_p2p.h5`（条件：`suffix is None or suffix.endswith(".h5")` 且文件存在，`lattice/preset.py:890-894`）；`_load_hdf5`（`lattice/preset.py:962-1021`）要求 `num_momentum` 非 None（否则 `ValueError`，`lattice/preset.py:971-972`），按 `disp_{i}/t_{t}` 层级读 `attrs["type"]`，`identity` 或 `sparse(indices [N,2], values [N,3,3])`，并把每个 disp 的数据重复 `num_momentum` 份。
  2. 否则回退 npz：suffix 中 `.t???` 被 `re.sub` 替换为 `.t{t:03d}.`（`lattice/preset.py:903-905`），`numpy.load(allow_pickle=True)`；按 `type_{i}` 键数确定条目数，重建 `{"type": "identity"}` 或 `{"type": "sparse", "indices", "values"}` 字典列表（`lattice/preset.py:909-935`）。
  3. npz 路径的动量扩展：`num_momentum is not None` 且条目非空时，按 `num_disp = len(result)` 逐 disp 重复 `num_momentum` 次（`lattice/preset.py:937-955`）。
- 语义（docstring，`lattice/preset.py:864-876`）：p2p 与动量无关，所以每份 disp 数据为所有 momentum_idx 重复；`identity` 条目代表 `delta_{l,r} delta_{c,c'}`。

#### 一点/两点函数类

**`Jpsi2gammaBinary(BinaryFile, TwoPoint)`** — `lattice/preset.py:674-685`
- 签名：`__init__(self, prefix: str, suffix: str, shape: List[int] = [128, 2, 3, 4, 27, 128])`
- 元数据：`FileMetaData(shape, "<f8", 0)`（实数双精度），默认后缀 `".mesonspec.2pt.bin"`。

**`Jpsi2gammaNpy(NdarrayFile, TwoPoint)`** — `lattice/preset.py:790-799`
- 签名：`__init__(self, prefix: str, suffix: str)` — **无 shape 参数**，`TwoPoint.__init__(self, None)`，即 **`elem` 为 `None`**。默认后缀 `".2pt.npy"`。
- 由于 `NdarrayFileData` 从 npy header 读真实 shape/dtype（`lattice/filedata/ndarray.py:29-45`），`elem=None` 只使 handle 的声明 `shape/dtype` 为 `None`（`lattice/filedata/ndarray.py:13-14`）；任何读取 `handle.shape` 的下游会拿到 `None`。

**`OnePointNpy(NdarrayFile, OnePoint)`** — `lattice/preset.py:801-811`
- 签名：`__init__(self, prefix: str, suffix: str)`，同样 `OnePoint.__init__(self, None)`。默认后缀 `".1pt.npy"`。`lattice/preset.py:806` 注释 `[2, 123, 128]` 是唯一形状线索。

## ③ 数据结构与数组形状约定

### `FileMetaData` 与 `extra` 轴

`FileMetaData(shape, dtype, extra)`（`lattice/filedata/abstract.py:5-9`）中 `extra` 只在 QDP 路径有意义：`self.shape = elem.shape[elem.extra:]` 是**每条 record 的数据形状**，`self.extraShape = elem.shape[:elem.extra]` 是 record key 的轴（`lattice/filedata/timeslice.py:39-42`）。npy/binary/ILDG 路径 `extra=0`（V2P/P2V/OverlapMatrixNpy/EigenvectorNpy 等传了非零值，但 npy 实现不消费该字段）。

### 传播子形状约定（docstring 给定）

| 类 | 单文件 shape | timeslice 每文件 shape | 轴语义 |
|---|---|---|---|
| `PropagatorPSVNpy` / `PropagatorPSVTimeslicesNpy` | `[Lt(t_src), Lt(t_rel), Ns_snk, Ns_src, Np, Nc, Ne]` | `[Lt(t_rel), Ns_snk, Ns_src, Np, Nc, Ne]` | `PSV[t_src, t_rel, s_snk, s_src, p, c, e] = <eta_{x_p}, c \| S(t_src + t_rel; t_src) \| xi_e>`；左端 (sink) 是 (point, color)，右端 (source) 是 eigenvector（`lattice/preset.py:60-77`） |
| `PropagatorVSPNpy` / `PropagatorVSPTimeslicesNpy` | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` | `[Lt, Ns, Ns, Ne, Np, Nc]` | `VSP[t_src, t_rel, s_snk, s_src, e, p, c] = <xi_e \| S \| eta_{x_p}, c>`，与 PSV 方向相反（`lattice/preset.py:87-112`） |
| `PropagatorPSPNpy` / `PropagatorPSPTimeslicesNpy` | `[Lt, Lt, Ns, Ns, Np_snk, Nc_snk, Np_src, Nc_src]` | 7 维（少第一个时间轴） | `PSP[t_src, t_rel, s_snk, s_src, p_snk, c_snk, p_src, c_src] = <eta_{x_psnk}, c_snk \| S \| eta_{x_psrc}, c_src>`（`lattice/preset.py:116-134`） |

timeslice 文件名 `{cfg}.t{t_src:03d}.npy`；timeslice 文件存的是 `np.roll(..., -t_src)` 后的相对时间数组（`lattice/preset.py:620-627` 生成示例）。

### elemental / current elemental 形状约定

| 类 | 磁盘布局 | 对外呈现 |
|---|---|---|
| `ElementalBinary` / `ElementalNpy` | `(num_disp, num_mom, Lt, Ne, Ne)` | 同磁盘 |
| `CurrentElementalV2V` | `(Lt, num_disp, num_mom, Ne, Ne)` | `_View` 归一为 `(num_disp, num_mom, Lt, Ne, Ne)` |
| `CurrentElementalV2P` | `[Lt, num_disp, Ne, Np, Nc]` | 同磁盘 |
| `CurrentElementalP2V` | `[Lt, num_disp, Np, Nc, Ne]` | 同磁盘 |
| `CurrentElementalP2P` | HDF5 `disp_{i}/t_{t}/{indices,values}` 或 npz `type_{i}, indices_{i}, values_{i}` | `List[dict]`，长度 = `num_disp * num_momentum` |

### Lazy handle 协议

`FileData` 抽象基类只有 `shape/dtype/time_in_sec/size_in_byte` 属性和抽象方法 `__getitem__(key: Tuple[int])`（`lattice/filedata/abstract.py:12-20`）。`load()` 返回 handle，`handle[...]` 才触发 mmap + 读取 + `backend.asarray` 转换；`time_in_sec/size_in_byte` 由每次 `__getitem__` 累计，是内建性能计量（`lattice/filedata/ndarray.py:46-47`）。

## ④ 算法与流程

### 一次加载的生命周期（以 `ElementalNpy` 为例）

```
user = ElementalNpy(prefix, suffix, shape, Ne)   # __init__ 只存元数据 (lattice/preset.py:725-735)
h = user.load("2000")        # NdarrayFile.get_file_data: 文件名变化→新建 NdarrayFileData (lattice/filedata/ndarray.py:52-59)
arr = h[d, m]                # __getitem__: 解析 npy header→mmap→h[key].copy()→backend.asarray (lattice/filedata/ndarray.py:18-46)
```

### 各格式基类的读取算法

| 格式基类 | 读取流程 | 校验 | 精度处理 |
|---|---|---|---|
| `NdarrayFile` (`lattice/filedata/ndarray.py:18-46`) | 读 npy magic + header → 按 `mmap.ALLOCATIONGRANULARITY` 对齐 mmap → `file[key].copy()` → `backend.asarray` | magic、version ∈ {(1,0),(2,0)}、`not fortran_order`（`lattice/filedata/ndarray.py:28-32`）；实际 shape/dtype 来自 npy header | 保持磁盘 dtype |
| `NdarrayTimeslicesFile` (`lattice/filedata/ndarray.py:66-116`) | 同上，但切片 key 首维被弹出为时间切片号，`re.sub(r"\.t\?\?\?\.", f".t{tsrc_idx:03d}.", self.file)` 后打开对应文件（`lattice/filedata/ndarray.py:70-73`） | 同上 | 保持磁盘 dtype |
| `BinaryFile` (`lattice/filedata/binary.py:17-62`) | 用 `elem.shape/elem.dtype` 计算 stride，mmap 整文件取 `file[key].copy()` | **无**——完全信任 `FileMetaData` | 保持磁盘 dtype |
| `QDPLazyDiskMapObjFile` (`lattice/filedata/timeslice.py:102-129`) | 读 magic `XXXXQDPLazyDiskMapObjFileXXXX` + 版本 + XML 元数据 + `(record key → 偏移)` 表 → 按 `key[:extra]` 定位 record mmap | magic（`lattice/filedata/timeslice.py:111`）、`decay_dir == 3`（`lattice/filedata/timeslice.py:47`）、key 不在 offsets 表则 `IndexError`（`lattice/filedata/timeslice.py:71-72`） | **强制降精度 `astype("<c8")`**（complex64，`lattice/filedata/timeslice.py:96`） |
| `IldgFile` (`lattice/filedata/ildg.py:76-104`) | 扫描 LIME record 头（magic `b"\x45\x67\x89\xAB\x00\x01"`，8 字节对齐）→ 取 `ildg-binary-data` 偏移与 `ildg-format` XML → mmap 数据区 | `bytes == precision//8*2` 与 `prod(shape)*bytes == record 长度`（`lattice/filedata/ildg.py:35-36`）——**唯一在加载器内校验 shape 的格式** | **升精度 `astype("<c16")`**（complex128，`lattice/filedata/ildg.py:70`） |

### 统一缓存模式

`BinaryFile`/`NdarrayFile`/`NdarrayTimeslicesFile`/`IldgFile`/`QDPLazyDiskMapObjFile` 的 `get_file_data` 全部是同一模式：`if self.file != name: 打开并替换 self.data`（如 `lattice/filedata/ndarray.py:52-59`）。同一 loader 实例对同一 cfg 重复 `load()` 复用 handle，换 cfg 才重新打开。

### 与 generator / propagators 层的关系

- `lattice/generator/` 的生成器把 preset 角色类作为**输入端口类型**（如 `ElementalGenerator(..., gauge_field: GaugeField, eigenvector: Eigenvector, ...)`，`lattice/generator/elemental.py:11`）；运行时只依赖传入对象的 duck-typed 能力（`load(cfg)`、`calc(t)` 等）。典型流水线：generator 产出数据 → 另一侧用对应 preset 加载器读回校验（`test/test_elemental.py:30`、`test/test_perambulator.py:53`）。
- `lattice/propagators.py` 持有 preset 加载器实例：`Meson`（`lattice/propagators.py:188`）通过 `_available_extent`（`lattice/propagators.py:60`）从 loader 的 `Ne` 属性或 `data.shape` 最后两轴推断可用 Ne，`_normalize_extent`（`lattice/propagators.py:46-57`）校验 `0 <= usedNe <= available` 后切片（`lattice/propagators.py:279-281`）；`PropagatorWithCurrent`（`lattice/propagators.py:921`）校验 `p2v_data.Ne` 与 elemental 的 Ne 一致（`lattice/propagators.py:397-401`），读 `p2v_data.Np` 作可用 Np（`lattice/propagators.py:403`）。

## ⑤ 不变量与校验

1. **文件名公式**：`{prefix}{key}{suffix}`；timeslice 版再经 `.t???` → `.t%03d` 替换（`lattice/filedata/ndarray.py:71-73`；`CurrentElementalP2P` 的 npz 路径同理，`lattice/preset.py:904-905`）。
2. **lazy 协议**：除 `CurrentElementalP2P`（返回 `List[dict]`）和 `CurrentElementalV2V`（返回 `_View`）外，`load()` 返回 `FileData` lazy handle；物化靠 `__getitem__`（`test/test_propagator_psv.py:43-45` 断言 handle 不是 `np.ndarray`）。
3. **校验强度分层**：ILDG（shape/精度断言，`lattice/filedata/ildg.py:35-36`）> QDP（magic + `decay_dir == 3` + offsets 表成员检查，`lattice/filedata/timeslice.py:111`、`47`、`71-72`）> npy（magic/version/fortran_order，`lattice/filedata/ndarray.py:28-32`）> binary（无校验）。`lattice/preset.py` 全文件没有一处 `assert` 或参数校验。
4. **精度处理**：QDP 强制降 `"<c8"`（`lattice/filedata/timeslice.py:96`）；ILDG 升 `"<c16"`（`lattice/filedata/ildg.py:70`）；npy/binary 保持磁盘 dtype（实际 dtype 来自 npy header）。
5. **缓存语义**：同一 loader 实例对同一 cfg 的重复 `load()` 返回同一 handle；换 cfg 自动换文件（见 ④）。
6. **`.t???` 占位符契约**：所有 timeslice 加载器（`PerambulatorTimeslicesNpy`、`PropagatorVSPTimeslicesNpy`、`PropagatorPSPTimeslicesNpy`、`PropagatorPSVTimeslicesNpy`）的**默认 suffix 都不含 `.t???`**——默认状态下 timeslice 切片必失败（`FileNotFoundError`）；必须显式传含 `.t???` 的 suffix。该行为被 `test/test_propagator_psv.py:110-118` 写成契约测试。
7. **轴约定冲突点**：gauge 在 QDP 格式为 `[Lt,4,Vol,Nc,Nc]`（`lattice/preset.py:153`）、ILDG 为 `[Lt,Vol,4,Nc,Nc]`（`lattice/preset.py:689`）；elemental 在 `ElementalNpy` 磁盘为 `(disp,mom,Lt,Ne,Ne)`、`CurrentElementalV2V` 磁盘为 `(Lt,disp,mom,Ne,Ne)`（由 `_View` 归一）。
8. **`suffix` 形参无默认值但按 `None` 分支**：除 `CurrentElementalP2P.__init__(self, prefix, suffix: str = None)`（`lattice/preset.py:878`）外，其余加载器的 `suffix: str` 均无默认值，调用时必须显式传 `None` 才会启用内置默认后缀，直接省略会 `TypeError`（如 `lattice/preset.py:152-156`、`lattice/preset.py:201-208`）。
9. **`CurrentElementalV2V._View` 只接受 5 元组索引**：`[:]`、裸 int、短 tuple 均抛 `IndexError`（`lattice/preset.py:763-766`）。
10. **`_loader_signature` 不 stat timeslice 文件**：`lattice/propagators.py:161-168` 中 `if "?" not in suffix` 才解析文件名并 stat；带 `.t???` 的 loader 不产生文件级缓存身份。
11. **`EigenvectorNpy`/`Jpsi2gammaNpy`/`OnePointNpy` 的 `extra` 元数据不被 npy 实现消费**：实际 shape/dtype 以 npy header 为准，`FileMetaData` 只是声明值；`elem=None` 时 handle 的 `shape/dtype` 为 `None`（`lattice/filedata/ndarray.py:13-14`）。

## ⑥ 相关测试

| preset 类 | 测试 | 测什么 |
|---|---|---|
| `GaugeFieldIldg` | `test/test_load.py:24-33` | ILDG 加载结果与 npy/binary 参考数据逐项 norm 比较；验证 chroma shape `[Lt,Lz,Ly,Lx,Nd,Nc,Nc]` 下的切片（`gaugeIldg.load("")[1:3, :, 5:8, ...]`）与 transpose 一致性 |
| `EigenvectorNpy` | `test/test_eigenvector.py:39` | 作为 `EigenvectorGenerator` 结果的读回参考（相位自由度比较）；另在 `test/test_elemental.py:19`、`test/test_perambulator.py:27` 中作为 generator 输入 |
| `ElementalNpy` | `test/test_elemental.py:30`、`test/test_meson_spectrum.py:38`、`test/scripts/profile_vertex_map_timing_gap.py:165` | `ElementalGenerator` 输出读回校验；meson spectrum 流程标准输入 |
| `PerambulatorNpy` | `test/test_perambulator.py:53`（读回，含 `np.roll(-t)` 时间对齐约定 `test/test_perambulator.py:66`）、`test/scripts/t2pp_skeleton.py:35,42` | `PerambulatorGenerator` 输出读回校验 |
| `PerambulatorBinary` | 无直接测试；`example/gen_two_particle_corr.py:40-44`、`example/gen_two_particle_corr_mom.py:44` 使用 | — |
| `PropagatorPSVNpy` / `PropagatorPSVTimeslicesNpy` | `test/test_propagator_psv.py:44-134` | 完整契约测试：lazy handle 非 ndarray、`[:]` 物化、1-tuple 选时间片、timeslice `.t???` 占位符必要性（缺则 `FileNotFoundError`，`:110-118`）、裸 int key 报 `TypeError`（`:120-134`） |
| `CurrentElementalV2P/P2V/P2P`、`OverlapMatrixNpy` | `test/test_current_elemental.py:181` 起（`TestOverlapMatrixNpy` 等） | 导入可用性、shape 定义、P2P 的 HDF5 构造（注意测试中大量 `skipTest` 包裹） |
| `PointSourceNpy` | `test/scripts/test_calc_all_comparison.py:61`、`test/test_gauge_links_methods.py` 等 | 作为 current elemental 生成流程输入；坐标布局 `(Np,Lt,3)` 的语义由生成端 `test/test_sparsened_point.py` 保障 |
| `EigenvectorTimeSlice` | 无直接测试 | — |
| `CurrentElementalV2V` | 无测试、无使用（仓库内除定义与导出外零调用） | — |
| `GaugeFieldTimeSlice` / `GaugeFieldBinary` / `ElementalBinary` / `Jpsi2gamma*` / `OnePointNpy` | 无测试、无使用 | — |

已知边界（复核性备注，非结论）：

- `test/test_current_elemental.py` 中部分用例以 `skipTest` 包裹，实际执行覆盖可能受限。
- 除 `PropagatorPSVNpy`/`PropagatorPSVTimeslicesNpy` 外，其余 timeslice 加载器的 `.t???` 契约无各自的真实文件测试，但共享同一实现路径（`lattice/filedata/ndarray.py:66-116`）。
- 各加载器的默认 shape 常量（`[128,4,16³,3,3]` 等）对应一台 16³×128、Ne=70 的生产 lattice，多数默认值与实际用法轴序不一致（如 `EigenvectorNpy` 默认 Ne 轴在前、实际用 Lt 轴在前），只能当示例看待。
