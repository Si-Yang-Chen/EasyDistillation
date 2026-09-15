# 文件格式与 IO 契约（data-formats）

> 定位说明：本文是 EasyDistillation 全部磁盘数据格式的参考契约——按磁盘格式分节说明字节/轴布局、dtype 约定、命名规则、加载路径（lazy `FileData`/mmap）、保存函数与原子性、schema/manifest 校验，并给出 propagator/current elemental 的统一 layout 总表。

## ① 模块概览（含依赖关系）

IO 域由四层组成，自下而上：

| 层 | 模块 | 职责 |
|---|---|---|
| 协议层 | `lattice/filedata/abstract.py` | 定义 `FileMetaData`（shape/dtype/extra 三元组）、抽象 `FileData`（惰性切片句柄）与抽象工厂 `File` |
| 格式后端层 | `lattice/filedata/{binary,ildg,ndarray,timeslice}.py` | 四种磁盘格式的 mmap 惰性读取器 |
| 预设层 | `lattice/preset.py` | 「物理角色基类 × 格式后端」多重继承组合出约 25 个加载器，统一暴露 `load(key)` |
| 工件层 | `lattice/current_elemental.py`、`lattice/quark_diagram.py` | 内容寻址的 directed-current V2V 工件（manifest + npy）与 diagram 缓存（`backend.save`） |

依赖方向（均由 import 语句核实）：

- `lattice/preset.py:3-8` 依赖 `filedata` 的全部格式基类与抽象协议；具体格式类不经 `filedata/__init__.py` 导出（该包只导出 `File, FileData, FileMetaData`，见 `lattice/filedata/__init__.py`），由 `preset.py` 显式 import。
- `preset` 的 25 个公开名由 `lattice/__init__.py:4-31` 重导出，是模块的公开门面。
- 所有格式后端的 `__getitem__` 末尾调用 `get_backend()`（`lattice/backend.py:10-14`，backend 是 numpy 或 cupy 模块本身），返回值统一转为当前后端数组——这是 backend 抽象进入 IO 层的接缝。
- 消费端 `lattice/propagators.py` 依赖 loader 的公开属性 `prefix/suffix/elem(.shape,.dtype)/Ne/Np` 与 `load(key)`（见 `_loader_signature`，`lattice/propagators.py:149-177`）。

按磁盘格式的总览：

| 磁盘格式 | 后端类 | 代表加载器 | 布局特征 |
|---|---|---|---|
| `.npy` 单文件 | `NdarrayFile` | `PerambulatorNpy`、`ElementalNpy`、`PropagatorPSVNpy` 等 | numpy header + C-order 数据区，整文件 mmap |
| `.npy` 按时间片分片 | `NdarrayTimeslicesFile` | `PropagatorPSVTimeslicesNpy`、`PerambulatorTimeslicesNpy` 等 | 文件名含 `.t???` 占位符，每片少一个时间轴 |
| 裸二进制 | `BinaryFile` | `PerambulatorBinary`、`ElementalBinary`、`GaugeFieldBinary`、`Jpsi2gammaBinary` | 无 header，布局完全由 `FileMetaData` 约定 |
| ILDG/Lime | `IldgFile` | `GaugeFieldIldg` | Lime 记录框 + `ildg-format` XML + 单条 `ildg-binary-data` |
| QDP `QDPLazyDiskMapObjFile` | `QDPLazyDiskMapObjFile` | `GaugeFieldTimeSlice`、`EigenvectorTimeSlice` | 魔数串 + XML + 记录偏移表，按 record key 惰性读取 |
| HDF5 / npz | 无格式基类（自实现） | `CurrentElementalP2P` | 稀疏 p2p 数据，按 `(key, t)` 粒度加载 |
| 内容寻址工件 | 纯函数 | `save/load_directed_current_v2v` | manifest.json + `v2v-{sha256}.npy` |
| 图缓存生成物 | — | `quark_diagram` 的 `save_dir` | `{save_dir}/{hash(e)}`，由当前 backend 序列化 |

## ② 公开 API 参考

### 2.1 协议层（`lattice/filedata/abstract.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| `FileMetaData` | `__init__(self, shape: List[int], dtype: str = "<c16", extra: Any = None)` (`lattice/filedata/abstract.py:5-9`) | 纯元数据记录类。`extra` 的语义由具体后端解释：QDP 路径的 `extra` 是「前 `extra` 个轴用作 record key」（`lattice/filedata/timeslice.py:39-41`）；`EigenvectorNpy`/`OverlapMatrixNpy`/`CurrentElementalV2P`/`P2V` 传 `2` 但其 Ndarray 后端并不消费该值；binary/ildg/npy 单文件路径传 `0` |
| `FileData` | 抽象基类，类属性 `shape = None`、`dtype = None`、`time_in_sec = 0.0`、`size_in_byte = 0`；唯一抽象方法 `__getitem__(self, key: Tuple[int])` (`lattice/filedata/abstract.py:12-20`) | 惰性句柄协议：`load()` 不读盘，索引时才 mmap 取切片；`time_in_sec`/`size_in_byte` 是内建性能计量 |
| `File` | 抽象工厂，`get_file_data(self, name: str, elem: FileMetaData) -> FileData` (`lattice/filedata/abstract.py:23-26`) | 具体后端按文件路径缓存句柄：路径变化才重新解析，否则复用 |

### 2.2 `.npy` 单文件（`lattice/filedata/ndarray.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| `NdarrayFileData` | `__init__(self, file: str, elem: FileMetaData)` (`lattice/filedata/ndarray.py:10-15`)；`__getitem__(self, key)` (`lattice/filedata/ndarray.py:17-49`) | 手工解析 `.npy` header：magic（`numpy.lib.format.MAGIC_PREFIX` + 2 字节版本号，`lattice/filedata/ndarray.py:31`）、版本断言 `(1,0)/(2,0)`（`lattice/filedata/ndarray.py:34`）、用私有 API `numpy.lib.format._read_array_header` 取 `(shape, fortran_order, dtype)`（`lattice/filedata/ndarray.py:35`）、断言 C-order（`lattice/filedata/ndarray.py:36`）；数据区偏移 `self_offset = f.tell()`（`lattice/filedata/ndarray.py:37`），mmap 起点对齐 `mmap.ALLOCATIONGRANULARITY`（`lattice/filedata/ndarray.py:38-43`），`file[key].copy()` 后 `backend.asarray`（`lattice/filedata/ndarray.py:44`）。**实际切片用的 shape/dtype 来自文件 header，`elem.shape/dtype` 只作声明值** |
| `NdarrayFile` | `__init__(self)`；`get_file_data(self, name: str, elem: FileMetaData) -> NdarrayFileData` (`lattice/filedata/ndarray.py:52-62`) | 按 `self.file != name` 缓存句柄（`lattice/filedata/ndarray.py:58-61`） |

### 2.3 `.npy` 按时间片分片（`lattice/filedata/ndarray.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| `NdarrayTimeslicesFileData` | `__init__(self, file: str, elem: FileMetaData)`（`lattice/filedata/ndarray.py:64-71`）；`__getitem__(self, key)`（`lattice/filedata/ndarray.py:72-106`） | **key 的第一个分量被解释为时间片号并从索引中剥离**（`lattice/filedata/ndarray.py:79-80`，故裸 int key 会在解包处抛 `TypeError`）；文件名替换 `re.sub(r"\.t\?\?\?\.", f".t{tsrc_idx:03d}.", self.file)`（`lattice/filedata/ndarray.py:83`）——**传入的文件名必须含字面 `.t???` 占位符**，否则打开错误文件名抛 `FileNotFoundError`；其余 mmap 流程与单文件版相同（`lattice/filedata/ndarray.py:94-101`） |
| `NdarrayTimeslicesFile` | `get_file_data(self, name: str, elem: FileMetaData)`（`lattice/filedata/ndarray.py:109-116`） | 同上缓存模式；`__init__` 把句柄存入 `self.data`（`lattice/filedata/ndarray.py:112`） |

### 2.4 裸二进制（`lattice/filedata/binary.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| `BinaryFileData` | `__init__(self, file: str, elem: FileMetaData)`（`lattice/filedata/binary.py:18-25`）；`get_count(key)`（`lattice/filedata/binary.py:27-28`）；`get_offset(key)`（`lattice/filedata/binary.py:30-34`）；`__getitem__(key)`（`lattice/filedata/binary.py:36-62`） | **完全信任 `FileMetaData`**：C-order 步长 `stride`（`lattice/filedata/binary.py:22`）、字节数由 dtype 正则 `^[<>=]?[iufc](?P<bytes>\d+)$` 提取（`lattice/filedata/binary.py:23`）；`get_offset` = `sum(key_i * stride_i) * bytes`（`lattice/filedata/binary.py:30-34`）；`__getitem__` 对整个 `prod(shape) * bytes` 区域 mmap（`lattice/filedata/binary.py:53-55`）后对完整 shape 索引（`lattice/filedata/binary.py:59`）。文件无 header、无对齐、无字节序转换，dtype 串的 `<`/`>` 前缀语义交给 numpy memmap 解释 |
| `BinaryFile` | `get_file_data(self, name: str, elem: FileMetaData) -> BinaryFileData`（`lattice/filedata/binary.py:65-74`） | 按路径缓存 |

### 2.5 ILDG/Lime（`lattice/filedata/ildg.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| `IldgFileData` | `__init__(self, file: str, elem: FileMetaData, offset: Tuple[int], xmlTree: ET.ElementTree)`（`lattice/filedata/ildg.py:20-38`）；`__getitem__(key)`（`lattice/filedata/ildg.py:49-73`） | 从 XML 命名空间前缀下读 `lx/ly/lz/lt` 得 `latt_size`（`lattice/filedata/ildg.py:25-30`）；两条核心读断言：`self.bytes == precision//8*2`（precision 64 → 16 字节/元素即 `>c16`，precision 32 → 8 字节即 `>c8`，`lattice/filedata/ildg.py:35`）与 `prod(elem.shape) * bytes == ildg-binary-data 记录长度`（`lattice/filedata/ildg.py:36`）。`__getitem__` 从数据偏移 mmap（对齐 `mmap.ALLOCATIONGRANULARITY`，`lattice/filedata/ildg.py:61`），**返回值强制 `.astype("<c16")`（小端 complex128）**（`lattice/filedata/ildg.py:70`）。注意该类统计属性名为驼峰 `timeInSec/sizeInByte`（`lattice/filedata/ildg.py:37-38`），与基类蛇形命名不同 |
| `IldgFile` | `__init__(self)` 设置 `self.magic = b"\x45\x67\x89\xAB\x00\x01"`（`lattice/filedata/ildg.py:76-80`）；`read_meta_data(self, f: BufferedReader) -> (offset, xml_tree)`（`lattice/filedata/ildg.py:82-95`）；`get_file_data(...)`（`lattice/filedata/ildg.py:98-104`） | Lime 记录扫描：每条记录头 = 6 字节魔数断言（`lattice/filedata/ildg.py:86`）+ 8 字节大端 `>Q` 长度向上取整到 8 的倍数（`lattice/filedata/ildg.py:87`）+ 128 字节记录名（strip `\x00` 后 utf-8 解码，`lattice/filedata/ildg.py:88`），记录数据体按 length 跳过；循环至 EOF 或换行符 `b"\x0A"`（`lattice/filedata/ildg.py:84-92`）；元数据取记录名 `"ildg-binary-data"`（数据偏移/长度）与 `"ildg-format"`（XML）（`lattice/filedata/ildg.py:93-95`） |

### 2.6 QDP `QDPLazyDiskMapObjFile`（`lattice/filedata/timeslice.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| （模块级读取函数） | `read_str(f)`（4 字节大端 `>i` 长度 + utf-8 串，`lattice/filedata/timeslice.py:13-16`）、`read_tuple(f)`（4 字节长度 n + n/4 个大端 int32 键元组，`lattice/filedata/timeslice.py:18-23`）、`read_pos(f)`（16 字节 `>qq`，**取第二个 8 字节作偏移**，`lattice/filedata/timeslice.py:25-26`） | — |
| `QDPLazyDiskMapObjFileData` | `__init__(self, file, elem, offsets: Dict[Tuple[int], int], xml_tree)`（`lattice/filedata/timeslice.py:36-52`）；`__getitem__(key)`（`lattice/filedata/timeslice.py:65-99`） | `self.shape = elem.shape[elem.extra:]` 是**单条记录的数据形状**，`self.extraShape = elem.shape[:elem.extra]` 是 record key 的轴（`lattice/filedata/timeslice.py:39-41`）；XML 中解析 `lattSize` 与 `decay_dir` 并断言 `decay_dir == 3`（`lattice/filedata/timeslice.py:43-47`）。`__getitem__`：`key[:extra]` 必须在 offsets 表中否则 `IndexError`（`lattice/filedata/timeslice.py:71-72`）；按记录偏移对齐 mmap（`lattice/filedata/timeslice.py:86-95`），索引 `key[extra:]` 后**强制 `.astype("<c8")`（complex64）**（`lattice/filedata/timeslice.py:96`）——仓库中唯一在加载器内降精度的路径 |
| `QDPLazyDiskMapObjFile` | `__init__` 设置 `self.magic = "XXXXQDPLazyDiskMapObjFileXXXX"`（`lattice/filedata/timeslice.py:102-108`）；`read_meta_data(f) -> Dict[Tuple[int], int]`（`lattice/filedata/timeslice.py:110-121`）：断言 magic（`lattice/filedata/timeslice.py:111`）→ 4 字节大端 version → XML 元数据 → seek 到偏移表（`>I` num_record，每条记录 = `read_tuple` 键 + `read_pos` 的 8 字节偏移）；`get_file_data(...)`（`lattice/filedata/timeslice.py:123-127`） | — |

### 2.7 preset 角色基类（mixin，`lattice/preset.py`）

角色基类只持有元数据、不实现 IO（`deepcopy(elem)` 后保存维度参数）：

| 类 | 行 | `__init__` 签名 | 保存的属性 |
|---|---|---|---|
| `GaugeField` | `lattice/preset.py:11` | `(elem: FileMetaData)` | `elem` |
| `Eigenvector` | `lattice/preset.py:16` | `(elem: FileMetaData, eigenNum: int)` | `elem, Ne` |
| `Elemental` | `lattice/preset.py:22` | `(elem: FileMetaData, eigenNum: int)` | `elem, Ne` |
| `CurrentElemental` | `lattice/preset.py:28` | `(elem: FileMetaData, eigenNum: int, pointNum: int = None)` | `elem, Ne, Np` |
| `Perambulator` | `lattice/preset.py:41` | `(elem: FileMetaData, eigenNum: int)` | `elem, Ne` |
| `PointSource` | `lattice/preset.py:47` | `(elem: FileMetaData, Np: int)` | `elem, Np` |
| `OverlapMatrix` | `lattice/preset.py:53` | `(elem: FileMetaData)` | `elem`（Ne/Np 由子类补，见 `lattice/preset.py:262-263`） |
| `PropagatorPSV` | `lattice/preset.py:58` | `(elem: FileMetaData, Np: int, Ne: int)` | `elem, Np, Ne` |
| `PropagatorVSP` | `lattice/preset.py:85` | `(elem: FileMetaData, Np: int, Ne: int)` | `elem, Ne, Np` |
| `PropagatorPSP` | `lattice/preset.py:114` | `(elem: FileMetaData, Np_snk: int, Np_src: int)` | `elem, Np_snk, Np_src` |
| `OnePoint` | `lattice/preset.py:141` | `(elem: FileMetaData)` | `elem` |
| `TwoPoint` | `lattice/preset.py:146` | `(elem: FileMetaData)` | `elem` |

### 2.8 preset 具体加载器

统一模式：`load(self, key: str)` 实现为 `return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)`（例：`lattice/preset.py:160-162`、`lattice/preset.py:213-215`），返回 lazy `FileData` 句柄。继承方向统一为「格式基类在前、角色基类在后」（如 `class PerambulatorBinary(BinaryFile, Perambulator)`，`lattice/preset.py:200`）。

**`.npy` 单文件系**

| 加载器 | 行 | dtype | shape（磁盘轴序） | 默认 suffix |
|---|---|---|---|---|
| `EigenvectorNpy(NdarrayFile, Eigenvector)` | `lattice/preset.py:183-198` | `<c16`（extra=2 仅声明） | 默认 `[70, 128, 16**3, 3]`；测试实际用 `[Lt, Ne, Lz, Ly, Lx, Nc]`（t 轴优先） | `.lime.npy` |
| `PerambulatorNpy(NdarrayFile, Perambulator)` | `lattice/preset.py:217-232` | `<c8` | `[Lt, Lt, Ns, Ns, Ne, Ne]` | `.stout.n20.f0.12.nev70.peram` |
| `PointSourceNpy(NdarrayFile, PointSource)` | `lattice/preset.py:234-248` | `<i4` | `[Np, Lt, 3]`（zero-based `[x, y, z]` 坐标） | `.npy` |
| `OverlapMatrixNpy(NdarrayFile, OverlapMatrix)` | `lattice/preset.py:250-268` | `<c16`（extra=2） | `[Lt, Ne, Np, Nc]`，子类自存 `self.Ne/Np`（`lattice/preset.py:262-263`） | `.overlap_matrix.npy` |
| `PropagatorPSVNpy(NdarrayFile, PropagatorPSV)` | `lattice/preset.py:270-327` | `<c16`（dtype 参数可覆盖） | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `.npy` |
| `PropagatorVSPNpy(NdarrayFile, PropagatorVSP)` | `lattice/preset.py:329-386` | `<c16` | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` | `.npy` |
| `PropagatorPSPNpy(NdarrayFile, PropagatorPSP)` | `lattice/preset.py:441-498` | `<c16` | `[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]` | `.npy` |
| `ElementalNpy(NdarrayFile, Elemental)` | `lattice/preset.py:717-739` | `<c8` | `[num_disp, num_mom, Lt, Ne, Ne]` | `.stout.n20.f0.12.nev70.meson.npy` |
| `CurrentElementalV2V(NdarrayFile, Elemental)` | `lattice/preset.py:741-788` | `<c16` | **磁盘** `[Lt, num_disp, num_mom, Ne, Ne]`，`_View` 对外呈现 `[num_disp, num_mom, Lt, Ne, Ne]` | `_v2v.npy` |
| `Jpsi2gammaNpy(NdarrayFile, TwoPoint)` | `lattice/preset.py:790-799` | 由 npy header 决定 | `elem=None`（shape/dtype 由文件头决定） | `.2pt.npy` |
| `OnePointNpy(NdarrayFile, OnePoint)` | `lattice/preset.py:801-811` | 由 npy header 决定 | `elem=None` | `.1pt.npy` |
| `CurrentElementalV2P(NdarrayFile, CurrentElemental)` | `lattice/preset.py:813-836` | `<c16`（extra=2） | `[Lt, num_disp, Ne, Np, Nc]` | `_v2p.npy` |
| `CurrentElementalP2V(NdarrayFile, CurrentElemental)` | `lattice/preset.py:838-861` | `<c16`（extra=2） | `[Lt, num_disp, Np, Nc, Ne]` | `_p2v.npy` |

`CurrentElementalV2V` 的特殊之处：`load()` 返回的不是原始句柄而是内部 `_View`（`lattice/preset.py:754-768`），其 `shape` 对外呈现 `(num_disp, num_mom, Lt, Ne, Ne)`，`__getitem__` 把 `(disp, mom, t, e1, e2)` 重排为磁盘的 `(t, disp, mom, e1, e2)` 去索引底层 mmap；**要求 `isinstance(key, tuple) and len(key) >= 5`，否则 `IndexError`**——`view[:]`、`view[0]` 等常规切片全部不可用。注意其角色基类是 `Elemental` 而非 `CurrentElemental`（`lattice/preset.py:741`），`Ne` 属性即构造参数。

**`.npy` 按时间片分片系**（每文件少一个 t_src 轴；默认 suffix 均不含 `.t???` 占位符，见 ⑤）

| 加载器 | 行 | dtype | 全量 shape / 每片 shape | 默认 suffix |
|---|---|---|---|---|
| `PropagatorVSPTimeslicesNpy(NdarrayTimeslicesFile, PropagatorVSP)` | `lattice/preset.py:388-437` | `<c16` | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` / `[Lt, Ns, Ns, Ne, Np, Nc]` | `.npy` |
| `PropagatorPSPTimeslicesNpy(NdarrayTimeslicesFile, PropagatorPSP)` | `lattice/preset.py:500-551` | `<c16` | 8 维全量 / 7 维每片 | `.npy` |
| `PerambulatorTimeslicesNpy(NdarrayTimeslicesFile, Perambulator)` | `lattice/preset.py:553-571` | `<c8` | `[Lt, Lt, Ns, Ns, Ne, Ne]` / `[Lt, Ns, Ns, Ne, Ne]` | `.stout.n20.f0.12.nev70.peram` |
| `PropagatorPSVTimeslicesNpy(NdarrayTimeslicesFile, PropagatorPSV)` | `lattice/preset.py:574-655` | `<c16` | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` / `[Lt, Ns, Ns, Np, Nc, Ne]` | `.npy` |

`PropagatorPSVTimeslicesNpy` 的 docstring 给出了分片文件的生成约定（`lattice/preset.py:578-636`）：每片存 `np.roll(..., -t_src)` 后的相对时间数组，保存为 `f"{save_dir}/cfg_{cfg}.t{t_src:03d}.npy"`（`lattice/preset.py:636`）。

**裸二进制系**

| 加载器 | 行 | dtype | shape | 默认 suffix |
|---|---|---|---|---|
| `GaugeFieldBinary(BinaryFile, GaugeField)` | `lattice/preset.py:700-715` | `<f8`（dtype 参数可覆盖） | `[128, 16**3, 4, 3, 3]` | `.dat` |
| `PerambulatorBinary(BinaryFile, Perambulator)` | `lattice/preset.py:200-215` | `<c16` | `[128, 128, 4, 4, 70, 70]` = `[t_src, t_snk, Ns, Ns, Ne, Ne]` | `.stout.n20.f0.12.nev70.peram` |
| `ElementalBinary(BinaryFile, Elemental)` | `lattice/preset.py:657-672` | `<c16` | `[40, 27, 128, 70, 70]` = `[num_disp, num_mom, Lt, Ne, Ne]` | `.stout.n20.f0.12.nev70.meson` |
| `Jpsi2gammaBinary(BinaryFile, TwoPoint)` | `lattice/preset.py:674-685` | `<f8` | `[128, 2, 3, 4, 27, 128]` | `.mesonspec.2pt.bin` |

**ILDG / QDP 系**

| 加载器 | 行 | dtype | shape | 默认 suffix |
|---|---|---|---|---|
| `GaugeFieldIldg(IldgFile, GaugeField)` | `lattice/preset.py:687-698` | `>c16`（extra=0） | 默认 `[128, 16**3, 4, 3, 3]` 仅为占位；实际使用 7 维 chroma 顺序 `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]`（`test/test_load.py:15,23`） | `.lime` |
| `GaugeFieldTimeSlice(QDPLazyDiskMapObjFile, GaugeField)` | `lattice/preset.py:151-162` | `>c16`（extra=2） | `[128, 4, 16**3, 3, 3]`；前两轴 `(t, mu)` 是 record key，单条记录 `[Vol, Nc, Nc]` | `.stout.n20.f0.12.mod` |
| `EigenvectorTimeSlice(QDPLazyDiskMapObjFile, Eigenvector)` | `lattice/preset.py:164-181` | `>c8`（extra=2） | `[128, 70, 16**3, 3]`；record key `(t, e)`，单条记录 `[Vol, Nc]` | `.stout.n20.f0.12.laplace_eigs.3d.mod` |

注意 gauge 两种格式的第 2/3 轴互换：QDP 为 `[Lt, Nd, Vol, Nc, Nc]`、ILDG 为 `[Lt, Vol(Lz,Ly,Lx), Nd, Nc, Nc]`。

**HDF5/npz：`CurrentElementalP2P`**（`lattice/preset.py:863-1021`）

- `__init__(self, prefix: str, suffix: str = None)`（`lattice/preset.py:878-883`）：唯一的**非 lazy、不继承任何格式基类**的加载器；`self.file = None; self.data = None`。
- `load(self, key: str, t: int, num_momentum: int = None) -> List[dict]`（`lattice/preset.py:885-961`）：
  1. 优先 HDF5：`f"{prefix}{key}_p2p.h5"`（条件 `suffix is None or suffix.endswith(".h5")` 且文件存在，`lattice/preset.py:906-910`）；`_load_hdf5`（`lattice/preset.py:964-1021`）要求 `num_momentum` 非 None 否则 `ValueError`（`lattice/preset.py:980`），按 `disp_{i}/t_{t}` 层级读 `attrs["type"]`，每个 disp 的数据按 `num_momentum` 复制。
  2. 回退 npz：suffix 默认 `".t???.p2p.npz"`，`.t???` 被 `re.sub` 替换为 `.t{t:03d}.`（`lattice/preset.py:913-919`）；`numpy.load(filename, allow_pickle=True)`（`lattice/preset.py:926`）；按 `type_{i}` 键数确定条目数（`lattice/preset.py:930`），重建 `{"type": "identity"}` 或 `{"type": "sparse", "indices": [N,2], "values": [N,3,3]}` 字典列表。
  3. 动量扩展：`num_momentum` 非 None 时把 `num_disp` 条数据复制为 `num_disp * num_momentum` 条（`lattice/preset.py:950-957`）——p2p 与动量无关。
- 数据语义：`type='identity'` 无数据，表示 $\delta_{l,r}\delta_{c,c'}$；`sparse` 的 `indices` 为 `[N,2]` 的 (l, r) 对、`values` 为 `[N,3,3]`。

### 2.9 directed-current V2V 内容寻址工件（`lattice/current_elemental.py`）

| 符号 | 签名 | 说明 |
|---|---|---|
| schema 常量 | `CURRENT_V2V_ARTIFACT_SCHEMA = "lattice.current.directed-v2v-artifact/v1"`、`CURRENT_V2V_CONTRACTION_SCHEMA = "lattice.current.v2v-termwise-contraction/v1"`、`CURRENT_V2V_PAIR_CONTRACTION_SCHEMA = "lattice.current.v2v-term-pair-contraction/v1"`（`lattice/current_elemental.py:23-25`） | 工件 manifest / 项级收缩结果 / 顶点对收缩结果的 schema 标识 |
| `save_directed_current_v2v` | `(directory, raw, contract, *, configuration, momenta, gauge_source, eigenvector_source, overwrite=False) -> Path`（`lattice/current_elemental.py:96-193`） | `raw` 必须是仅含 `"v2v"` 的 Mapping，先经 `validate_current_raw_contract(require_temporal=True)` 校验；`sources` 记录 gauge/eigenvector 源文件的绝对路径 + 整文件 sha256；数据以内容寻址命名发布（见 ④）；返回 manifest.json 路径 |
| `load_directed_current_v2v` | `(path, *, expected_configuration=None, expected_gauge_sha256=None, expected_eigenvector_sha256=None, verify_sources=True, mmap_mode="r") -> dict`（`lattice/current_elemental.py:222-330`） | 全校验式加载，返回 `{"raw": {"v2v": ndarray}, "contract": validated, "manifest": ..., "manifest_path": ..., "verified_files": {...}}`；`mmap_mode` 只允许 `None` 或 `"r"`（`lattice/current_elemental.py:234-235`）；数据以 `np.load(..., allow_pickle=False, mmap_mode=mmap_mode)` 读取（`lattice/current_elemental.py:291`） |

raw V2V 数据契约由 `lattice/insertion/current.py:112-153` 的 `build_current_raw_contract` 定义：shape 恰为 `(8, Lt, momentum_count, used_ne, used_ne)`，轴名 `["direction", "time", "momentum", "sink_ne", "source_ne"]`（`lattice/insertion/current.py:140`），dtype 必须为复数，`direction` 轴的 8 个值来自 `DirectedCurrentBasis.DIRECTIONS`（0..7 = +x, +y, +z, -x, -y, -z, +t, -t，后四个带 dagger 标志，`lattice/insertion/gauge_link.py:137-146`）；manifest 的 `consumer.raw_axes` 把该时间轴命名为 `bar_time`（`lattice/current_elemental.py:163-169`），即 raw 的时间轴锚定在 bar 端点。

### 2.10 图缓存生成物（`lattice/quark_diagram.py`）

`compute_diagrams`/`compute_diagrams_multitime` 接受 `save_dir` 参数；计算完成后 `_replace_diagrams` 对每个已求值的 `Diagram` 执行 `backend.save(f"{save_dir}/{hash(e)}", diagram_list[e.value_pointer].value)`（`lattice/quark_diagram.py:3149-3154`）——文件名是 Diagram 表达式对象的 `hash(e)`（`Diagram.__hash__`，`lattice/quark_diagram.py:2129`）。`backend` 是当前 numpy/cupy 模块本身（`lattice/backend.py:10-30`），因此 numpy 后端下即 `numpy.save`（`.npy` 格式）；cupy 后端下的序列化格式由 cupy 决定，本仓库未验证其与 numpy 后端的互通性。对应的按名回读分支当前被禁用（`lattice/quark_diagram.py:2825` 的 `if save_dir is not None and False:`，其内 `backend.load(f"{save_dir}/{hash(e)}.npy")` 不可达）。

## ③ 数据结构与数组形状约定

### 3.1 命名公式与 `.t???` 占位符

- 所有 preset 加载器的文件名公式：`{prefix}{key}{suffix}`。
- timeslice 加载器（`NdarrayTimeslicesFile` 系与 `CurrentElementalP2P` 的 npz 分支）在切片时把文件名中的字面 `.t???.` 替换为 `.t{tsrc:03d}.`（`lattice/filedata/ndarray.py:83`、`lattice/preset.py:919`）——**构造这类加载器时 suffix 必须含 `.t???`**（`PropagatorPSVTimeslicesNpy` 的测试用例即显式传含占位符的 suffix，`test/test_propagator_psv.py:88-102`）。
- 三个 timeslice 加载器与 `PerambulatorTimeslicesNpy` 的默认 suffix 均不含 `.t???`（`lattice/preset.py:435`、`lattice/preset.py:547`、`lattice/preset.py:568`、`lattice/preset.py:651`），用默认 suffix 时时间片切片必失败。
- preset 各 `suffix` 形参均无默认值但代码按 `suffix is None` 分支给默认字符串——调用时必须显式传 `None` 才启用默认后缀，直接省略会 `TypeError`；唯一例外是 `CurrentElementalP2P.__init__` 的 `suffix: str = None`（`lattice/preset.py:878`）。

### 3.2 propagator / current elemental layout 总表

缩写：V = eigenvector（低模，`Ne` 个）、P = point source（`Np` 个点，各带 `Nc` 色）。四类 propagator 与四类 current elemental 共用同一套 V/P 字母。轴表中 `Ns = 4`（spin）、`Nc = 3`（color）（`lattice/constant.py`）。

| 类型 | 语义 | 全量磁盘轴序 | timeslice 每文件轴序 | 出处 |
|---|---|---|---|---|
| VSV（perambulator） | S_ij = ⟨ξ_i\|S\|ξ_j⟩ | `[Lt(t_src), Lt(Δt), Ns_snk, Ns_src, Ne, Ne]` | `[Lt(Δt), Ns, Ns, Ne, Ne]` | `lattice/preset.py:558-561`、`lattice/preset.py:201-204` |
| VSP | S_{i,xa} = ⟨ξ_i\|S\|η_{x,a}⟩ | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` | `[Lt, Ns, Ns, Ne, Np, Nc]` | `lattice/preset.py:342`、`lattice/preset.py:423-433` |
| PSV | S_{xa,i} = ⟨η_{x,a}\|S\|ξ_i⟩ | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` | `lattice/preset.py:283`、`lattice/preset.py:639-651` |
| PSP | S_{xa,yb} = ⟨η_{x,a}\|S\|η_{y,b}⟩ | `[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]` | 8 维去第 1 轴 | `lattice/preset.py:454,476-481`、`lattice/preset.py:535-547` |
| current elemental V2V | 电流顶点 V×V | 磁盘 `[Lt, num_disp, num_mom, Ne, Ne]`；`_View` 视图 `[num_disp, num_mom, Lt, Ne, Ne]` | — | `lattice/preset.py:754-768` |
| current elemental V2P | 电流顶点 V→P | `[Lt, num_disp, Ne, Np, Nc]` | — | `lattice/preset.py:815-818` |
| current elemental P2V | 电流顶点 P→V | `[Lt, num_disp, Np, Nc, Ne]` | — | `lattice/preset.py:840-843` |
| current elemental P2P | 电流顶点 P→P（稀疏） | HDF5 `disp_{i}/t_{t}`；npz `type_{i}/indices_{i}/values_{i}`，每条目按 momentum 复制 | npz 文件名 `.t{t:03d}.p2p.npz` | `lattice/preset.py:864-876` |

propagator 时间轴约定：全量数组第 0 轴 = t_source，第 1 轴 = 相对时间 Δt（通常 Dt < Lt）；消费端以 `(t_to - t_from) % Lt` 索引第 1 轴，并检查 `Δt` 未超出所存分片数（`lattice/propagators.py:802-821`）。

尾部 Ne/Np 轴位置由 `propagators.py` 的轴表固化（`lattice/propagators.py:101-107`，对 timeslice 单文件形状从尾部数）：

| role | Ne 轴 | Np 轴 |
|---|---|---|
| `vsv` | `(-2, -1)` | 无 |
| `vsp` | `(-3,)`（`Ne, Np, Nc`） | `(-2,)` |
| `psv` | `(-1,)`（`Np, Nc, Ne`） | `(-3,)` |
| `overlap` | `(-3,)`（`Ne, Np, Nc`，对应 `OverlapMatrixNpy` 的 `[Lt, Ne, Np, Nc]`） | `(-2,)` |
| `psp` | 无 | `(-4, -2)`（`Np_snk, Nc, Np_src, Nc`） |

meson/current elemental 的消费端轴序（与上述磁盘布局互证）：`ElementalNpy`/`_View` 数据按 `elemental_data[derivative_idx, momentum_idx, :, :usedNe, :usedNe]` 切片（`lattice/propagators.py:279-282`）；P2V 数据按 `p2v_full[:, gaugelink_idx, :usedNp, :, :usedNe]` 切片（轴 0=Lt、1=disp、2=Np、3=Nc、4=Ne，`lattice/propagators.py:530-532`）。

### 3.3 dtype 约定（complex64/complex128 差异）

| 磁盘 dtype | 格式 | 读取后 dtype | 说明 |
|---|---|---|---|
| `<c16`（小端 complex128） | npy（perambulator 二进制同构、elemental binary、propagator、current elemental） | 保持 `<c16` | npy 路径以 header 为准；binary 路径完全信任元数据 |
| `<c8`（complex64） | npy（`PerambulatorNpy`、`PerambulatorTimeslicesNpy`、`ElementalNpy`） | 保持 `<c8` | 同一物理量的 binary 版是 `<c16`、npy 版是 `<c8`（`lattice/preset.py:209` vs `lattice/preset.py:226`） |
| `>c16`（大端 complex128） | ILDG gauge | **强制转 `<c16`**（`lattice/filedata/ildg.py:70`） | 大端磁盘、小端内存 |
| `>c16` / `>c8` | QDP gauge/eigenvector | **强制转 `<c8`**（`lattice/filedata/timeslice.py:96`） | 无论磁盘精度，读出统一降为 complex64 |
| `<f8` | `GaugeFieldBinary`、`Jpsi2gammaBinary` | 保持 `<f8` | 实数双精度 |
| `<i4` | `PointSourceNpy` | 由 npy header 决定（元数据声明 `<i4`） | 整型坐标 |

## ④ 算法与流程

### 4.1 惰性加载生命周期（lazy `FileData` / mmap）

以 `ElementalNpy` 为例（其余 npy 系加载器同构）：

```
loader = ElementalNpy(prefix, suffix, shape, Ne)   # 仅存元数据, lattice/preset.py:725-735
h = loader.load("2000")     # get_file_data: 路径变化才新建句柄, lattice/filedata/ndarray.py:57-62
arr = h[d, m]               # __getitem__: 解析 header → mmap → copy() → backend.asarray
```

- `load()` 本身不读数据，索引句柄才触发 mmap + 读取；所有后端的 mmap 均为只读（`mmap.ACCESS_READ`），起点对齐 `mmap.ALLOCATIONGRANULARITY`（`lattice/filedata/ndarray.py:38`、`lattice/filedata/ildg.py:61`、`lattice/filedata/timeslice.py:87`）。
- 每次切片累计 `time_in_sec` 与 `size_in_byte`（`lattice/filedata/ndarray.py:47-48`）。
- 同一加载器实例对同一文件名重复 `load()` 复用同一句柄（各格式 `get_file_data` 的 `if self.file != name` 分支）。

### 4.2 保存函数与原子性

仓库内的磁盘写入路径有两类：

1. **directed-current V2V 工件**（`save_directed_current_v2v`，`lattice/current_elemental.py:96-193`）：
   - 数据：写临时文件 `.v2v-{uuid}.tmp` → `np.save(..., allow_pickle=False)` + flush + `os.fsync` → 计算整文件 sha256 → 内容寻址命名 `v2v-{sha256}.npy`（`lattice/current_elemental.py:129-137`）；若同名文件已存在则校验其哈希一致（内容寻址幂等），否则抛 `ValueError("existing content-addressed V2V file is corrupt")`；发布用 `os.replace`（`lattice/current_elemental.py:143`）；`finally` 清理残留临时文件。
   - manifest：字段集固定（含 `schema/version/configuration/data/raw_contract/momenta/sources/consumer`），`artifact_identity = sha256(canonical_json(manifest 去掉 identity 字段))`（`lattice/current_elemental.py:43-45,175`）；同样走临时文件 + fsync（`lattice/current_elemental.py:177-183`），`overwrite=True` 时 `os.replace`、否则 `os.link` 硬链接发布（已存在则 `FileExistsError`，`lattice/current_elemental.py:184-190`）。**manifest 是提交点**：它绑定 npy 字节、raw 契约、动量向量、configuration key 与输入文件哈希。
2. **diagram 缓存**（`lattice/quark_diagram.py:3149-3154`）：`backend.save(f"{save_dir}/{hash(e)}", value)`，文件不存在时才写；无临时文件/原子替换机制（同一 `hash(e)` 已存在则跳过写）。

### 4.3 `CurrentElementalP2P` 的格式选择流程

`load(key, t, num_momentum)`：HDF5 优先（suffix 为 `None` 或以 `.h5` 结尾且 `{prefix}{key}_p2p.h5` 存在）→ 否则 npz（`.t???` 替换为 `.t{t:03d}.`，`allow_pickle=True`）→ 两处文件都不存在抛 `FileNotFoundError`（`lattice/preset.py:906-924`）。返回物化后的 `List[dict]`，不走 backend、不做 mmap，一次只取一个时间片。

## ⑤ 不变量与校验

### 5.1 各格式的校验强度分层

| 格式 | 加载器内校验 | 出处 |
|---|---|---|
| ILDG | precision 与 dtype 字节数一致；`prod(shape) * bytes == ildg-binary-data 记录长度`（**唯一校验 shape 的格式**） | `lattice/filedata/ildg.py:35-36` |
| QDP | magic 字符串断言；`decay_dir == 3`（时间轴）；record key 必须在偏移表中（越界 `IndexError`） | `lattice/filedata/timeslice.py:111`、`lattice/filedata/timeslice.py:47`、`lattice/filedata/timeslice.py:71-72` |
| npy | magic；版本 `(1,0)/(2,0)`；C-order 断言（**shape/dtype 以文件头为准，`elem.shape` 不参与校验**） | `lattice/filedata/ndarray.py:31-36` |
| binary | 无任何校验：shape/dtype 全由调用方声明，声明错误会静默读出错误数据或 mmap 越界 | `lattice/filedata/binary.py:18-25,53-55` |
| preset.py 本身 | 全文件无一处 assert/参数校验，校验都在 filedata 层或消费方 | — |

消费端的交叉校验：`propagators.py` 的 `_loader_extents` 把 loader 声明的 `Ne`/`Np` 属性与 `elem.shape` 推断值比对，冲突抛 `ValueError`（`lattice/propagators.py:93-122`）；`_normalize_extent` 校验 `0 <= used <= available`（`lattice/propagators.py:46-56`）。

### 5.2 命名/后缀不变量

- timeslice 加载器的 suffix 必须含字面 `.t???`（见 ③3.1）；缺占位符时切片抛 `FileNotFoundError`，裸 int key 抛 `TypeError`（契约测试：`test/test_propagator_psv.py:104-135`）。
- `CurrentElementalV2V._View` 只接受长度 ≥ 5 的 tuple 索引（`lattice/preset.py:763-766`）。
- `CurrentElementalP2P` HDF5 分支必须显式传 `num_momentum`（`lattice/preset.py:980`）。

### 5.3 schema/manifest 校验（directed-current V2V 工件）

`load_directed_current_v2v` 的校验序列（`lattice/current_elemental.py:222-330`）：

1. manifest 字段集必须恰为 `_ARTIFACT_KEYS`（`schema, version, configuration, data, raw_contract, momenta, sources, consumer, artifact_identity`，`lattice/current_elemental.py:26-35`），schema/version 强校验（`lattice/current_elemental.py:212-214`）。
2. `artifact_identity` 重算比对——防 manifest 篡改（`lattice/current_elemental.py:216`）。
3. `configuration` 可与 `expected_configuration` 绑定（`lattice/current_elemental.py:240-242`）。
4. 数据元数据：`format == "npy"`、`allow_pickle is False`、文件名无目录成分且恰为 `v2v-{sha256}.npy`（内容寻址互锁）、整文件哈希与 manifest 一致（`lattice/current_elemental.py:246-262`）。
5. sources：gauge/eigenvector 各自绝对路径 + 64 位 hex sha256；`verify_sources=True` 时实读源文件比对；`expected_gauge_sha256`/`expected_eigenvector_sha256` 可选绑定（`lattice/current_elemental.py:263-289`）。
6. 用磁盘上的 `raw_contract` 重新 `validate_current_raw_contract(require_temporal=True)`（legacy spatial 契约被显式拒绝），并复核 momenta 数量与 consumer 块逐字段比对（`lattice/current_elemental.py:292-309`）。
7. TOCTOU 防护：末尾重读 manifest 的 sha256 与内容确认校验期间未变（`lattice/current_elemental.py:310`）。

## ⑥ 相关测试

| 测试 | 覆盖内容 |
|---|---|
| `test/test_load.py` | 同一规范场分别从 `.npy`、裸二进制、ILDG lime 读出并做逐元素 norm 比较，全部为 0 才 pass；固化 ILDG gauge 的 chroma 轴序 `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]`（`test/test_load.py:13-37`） |
| `test/test_propagator_psv.py` | `PropagatorPSVNpy`/`PropagatorPSVTimeslicesNpy` 契约：`load()` 返回 lazy 句柄而非 ndarray；`[:]` 物化；timeslice `.t???` 占位符缺失 → `FileNotFoundError`；裸 int key → `TypeError`；每片 shape `(Lt, Ns, Ns, Np, Ne)`（`test/test_propagator_psv.py:34-135`） |
| `test/test_current_elemental.py` | `CurrentElementalV2P/P2V/P2P`、`OverlapMatrixNpy` 的 shape/一致性/错误处理与 P2P 的 HDF5 构造 |
| `test/test_current_v2v_persistence.py` | directed-current V2V 工件的 round trip、源文件哈希绑定、数据/manifest 篡改拒绝、拒绝覆盖 |
| `test/test_sparsened_point.py` | `PointSourceNpy` 消费的点源坐标布局 `(num_points, Lt, 3)`、int32、`0 <= coord < L`（生成端校验） |
| `test/test_elemental.py`、`test/test_perambulator.py` | `ElementalNpy`/`PerambulatorNpy` 作为 generator 输出的读回校验（后者含 `np.roll(-t)` 时间对齐约定） |
