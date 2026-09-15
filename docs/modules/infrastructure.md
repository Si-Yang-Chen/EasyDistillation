# 基础设施层：backend / dispatch / base_types / constant / data / result_provenance / filedata

> 定位说明：本文描述 EasyDistillation 的基础设施层——数值后端选择（numpy/cupy）、MPI 任务分发、最小公共类型与物理常数、elemental 数据到收缩输入 φ 的构建、结果目录溯源 manifest 机制，以及惰性文件 I/O 后端族（binary/ILDG/.npy/QDP timeslice），它们构成全部计算模块的公共底座。

---

## ① 模块概览（含依赖关系）

| 模块 | 职责 | 关键符号 |
|---|---|---|
| `lattice/backend.py` | CPU/GPU 数值后端单例注册表与 pyquda (QUDA) 探测 | `get_backend`、`set_backend`、`check_QUDA`、`log_gpu_memory` |
| `lattice/dispatch.py` | MPI 进程间互斥的 cfg 任务队列 | `Dispatch`、`AtomicOpen`、`lock`/`unlock`、`combine` |
| `lattice/base_types.py` | flavor 字面量类型与夸克线追踪 Tag | `Flavor`、`Tag` |
| `lattice/constant.py` | 格点 QCD 维度常数 | `Nc`、`Ns`、`Nd` |
| `lattice/data.py` | Operator + elemental → (gamma, elemental) 收缩输入对 | `get_elemental_data` |
| `lattice/result_provenance.py` | 结果目录内容寻址 manifest（config_sha256 / run_id / 输出哈希） | `build_manifest`、`prepare_result_directory`、`validate_manifest`、`load_result_manifest`、`finalize_result` |
| `lattice/filedata/` | 惰性文件 I/O 后端族（切片时才读盘） | `File`、`FileData`、`FileMetaData` 及 binary/ildg/ndarray/timeslice 实现 |

### 依赖关系

- `backend` 被几乎全部计算模块使用：`lattice/data.py:8`、`lattice/correlator/one_particle.py:9`、全部 `lattice/filedata/*.py`（`lattice/filedata/binary.py:7`、`lattice/filedata/ildg.py:10`、`lattice/filedata/ndarray.py:6`、`lattice/filedata/timeslice.py:10`）以及 `lattice/generator/` 下的 elemental、eigenvector、perambulator、noisevector、density_perambulator、displacement_elemental 等。
- `filedata` 被 `lattice/preset.py:4-8`（全部具体格式类）、`lattice/data.py:4`、`lattice/correlator/dispersion_relation.py:4`、`lattice/correlator/one_particle.py:8` 使用（后三者只依赖 abstract 中的 `FileData`）。
- `Dispatch` 由 `lattice/__init__.py:2` 导出，仅被 example 脚本使用（`example/gen_twopt_matrix_mom.py:6`、`example/gen_two_particle_corr.py:15`、`example/gen_two_particle_corr_mom.py:17`），test/ 内无直接单测。
- `result_provenance` 仅被 `test/test_ne_provenance.py:13-19` 引用，`lattice/` 包内无其他使用者。
- `base_types`（`Tag`/`Flavor`）被 `lattice/flavor_structure.py:7`、`lattice/hadron.py:16`、`lattice/spatial_structure.py:15`、`lattice/quark_diagram.py:3242` 使用。
- `constant`（Nc/Ns/Nd）被 `lattice/__init__.py:54`、`lattice/propagators.py:29` 与 `lattice/generator/` 多个模块使用。
- `data.get_elemental_data` 被 `lattice/correlator/one_particle.py:7` 调用（一粒子关联函数的 φ 构造入口）。

顶层导出面：`lattice/__init__.py:1-2` 导出 `get_backend, set_backend, check_QUDA, log_gpu_memory` 与 `Dispatch`。`lattice/filedata/__init__.py:26-27` 只再导出抽象基类 `File, FileData, FileMetaData`（abstract 位于依赖图根部，不会引入循环 import）；具体格式类由 `lattice/preset.py:5-8` 直接 import。

---

## ② 公开 API 参考

### 2.1 lattice/backend.py

后端机制：以模块对象本身为"后端"——`set_backend` 把 numpy 或 cupy 模块赋给全局单例，调用方以 `backend = get_backend()` 取得后用统一数组 API（`backend.asarray`、`backend.prod` 等）写代码，实现 numpy/cupy 透明切换。使用例：`lattice/data.py:11`、`lattice/filedata/binary.py:39`、`lattice/correlator/one_particle.py:47`。

| 函数 | 签名 | 参数 | 返回值 | 说明 |
|---|---|---|---|---|
| `get_backend` | `get_backend() -> ModuleType`（`lattice/backend.py:10`） | 无 | 当前 backend 模块 | `_BACKEND` 为 None 时先隐式 `set_backend("numpy")`（`lattice/backend.py:12-13`） |
| `set_backend` | `set_backend(backend: Literal["numpy", "cupy"])`（`lattice/backend.py:17`） | backend：字符串或模块对象（非 str 时取 `backend.__name__`，`lattice/backend.py:19-20`），lower 后 assert 属于 `["numpy", "cupy"]`（`lattice/backend.py:22`） | None | 惰性 import numpy（`lattice/backend.py:23-26`）或 cupy（`lattice/backend.py:27-30`）并赋给全局 `_BACKEND` |
| `check_QUDA` | `check_QUDA(grid_size: List[int] = None, backend: Literal["cupy", "torch"] = "cupy", resource_path: str = None) -> bool`（`lattice/backend.py:39`） | grid_size：QUDA 网格；backend：pyquda 后端；resource_path：pyquda 资源目录 | bool：pyquda 是否成功初始化 | 首次调用（全局 `PYQUDA is None` 时，`lattice/backend.py:44-45`）尝试 `import pyquda; pyquda.init(...)`（`lattice/backend.py:48-49`），要求 `pyquda.__version__ >= 0.9.0` 否则抛 ImportError（`lattice/backend.py:51-55`）；ImportError/RuntimeError 只 warning 不上抛（`lattice/backend.py:56-60`）；成功置 `PYQUDA=True`（`lattice/backend.py:63`），失败置 False 并返回（`lattice/backend.py:64-67`）。进程级幂等单例 |
| `log_gpu_memory` | `log_gpu_memory(tag: str) -> None`（`lattice/backend.py:70`） | tag：日志条目标签 | None | cupy 后端查 `backend.cuda.runtime.memGetInfo()`（`lattice/backend.py:91`）；numpy 后端优先 psutil 查 RSS/VMS/百分比（`lattice/backend.py:97-114`），psutil 缺失时退化为 `gc.get_objects()` 逐对象 `sys.getsizeof` 求和的近似值（`lattice/backend.py:115-126`）；任何查询异常只 warning（`lattice/backend.py:127-128`） |

> 注：`lattice/backend.py:31-36` 存在被注释掉的 torch 分支，其后 `raise ValueError(R'backend must be "numpy", "cupy" or "torch"')` 因前面 `assert backend in ["numpy", "cupy"]`（`lattice/backend.py:22`）永不可达——本模块实际只承诺 numpy/cupy。

### 2.2 lattice/dispatch.py

| 符号 | 签名 | 说明 |
|---|---|---|
| `lock` / `unlock` | `lock(f: FileIO)` / `unlock(f: FileIO)`（`lattice/dispatch.py:11-30`） | Windows（`os.name == "nt"`）用 `msvcrt.locking(fd, LK_NBRLCK, 1)`，锁失败 sleep(0.1) 重试直到成功（`lattice/dispatch.py:13-18`）；POSIX 用 `fcntl.lockf(f, LOCK_EX)`，且仅当 `f.writable()` 才加锁（`lattice/dispatch.py:27-29`） |
| `class AtomicOpen` | 上下文管理器（`lattice/dispatch.py:34`） | `__init__` 打开文件、记录 `self.begin = self.file.tell()` 后加锁（`lattice/dispatch.py:35-38`）；`__exit__` flush + `os.fsync`、seek 回 begin、解锁并关闭（`lattice/dispatch.py:44-48`）；异常时返回 False（异常照常上抛），正常返回 True（`lattice/dispatch.py:49-52`） |
| `rand` | `rand(suffix: str = None) -> str`（`lattice/dispatch.py:55`） | suffix 为 None 时返回 8 位十六进制随机串，否则原样返回 suffix |
| `class Dispatch` | `Dispatch(filename: str, suffix: str = None, cfg_list: Iterable = None)`（`lattice/dispatch.py:63`） | 构造临时文件名 `f"{filename}.{rand(suffix)}.tmp"`（`lattice/dispatch.py:64`）；`self.comm = MPI.COMM_WORLD` 并记录 rank（`lattice/dispatch.py:66-67`）；各 rank 竞争 `AtomicOpen(tmp, "x+")` 独占创建，成功者把 `filename` 内容整份拷入，`FileExistsError` 则跳过（`lattice/dispatch.py:69-73`）。参数 `cfg_list` 未被使用（其上一行是被注释掉的代码，`lattice/dispatch.py:62`） |
| `Dispatch.__iter__` | `__iter__(self) -> Iterator[str]`（`lattice/dispatch.py:76`） | 见 ④ 算法与流程 |
| `Dispatch.clear` | `clear(self)`（`lattice/dispatch.py:103`） | 删除 tmp 文件 |
| `combine` | `combine(output: str, line: str)`（`lattice/dispatch.py:107`） | 以 `AtomicOpen(output, "a")` 原子追加一行结果 |

### 2.3 lattice/base_types.py

| 符号 | 定义 | 说明 |
|---|---|---|
| `Flavor` | `Literal["u", "d", "s", "c", "t", "b"]`（`lattice/base_types.py:3`） | 夸克味字符串字面量类型 |
| `class Tag` | `Tag(NamedTuple)`，字段 `tag: int`、`time: int`（`lattice/base_types.py:6-11`） | 自定义 `__eq__` 逐字段比较 tag 与 time（`lattice/base_types.py:10-11`）；NamedTuple 默认生成的 `__eq__` 功能上等价，且未覆写 `__hash__`（继承默认 hash）。用于把夸克线图中的夸克按 (编号, 时间) 唯一标识：`lattice/quark_diagram.py:3273` 以 `Tag(hadron_id * 3 + quark_id, factor.time)` 构造、`lattice/hadron.py:90` 以 `Tag(sub_expr.tag.tag, time)` 重建 |

### 2.4 lattice/constant.py

全文仅三行（`lattice/constant.py:1-3`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `Nc` | 3 | 色数（number of colors） |
| `Ns` | 4 | Dirac spin 维度 |
| `Nd` | 4 | 时空维数 |

### 2.5 lattice/data.py

| 函数 | 签名 | 参数 | 返回值 |
|---|---|---|---|
| `get_elemental_data` | `get_elemental_data(operators: List[Operator], elemental: FileData, usedNe: int) -> List[Tuple[ndarray, ndarray]]`（`lattice/data.py:7`） | operators：`lattice.insertion.Operator` 列表；elemental：elemental 文件的 `FileData` 句柄；usedNe：实际使用的 eigenvector 数（Ne 截断） | 每个 operator 一对 `(gamma 数组, elemental 数组)`，均经 `backend.asarray` 转换（`lattice/data.py:35`） |

### 2.6 lattice/result_provenance.py

常量（`lattice/result_provenance.py:14-32`）：`SCHEMA = "localized-blending-result-manifest/v1"`；`REQUIRED_FIELDS` 共 14 个字段：`schema, logical_test_id, configuration, source_ne, sink_ne, available_ne, attempt, code, inputs, parameters, config_sha256, run_id, result_directory, status`；`_HASH_RE` 匹配 64 位小写 hex，`_COMMIT_RE` 匹配 40 或 64 位 hex，`_SLUG_RE` 为 `[^A-Za-z0-9_.-]+`。模块全部为函数，无类。

| 函数 | 签名 | 说明 |
|---|---|---|
| `canonical_json` | `canonical_json(value: Any) -> str`（`lattice/result_provenance.py:33`） | `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)` 规范化序列化，是哈希稳定性的基础 |
| `sha256_bytes` | `sha256_bytes(value: bytes) -> str`（`lattice/result_provenance.py:37`） | 字节串的 SHA-256 hex |
| `sha256_file` | `sha256_file(path: str \| Path) -> str`（`lattice/result_provenance.py:41`） | 按 1MB 块流式计算文件 SHA-256 |
| `_slug` | `_slug(value) -> str`（`lattice/result_provenance.py:49`） | 非法字符替换为 `-`，剥掉首尾 `-._`；无安全字符抛 `ValueError` |
| `_normalize_ne` | `_normalize_ne(value, available_ne, name) -> int`（`lattice/result_provenance.py:56`） | bool/非 int 拒绝（TypeError）；`0 <= value <= available_ne` 否则 `ValueError` |
| `current_git_state` | `current_git_state(repository: str \| Path) -> dict[str, Any]`（`lattice/result_provenance.py:66`） | 对仓库执行 `git rev-parse HEAD`、`git status --porcelain=v1 --untracked-files=all`、`git diff HEAD --binary`、`git ls-files --others --exclude-standard`（`lattice/result_provenance.py:73-85`）；`worktree_sha256` = SHA-256(status 串 + 完整 binary diff + 排序后每个未跟踪文件的 (相对路径字节 + 文件内容哈希字节))（`lattice/result_provenance.py:86-96`）；返回 `{repository, commit, dirty, status_sha256, worktree_sha256}`（`lattice/result_provenance.py:97-100`）。git 调用失败因 `check=True` 直接抛异常 |
| `atomic_write_json` | `atomic_write_json(path, value) -> None`（`lattice/result_provenance.py:103`） | 先 mkdir 父目录；写临时文件 `.{name}.{pid}.tmp`（"x" 独占创建）→ flush + fsync → `replace` 原子替换（`lattice/result_provenance.py:105-113`）；finally 删除残留临时文件（`lattice/result_provenance.py:114-117`） |
| `build_manifest` | `build_manifest(*, output_root, logical_test_id, configuration, source_ne, sink_ne, available_ne, attempt, code: Mapping, inputs: Mapping[str, str], parameters: Mapping) -> dict`（`lattice/result_provenance.py:120`） | 全部 keyword-only。校验与组装流程见 ④；返回 `{"schema": SCHEMA, **config_payload, "config_sha256", "run_id", "result_directory", "status": "prepared"}`（`lattice/result_provenance.py:176-183`） |
| `prepare_result_directory` | `prepare_result_directory(manifest) -> Path`（`lattice/result_provenance.py:186`） | 先 `validate_manifest`；`mkdir(parents=True, exist_ok=False)`（并发下只有一个创建者成功）；写入 `manifest.json`；返回目录 |
| `validate_manifest` | `validate_manifest(manifest) -> dict`（`lattice/result_provenance.py:194`） | 缺字段/错 schema 抛错（`lattice/result_provenance.py:197-201`）；Ne 规范化并要求已规范化（`lattice/result_provenance.py:202-208`）；**重放 build_manifest**（用 `result_directory.parents[3]` 反推 output_root，`lattice/result_provenance.py:209-217`）比对 `config_sha256`、`run_id`、`result_directory` 三字段一致（`lattice/result_provenance.py:218-221`）；返回副本 |
| `load_result_manifest` | `load_result_manifest(result_directory, *, allow_legacy: bool = False) -> dict`（`lattice/result_provenance.py:226`） | 无 manifest.json 时：`allow_legacy=False` 抛 `ValueError("legacy result has no manifest and is not verified")`（`lattice/result_provenance.py:231-232`）；True 则返回 `{"schema": ".../legacy", "result_directory", "status": "legacy-unverified", "verified": False}`（`lattice/result_provenance.py:233-238`）。有 manifest：读 JSON → `validate_manifest` → 目录一致性校验（`lattice/result_provenance.py:239-243`）；`verified = status in {"completed", "validated"}`（`lattice/result_provenance.py:244`）；verified 时要求 `outputs` 非空且每个值为 64 位 hex，并逐文件 `sha256_file(output_path) == expected_hash` 校验，否则抛错（`lattice/result_provenance.py:245-254`） |
| `finalize_result` | `finalize_result(manifest, outputs: Mapping[str, str \| Path]) -> dict`（`lattice/result_provenance.py:258`） | validate → 必须已 prepare（manifest.json 存在，`lattice/result_provenance.py:261-263`）；每个输出名必须相对路径且无 `..`，实际文件必须位于 `directory/name`（`lattice/result_provenance.py:264-270`）；计算输出哈希，写回 `status="completed"` + 排序的 `outputs` 哈希表（`lattice/result_provenance.py:271-280`） |

### 2.7 lattice/filedata/

#### abstract.py（协议层，`lattice/filedata/abstract.py`）

| 类 | 定义 | 说明 |
|---|---|---|
| `FileMetaData` | `FileMetaData(shape: List[int], dtype: str = "<c16", extra: Any = None)`（`lattice/filedata/abstract.py:5`） | 属性 shape/dtype/extra。`extra` 语义由具体后端决定：binary/ildg/ndarray 单文件版不用；timeslice 用它标记"前 extra 个轴用于记录选择"（见 `lattice/filedata/timeslice.py:39-41`） |
| `FileData` | `FileData(metaclass=abc.ABCMeta)`（`lattice/filedata/abstract.py:12`） | 类属性 `shape = None`、`dtype = None`、`time_in_sec = 0.0`、`size_in_byte = 0`（读耗时/累计字节统计，`lattice/filedata/abstract.py:15-18`）；唯一抽象方法 `__getitem__(self, key: Tuple[int])`（`lattice/filedata/abstract.py:19`）——惰性切片接口，只有被索引时才读盘 |
| `File` | `File(metaclass=abc.ABCMeta)`（`lattice/filedata/abstract.py:23`） | 抽象方法 `get_file_data(name: str, elem: FileMetaData) -> FileData`：按文件路径 + 元数据产出 FileData 句柄 |

所有具体实现遵循同一模式：`File` 子类缓存 `self.file`/`self.data`，文件路径变化才重新解析元数据（`lattice/filedata/binary.py:70-74`、`lattice/filedata/ildg.py:98-104`）；`FileData.__getitem__` 内部用 `mmap` + `numpy.ndarray.__new__(numpy.memmap, ...)` 建只读内存映射，`file[key].copy()` 取出数据，再 `backend.asarray(...)` 转为当前后端数组。

#### binary.py（裸二进制 blob，`lattice/filedata/binary.py`）

| 类/方法 | 签名 | 说明 |
|---|---|---|
| `BinaryFileData` | `__init__(file: str, elem: FileMetaData)`（`lattice/filedata/binary.py:18`） | `self.stride = [prod(shape[i:]) for i in 1..len-1] + [1]`（C-order 步长，`lattice/filedata/binary.py:22`）；`self.bytes` 用正则 `^[<>=]?[iufc](?P<bytes>\d+)$` 从 dtype 提取字节数（`lattice/filedata/binary.py:23`） |
| `get_count` | `get_count(key: Tuple[int])`（`lattice/filedata/binary.py:27`） | `stride[len(key)-1]`：key 索引到第 k 层时剩余元素数 |
| `get_offset` | `get_offset(key: Tuple[int])`（`lattice/filedata/binary.py:30`） | `sum(key_i * stride_i) * bytes`：线性化字节偏移 |
| `__getitem__` | `__getitem__(key: Tuple[int])`（`lattice/filedata/binary.py:36`） | `mmap` 整个数组区域（大小 `prod(shape)*bytes`），memmap 上 `file[key].copy()` 后 `backend.asarray`；key 是对**完整 shape** 的索引（`lattice/filedata/binary.py:43-46`）；累加 `time_in_sec`/`size_in_byte`（`lattice/filedata/binary.py:53-54`） |
| `BinaryFile` | `get_file_data(name: str, elem: FileMetaData)`（`lattice/filedata/binary.py:65-74`） | 文件路径变化才新建 `BinaryFileData`，否则复用缓存句柄 |

字节级布局：文件 = 连续 C-order 元素序列，总字节数 = prod(shape) × bytes(dtype)，无 header、无对齐，字节序语义交给 numpy memmap 按 dtype 串（含 `<`/`>` 前缀）解释。

#### ildg.py（ILDG/Lime 规范场文件，`lattice/filedata/ildg.py`）

| 类/方法 | 签名 | 说明 |
|---|---|---|
| `IldgFileData` | `__init__(file: str, elem: FileMetaData, offset: Tuple[int], xmlTree: ET.ElementTree)`（`lattice/filedata/ildg.py:21`） | 从 XML 取命名空间前缀 `tag = re.match(r"\{.*\}", root.tag)`，读 `lx/ly/lz/lt` 得 `latt_size`（`lattice/filedata/ildg.py:26-31`）；两条核心断言：`assert self.bytes == int(precision)//8*2`（precision 64 → 16 字节/元素即 `>c16`，precision 32 → 8 字节即 `>c8`，`lattice/filedata/ildg.py:35`）、`assert prod(elem.shape) * bytes == offset[1]`（shape 总大小必须等于 `ildg-binary-data` 记录长度，`lattice/filedata/ildg.py:36`） |
| `get_count` / `get_offset` | 同 binary 版（`lattice/filedata/ildg.py:38-45`） | 线性化步长/偏移计算 |
| `__getitem__` | `__getitem__(key: Tuple[int])`（`lattice/filedata/ildg.py:49`） | mmap 起点对齐 `mmap.ALLOCATIONGRANULARITY`（`lattice/filedata/ildg.py:57-58`）；memmap 读取后 `.copy().astype("<c16")`——返回值强制转为小端 complex128（`lattice/filedata/ildg.py:66`）。注意此实现统计属性名为驼峰 `timeInSec`/`sizeInByte`（`lattice/filedata/ildg.py:33-34`），与 abstract.py:15-18 的蛇形命名不同 |
| `IldgFile` | `get_file_data(name: str, elem: FileMetaData)`（`lattice/filedata/ildg.py:76-104`） | 持有 magic `b"\x45\x67\x89\xAB\x00\x01"`（`lattice/filedata/ildg.py:78`）；`read_meta_data(f: BufferedReader) -> (offset, xml_tree)` 解析 Lime 记录；按路径缓存句柄 |

Lime 记录头布局（`lattice/filedata/ildg.py:82-96`）：8 字节魔数 `b"\x45\x67\x89\xAB\x00\x01"`（`lattice/filedata/ildg.py:86`）+ 8 字节大端 `>Q` 数据长度（向上取整到 8 的倍数，`lattice/filedata/ildg.py:87-88`）+ 128 字节记录名（补 `\x00`，strip 后 utf-8 解码，`lattice/filedata/ildg.py:89-90`），随后跳过 length 字节的数据体。循环读记录直到读空或读到文件尾换行 `b"\x0A"`（`lattice/filedata/ildg.py:84-92`）；元数据取自记录名 `"ildg-format"`（XML）与 `"ildg-binary-data"`（数据偏移，`lattice/filedata/ildg.py:94-96`）。

#### ndarray.py（.npy 文件：单文件版 + 分时间片版，`lattice/filedata/ndarray.py`）

| 类/方法 | 签名 | 说明 |
|---|---|---|
| `NdarrayFileData` | `__init__(file: str, elem: FileMetaData)`（`lattice/filedata/ndarray.py:9`）；`__getitem__(key)`（`lattice/filedata/ndarray.py:20`） | .npy 头解析契约：手工读 8 字节 magic（`numpy.lib.format.MAGIC_PREFIX` + 2 字节版本号），`assert version in [(1,0),(2,0)]`；用私有 API `numpy.lib.format._read_array_header(f, version)` 解析 `(shape, fortran_order, dtype)`；`assert not fortran_order`（只支持 C-order）（`lattice/filedata/ndarray.py:31-36`）。shape/dtype 完全以文件内 header 为准——elem.shape/dtype 在此实现中不参与读取（`lattice/filedata/ndarray.py:36-40`）。数据区偏移对齐 ALLOCATIONGRANULARITY 后 memmap 读取 |
| `NdarrayFile` | `get_file_data(name: str, elem: FileMetaData)`（`lattice/filedata/ndarray.py:52-62`） | 按路径缓存句柄 |
| `NdarrayTimeslicesFileData` | `__getitem__(key)`（`lattice/filedata/ndarray.py:65-107`） | `tsrc_idx, *rest = key`——第一个轴被解释为时间片号并从索引中剥离（`lattice/filedata/ndarray.py:74`）；裸 int key 会在解包处抛 TypeError。文件名替换 `re.sub(r"\.t\?\?\?\.", f".t{tsrc_idx:03d}.", self.file)`（`lattice/filedata/ndarray.py:76-77`）——调用方的文件名必须含字面 `.t???.` 占位符。其余 mmap 流程同单文件版 |
| `NdarrayTimeslicesFile` | `get_file_data(name: str, elem: FileMetaData)`（`lattice/filedata/ndarray.py:109-118`） | 按路径缓存句柄。注意 `__init__` 的类型注解写的是 `self.data: NdarrayFileData`（`lattice/filedata/ndarray.py:112`），运行时实际存的是 `NdarrayTimeslicesFileData`（`lattice/filedata/ndarray.py:116`），注解与实现不一致 |

#### timeslice.py（QDP 惰性磁盘映射文件，`lattice/filedata/timeslice.py`）

| 类/方法 | 签名 | 说明 |
|---|---|---|
| `read_str` | `read_str(f: BufferedReader) -> str`（`lattice/filedata/timeslice.py:13`） | 4 字节大端 `>i` 长度 + 该长度字节 utf-8 字符串 |
| `read_tuple` | `read_tuple(f: BufferedReader) -> Tuple[int]`（`lattice/filedata/timeslice.py:18`） | 4 字节长度 n + n/4 个大端 int32 组成的键元组 |
| `read_pos` | `read_pos(f: BufferedReader) -> int`（`lattice/filedata/timeslice.py:25`） | 16 字节 `>qq`，取第二个 8 字节作为偏移（8 字节对齐填充 + 8 字节 int64 偏移） |
| `QDPLazyDiskMapObjFileData` | `__init__(file, elem: FileMetaData, offsets: Dict[Tuple[int], int], xml_tree)`（`lattice/filedata/timeslice.py:36`） | `self.shape = elem.shape[elem.extra:]`、`self.extraShape = elem.shape[:elem.extra]`（`lattice/filedata/timeslice.py:39-41`）——前 `elem.extra` 个轴用于查 offsets 字典，其余轴是单条记录内的 shape。XML 中解析 `lattSize` 与 `decay_dir`，`assert decay_dir == 3`（衰减方向必须为 t 轴，`lattice/filedata/timeslice.py:43-47`） |
| `get_count` | `get_count(key: Tuple[int])`（`lattice/filedata/timeslice.py:53`） | `key == ()` 特例返回 `prod(self.shape)`；否则同 binary 版 |
| `__getitem__` | `__getitem__(key: Tuple[int])`（`lattice/filedata/timeslice.py:60`） | `key[:self.extra]` 必须在 offsets 中，否则 `raise IndexError`（`lattice/filedata/timeslice.py:71-73`）；以记录偏移对齐 mmap，读取 `self.shape` 形状的记录数据后索引 `key[self.extra:]`，`copy().astype("<c8")`——返回值强制转 complex64（`lattice/filedata/timeslice.py:83-90`） |
| `QDPLazyDiskMapObjFile` | `read_meta_data(f: BufferedReader) -> (offsets, xml_tree)`（`lattice/filedata/timeslice.py:102-129`） | 文件头：`assert self.magic == read_str(f)`，magic = `"XXXXQDPLazyDiskMapObjFileXXXX"`（`lattice/filedata/timeslice.py:102,111`）；4 字节大端 `>i` version（`lattice/filedata/timeslice.py:112`）；一段 XML 字符串（`lattice/filedata/timeslice.py:113`）；`f.seek(read_pos(f))` 跳到偏移表（`lattice/filedata/timeslice.py:114`）；4 字节 `>I` num_record（`lattice/filedata/timeslice.py:115`）；每条记录 = `read_tuple` 键 + `read_pos` 的偏移（`lattice/filedata/timeslice.py:117-118`），构建 `offsets: Dict[Tuple[int], int]` |

---

## ③ 数据结构与数组形状约定

### elemental（供 `data.get_elemental_data` 消费）

- elemental 的 `FileData` 磁盘布局前两轴为 (displacement, momentum)，随后为 (Lt, Ne, Ne)：`get_elemental_data` 按 `elemental[derivative_idx, momentum_idx, :, :usedNe, :usedNe]` 切片（`lattice/data.py:27-29`），得到 shape `(Lt, usedNe, usedNe)` 的数组。
- `Operator.parts` 契约（`lattice/insertion/__init__.py:147-180`）：偶数位为 gamma 索引，奇数位为 `[coefficient * derivative_coeff, derivative_idx, momentum, profile]` 4 元组列表（`lattice/insertion/__init__.py:163-172`），与 `lattice/data.py:21-26` 的解包一致。
- `get_elemental_data` 返回值：按每个 operator 一个元素堆叠为 `List[Tuple[ndarray, ndarray]]`（`lattice/data.py:35`）。

### 各文件后端的 shape/dtype 约定（由 `lattice/preset.py` 中的加载器默认参数给出）

| 后端 | 代表加载器（preset） | dtype | shape 约定 | 出处 |
|---|---|---|---|---|
| binary | `PerambulatorBinary` | `<c16` | [Lt,Lt,Ns,Ns,Ne,Ne] = [128,128,4,4,70,70] | `lattice/preset.py:200-215` |
| ndarray | `PerambulatorNpy` | `<c8` | 同上 | `lattice/preset.py:217-232` |
| ndarray（时间片） | `PerambulatorTimeslicesNpy` | `<c8` | 全局 [Lt,Lt,4,4,Ne,Ne]，每片 [Lt,4,4,Ne,Ne]，key 首轴为时间片 | `lattice/preset.py:553-571` |
| ildg | `GaugeFieldIldg` | `>c16` | 默认 [128,16³,4,3,3]；测试按 [Lt,Lz,Ly,Lx,Nd,Nc,Nc]（chroma 约定）传入 | `lattice/preset.py:687-693`；`test/test_load.py:23-24` |
| timeslice | `GaugeFieldTimeSlice` | `>c16`, extra=2 | FileMetaData([128,4,16³,3,3], ">c16", 2)：前 2 轴查 offsets，单条记录 shape [4,16³,3,3] = (Nd, Vol3, Nc, Nc) | `lattice/preset.py:151-161` |
| timeslice | `EigenvectorTimeSlice` | `>c8`, extra=2 | ([128,70,16³,3], ">c8", 2) | `lattice/preset.py:164-181` |
| ndarray | `EigenvectorNpy` | `<c16`, extra=2 | [Ne,Lt,Vol3,3] = [70,128,16³,3] | `lattice/preset.py:183-197` |
| ndarray | `ElementalNpy` | `<c8` | [num_disp,num_mom,Lt,Ne,Ne] = [4,123,128,70,70] | `lattice/preset.py:717-738` |
| ndarray | `CurrentElementalV2V` | `<c16` | 磁盘 [Lt,num_disp,num_mom,Ne,Ne]，`_View` 视图重排为 [disp,mom,Lt,Ne,Ne]（非 5 元组 key 抛 IndexError） | `lattice/preset.py:741-788`（`_View`：`lattice/preset.py:754-767`） |
| ndarray | `CurrentElementalV2P` / `P2V` | `<c16`, extra=2 | [Lt,num_disp,Ne,Np,Nc] / [Lt,num_disp,Np,Nc,Ne] | `lattice/preset.py:813-862` |
| ndarray/-ts | `PropagatorPSVNpy`、`PropagatorVSPNpy`、`PropagatorPSPNpy` 及各自 Timeslices 版 | `<c16` | PSV 全量 [Lt,Lt,Ns,Ns,Np,Nc,Ne]，每片 [Lt,Ns,Ns,Np,Nc,Ne] | `lattice/preset.py:270-327, 329-439, 441-551` |

轴语义补充：`lattice/constant.py` 只定义数值，数组轴语义由使用方决定，例如 `test/test_load.py:16` 的 `reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)`。

### result_provenance 的 manifest 结构

manifest 是一个 dict，必需字段见 `REQUIRED_FIELDS`（`lattice/result_provenance.py:15-29`）；其中 `code` 子对象必须包含 `{repository, commit, dirty, status_sha256, worktree_sha256}`（`lattice/result_provenance.py:142-155`）；`config_sha256` 是对规范化 `config_payload` 的 SHA-256，`run_id = config_sha256[:16]`（`lattice/result_provenance.py:167-169`）。

---

## ④ 算法与流程

### 4.1 Dispatch 的 MPI 工作队列

1. `__init__`：各 rank 竞争以 `"x+"` 独占创建 tmp 文件（`{filename}.{rand}.tmp`），胜利者把 cfg 列表文件内容整份拷入（`lattice/dispatch.py:69-73`）。
2. `__iter__` 循环：rank 0 在 `AtomicOpen` 独占锁内 readlines 后弹出首行并回写 + truncate（弹出即消费，`lattice/dispatch.py:80-88`）；其他 rank 置 `line = "dummy"` 仅作广播占位（`lattice/dispatch.py:89-90`）；`comm.bcast(line, root=0)` 广播（`lattice/dispatch.py:91`）。
3. 广播后仍等于 `"dummy"` 说明广播失败，抛 `ValueError`（`lattice/dispatch.py:92-93`）；`None`（文件已空）时打印 "Dispatch: finish all!" 并结束（`lattice/dispatch.py:94-96`）；空行跳过（`lattice/dispatch.py:97-98`）；否则 `yield line`（`lattice/dispatch.py:99`）。

MPI 多进程运行同一脚本时每个 cfg 只被一个进程领取，使用例见 `example/gen_twopt_matrix_mom.py:57-60`。

### 4.2 get_elemental_data：Operator → (gamma, elemental) 对

1. 取 `backend = get_backend()` 与 `gamma` 函数（惰性 import，`lattice/data.py:8-9`）；初始化 `cache`，键为 `(derivative_idx, momentum_idx)` 二元组（`lattice/data.py:12`）。
2. 对每个 operator：`parts = operator.parts`，成对遍历（`len(parts)//2` 次，`lattice/data.py:16-18`）：偶数位 `parts[2i]` 是 gamma 编号，经 `gamma(...)` 转矩阵；奇数位 `parts[2i+1]` 是 elemental 项列表。
3. 每个 elemental 项解包 `elemental_coeff, derivative_idx, momentum_idx, _profile = elemental_part[j]`（profile 为溯源元数据，收缩不需要，`lattice/data.py:21-26`）。
4. 以 `(derivative_idx, momentum_idx)` 为键缓存 elemental 切片（`lattice/data.py:27-29`）。
5. 同一 (gamma, elemental) 对内多系数项线性叠加：首项 `elemental_coeff * slice`，后续 `+= elemental_coeff * slice`（`lattice/data.py:30-33`），即 φ_γ = Σ_j c_j · O_(d_j,m_j)[:usedNe, :usedNe]。
6. 每个 operator 产出 `(backend.asarray(ret_gamma), backend.asarray(ret_elemental))`（`lattice/data.py:35`）。

### 4.3 result_provenance 的三层溯源

1. **build_manifest 校验链**（`lattice/result_provenance.py:132-155`）：`source_ne/sink_ne ∈ [0, available_ne]`；`attempt` 为正整数；`inputs` 值必须是 64 位小写 hex；`code.commit` 匹配 40/64 hex；`code` 必含五个必需键且两个哈希字段合法。
2. **内容寻址**：组装 `config_payload`（`logical_test_id, configuration` 字符串化, `source_ne, sink_ne, available_ne, attempt, code, inputs` 排序, `parameters`，`lattice/result_provenance.py:156-166`）→ `config_sha256 = sha256(canonical_json(config_payload))` → `run_id = config_sha256[:16]`（`lattice/result_provenance.py:167-169`）。
3. **目录编码**：`output_root / slug(logical_test_id) / "cfg-{slug(configuration)}" / "srcNe-{source_ne}_snkNe-{sink_ne}" / "attempt-{attempt:02d}-{run_id}"`（`lattice/result_provenance.py:170-175`）——目录路径本身编码全部区分维度，天然避免不同 run 互相覆盖。
4. **代码溯源**：`code` 字段由 `current_git_state` 生成，含 commit、dirty 标志、status 哈希与含未跟踪文件内容的 `worktree_sha256`（`lattice/result_provenance.py:66-100`）。
5. **产物溯源**：`finalize_result` 登记每个输出的 SHA-256（`lattice/result_provenance.py:271-280`）；`load_result_manifest` 对 verified 结果重算逐文件哈希校验（`lattice/result_provenance.py:245-254`）。
6. **篡改检测**：`validate_manifest` 通过重放 `build_manifest` 比对 `config_sha256`/`run_id`/`result_directory` 三字段，任何 payload 字段被篡改都会导致目录名或哈希不匹配（`lattice/result_provenance.py:209-221`）。

### 4.4 惰性读盘的公共模式

所有 `FileData.__getitem__` 的共同流程：打开文件 → `mmap.mmap(fd, 大小, access=ACCESS_READ, offset=对齐后的 start)` → `numpy.ndarray.__new__(numpy.memmap, shape=..., dtype=..., buffer=mm, offset=...)` 建只读内存映射视图 → `file[key].copy()` 物化数据 → `backend.asarray(...)` 转为当前后端数组 → 累加 `time_in_sec`/`size_in_byte` 统计。mmap 起点统一对齐 `mmap.ALLOCATIONGRANULARITY`（`lattice/filedata/ildg.py:57-58`、`lattice/filedata/ndarray.py:41-42`、`lattice/filedata/timeslice.py:78-79`）。

---

## ⑤ 不变量与校验

| # | 不变量 | 出处 |
|---|---|---|
| 1 | `set_backend` 只接受 numpy/cupy（assert）；`get_backend` 首次调用隐式选 numpy | `lattice/backend.py:12-13, 22` |
| 2 | `check_QUDA` 是幂等的进程级单例；失败不抛出、返回 False | `lattice/backend.py:44-67` |
| 3 | Dispatch 的任务行是"读即消费"（pop + truncate），配合独占锁保证 MPI 下不重复；`bcast` 哨兵 `"dummy"` 既是占位也是错误检测 | `lattice/dispatch.py:80-93` |
| 4 | `AtomicOpen.__exit__` 先 flush + fsync 再解锁关闭；异常时返回 False 让异常上抛 | `lattice/dispatch.py:44-52` |
| 5 | `Tag.__eq__` 逐字段比较 tag 与 time；未覆写 `__hash__`（NamedTuple 默认 hash 可用） | `lattice/base_types.py:10-11` |
| 6 | `get_elemental_data`：elemental 前两轴为 (derivative, momentum)，切片后保留 (Lt, usedNe, usedNe)；`Operator.parts` 为 2k/2k+1 交替结构 | `lattice/data.py:16-29`；`lattice/insertion/__init__.py:163-172` |
| 7 | result_provenance：所有 manifest 必须通过"重放 build_manifest"的字段一致性校验；输出必须相对路径且落在结果目录内；legacy 结果默认拒绝（`allow_legacy=True` 才以 `verified=False` 放行） | `lattice/result_provenance.py:209-221, 264-270, 231-238` |
| 8 | binary：文件为无 header 的 C-order 裸序列，总字节数 = prod(shape) × bytes(dtype) | `lattice/filedata/binary.py:22-23, 43-46` |
| 9 | ILDG：precision 与 dtype 字节数断言、数据长度与 shape 乘积断言；读取结果强制 `<c16` | `lattice/filedata/ildg.py:35-36, 66` |
| 10 | .npy：版本 (1,0)/(2,0)、C-order 断言；shape/dtype 以文件 header 为准 | `lattice/filedata/ndarray.py:31-36` |
| 11 | .npy 时间片版：文件名必须含 `.t???.` 占位符；key 首轴为时间片且必须以元组传入 | `lattice/filedata/ndarray.py:74-77`；`test/test_propagator_psv.py:104-135` |
| 12 | QDPLazyDiskMapObjFile：magic 字符串断言、`decay_dir == 3` 断言、offsets 键必须匹配前 extra 轴否则 IndexError、读取结果强制 `<c8` | `lattice/filedata/timeslice.py:47, 71-73, 90, 111` |
| 13 | filedata 所有读取均经 `backend.asarray` 转换为当前后端数组（backend 抽象进入 I/O 层的接缝） | `lattice/filedata/binary.py:50-51`、`lattice/filedata/ildg.py:66`、`lattice/filedata/ndarray.py:46`、`lattice/filedata/timeslice.py:88-89` |

---

## ⑥ 相关测试

| 测试 | 覆盖内容 | 出处 |
|---|---|---|
| `test/test_load.py` | `set_backend("numpy")` 后，把同一规范场分别从 `.npy`（`reshape(Nd,Lt,Lz,Ly,Lx,Nc,Nc)`）、裸二进制（`backend.fromfile`）与 ILDG lime 读出并做三项 norm 比较，全部为 0 才 pass——直接测试 filedata 三种后端的一致性契约与 ILDG 的 chroma 轴序 | `test/test_load.py:16-36` |
| `test/test_propagator_psv.py` | 测试 `PropagatorPSVNpy`（含无 Dirac 轴的简化 shape round-trip，:59-74）与 `PropagatorPSVTimeslicesNpy`：(a) `.t???` 占位符必须存在否则 FileNotFoundError（:104-119）；(b) 1-tuple key `(t_src,)` 选时间片、返回 shape `(Lt,Ns,Ns,Np,Ne)`（:99-102）；(c) 裸 int key 抛 TypeError（:121-135）——正是 `NdarrayTimeslicesFileData.__getitem__` 的 `tsrc_idx, *rest = key` 行为 | `test/test_propagator_psv.py` |
| `test/test_ne_provenance.py` | 完整覆盖 result_provenance：目录隔离与 manifest 字段完备性（`test_result_directories_are_isolated_and_manifest_complete`，:218-245）、并发 prepare 恰好一个创建者（`test_concurrent_prepare_allows_exactly_one_writer`，:247-261）、目录冲突与内容篡改检测（:281-315）、不同参数 → 不同 run_id（`test_parameters_change_run_identity`，:383-387）、legacy 结果的拒绝与放行（:295-304）、`current_git_state` 的 worktree 身份随 git 内容变化（:317-381） | `test/test_ne_provenance.py` |
| 物理测试套件 | `GaugeFieldIldg`/`EigenvectorNpy`/`ElementalNpy` 被约 10 个物理测试用作规范场/本征矢输入（`test/test_displacement_elemental.py:11`、`test/test_eigenvector.py:25`、`test/test_elemental.py:11`、`test/test_gauge_links_methods.py:16`、`test/test_perambulator*.py` 等）——filedata 是整个测试套件的公共 I/O 底座 | 各 test 文件 |
| 无专属单测的符号 | `Dispatch`、`AtomicOpen`、`combine` 无专属单测（grep 全 test/ 无命中）；`backend.set_backend` 被多测试首行调用（如 `test/test_load.py:8`、`test/test_ne_provenance.py:22`）；`base_types.Tag` 间接经 hadron/quark_diagram 被物理测试覆盖 | — |

最小验证命令（来自分析底稿建议）：

```bash
python test/test_load.py                      # numpy backend + binary/ildg/npy 三后端读取一致性
python -m pytest test/test_propagator_psv.py test/test_ne_provenance.py -q  # 时间片加载契约 + 溯源 manifest 全流程
```
