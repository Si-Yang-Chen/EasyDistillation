# Propagator 句柄层：`lattice/propagators.py` 与 `lattice/current_elemental.py`

> 定位说明：本文档描述 contraction 层的数据句柄域——`lattice/propagators.py` 把 preset 的懒加载器（perambulator、elemental、overlap matrix 等）包装成 quark-diagram contraction 消费的带索引视图（VSV/VSP/PSV/PSP propagator 与 Meson/Current 顶点）；`lattice/current_elemental.py` 提供 directed current 的 V2V 工件持久化与 term-wise 收缩函数。

## ① 模块概览（含依赖关系）

### 1.1 模块职责

| 模块 | 行数 | 职责 |
|---|---|---|
| `lattice/propagators.py` | 2344 | propagator/meson/current 句柄：每个 `load()` 派生一个 identity（key + Ne/Np 范围 + loader 签名 + 数据签名），identity 变化时释放派生缓存；对 contraction 暴露 `get/get_v2p/get_p2v/get_p2p` 与高模投影接口（`propagators.py:1-13`） |
| `lattice/current_elemental.py` | 721 | "Persistent directed-current V2V artifacts and term-wise VSV contraction"（`current_elemental.py:1`）：内容寻址工件保存/加载（`save/load_directed_current_v2v`）与两个纯函数式收缩（`contract_directed_current_v2v` / `contract_directed_current_pair_v2v`） |

### 1.2 分工与依赖

- `propagators.py` 是 `lattice.quark_diagram` 的实现细节，外部应经 `quark_diagram` 或 `lattice` 导入（`propagators.py:11-13`）；`quark_diagram.py:20-27` re-export 全部 6 个公开类。分工（依据 import 单向与求值调用方向）：propagators.py 管"数据从哪来、怎么缓存"，quark_diagram.py 管"拓扑怎么连、怎么收缩"（`quark_diagram.py:1730-1866` 调 propagator `get_*`）。
- `propagators.py` import：标准库 `gc, hashlib, logging, os, typing.Dict`、`numpy`、`opt_einsum.contract`；仓库内 `lattice.constant.Nc`（`propagators.py:30`，导入后实际未使用——`Current.get_p2p` 内硬编码 `Nc = 3`（`propagators.py:664`）；grep 确认全文件仅这两处 Nc）；`.preset`（`Perambulator, PropagatorVSP/PSV/PSP, OverlapMatrix, CurrentElementalV2P/P2V/P2P`）；`.backend`（`get_backend, log_gpu_memory`）；延迟导入 `lattice.insertion.gamma.gamma`（`propagators.py:271, 460, 825, 1309`）、`lattice.insertion.GaugeLink`（`propagators.py:432, 462`）。
- `current_elemental.py` import：标准库（hashlib/json/os/pathlib/uuid/typing/numbers）+ numpy；仓库内仅 `.insertion.current` 的 `Current, resolve_current_term_endpoints, resolve_current_term_spin, resolve_directed_current_raw, validate_current_raw_contract`（`current_elemental.py:11-18`）。被 `lattice/__init__.py:55` re-export；测试引用 `test/test_current_v2v_contraction.py:4`、`test/test_current_v2v_persistence.py:7`、`test/test_conserved_charge_v2v.py:14`。
- `lattice/quark_diagram.py`（含 propagators.py 的 `Current` 类）不使用 current_elemental.py 的收缩函数——两条 V2V 路径（legacy `Current` 类 vs 本模块 term-wise 函数）在当前代码中互不调用（grep 确认）。

### 1.3 缩写与 layout 词汇表

V = eigenvector（低模，Ne 个，下标 i/j）；P = point source（Np 个点，每点带 color Nc，下标 x/y 与 a/c…）。`_loader_extents` 的轴表以代码形式固化了各 layout（`propagators.py:100-106`）：

| 缩写 | 语义 | 数据尾部轴（timeslice 单文件，负轴从尾部数） | 出处 |
|---|---|---|---|
| VSV (perambulator) | eigen→eigen，S_ij = \<xi_i\|S\|xi_j\> | Ne 占最后两轴；全量 `[Lt(t_src), Lt(Δt), Ns_snk, Ns_src, Ne, Ne]` | `preset.py:41-44, 200-231, 553-572` |
| VSP | eigen→point，S_{i,xa} = \<xi_i\|S\|eta_{x,a}\> | Ne 在倒数第 3、Np 在倒数第 2（Ne,Np,Nc） | `preset.py:85-112, 329-386, 388-437` |
| PSV | point→eigen，S_{xa,i} = \<eta_{x,a}\|S\|xi_i\> | Ne 在最后、Np 在倒数第 3（Np,Nc,Ne） | `preset.py:58-83, 574-655` |
| PSP | point→point，S_{xa,yb} = \<eta_{x,a}\|S\|eta_{y,b}\> | 无 Ne、Np 占倒数第 4 与倒数第 2（Np_snk,Nc,Np_src,Nc） | `preset.py:114-139, 441-551` |
| overlap | `OverlapMatrixNpy`，M_{jx,a} = \<xi_j\|eta_{x,a}\> | 同 VSP（Ne,Np,Nc），shape `[Lt, Ne, Np, Nc]` | `preset.py:250-264` |

时间轴约定：全量数组第 0 轴 = t_source（或被缓存锚定的那个整数时间），第 1 轴 = 相对时间 Δt（0..Dt-1，通常 Dt<Lt）；`get` 族统一用 `(t_to - t_from) % Lt` 索引第 1 轴（`propagators.py:802-821, 1376, 1460, 1522, 1680` 等）。

## ② 公开 API 参考

### 2.1 内部校验辅助（propagators.py，缓存身份机制的底座）

| 函数 | 位置 | 说明 |
|---|---|---|
| `_normalize_extent(value, available, name)` | `propagators.py:46-56` | `value=None` 规范化为 `available`；校验为非 bool 整数且 `0 <= value <= available`，否则 `TypeError`/`ValueError`；返回 int |
| `_available_extent(loader, data, attribute, axes)` | `propagators.py:60-82` | 从 loader 声明属性（`Ne`/`Np`）与数据 shape 指定轴取 extent；两者冲突抛 `ValueError`（"conflicts with data shape"）；都没有返回 None |
| `_metadata_shape(loader)` | `propagators.py:86-91` | 优先 `loader.shape`，否则 `loader.elem.shape`，必须是 tuple/list |
| `_loader_extents(loader, role)` | `propagators.py:93-127` | 按 §1.3 轴表从元数据 shape 推断 (Ne, Np)；loader 声明与 shape 推断冲突抛 `ValueError` |
| `_data_signature(data)` | `propagators.py:124-147` | `(模块名, 类型名, shape, dtype, cache_version/version, 文件状态, ndarray SHA-256 内容哈希)`——缓存身份的数据部分 |
| `_loader_signature(loader, key)` | `propagators.py:149-177` | `(loader 类型, id(loader), prefix, suffix, elem.shape, elem.dtype, Ne, Np, cache_version, 文件路径, 文件(size,mtime))`——含 `id(loader)`，loader 对象身份变化也会使缓存失效 |
| `_operator_signature(operator)` | `propagators.py:180-181` | `sha256(repr(operator.parts))`——算符结构变化使 Meson 缓存失效 |
| `_validate_current_extents` | `propagators.py:1070-1094` | 对 `PropagatorWithCurrent` 的 5 个输入逐一 `_loader_extents`，任何两个声明的 Ne（或 Np）不一致 → `ValueError("propagator inputs have inconsistent Ne/Np")`；usedNe/usedNp 规范化 |

### 2.2 `Particle`（`propagators.py:184-185`）

空标记基类：`class Particle: pass`。

### 2.3 `Meson(Particle)`（`propagators.py:188-324`）

包装 meson elemental 加载器与一个算符，在 `load()` 时把算符 gamma 结构与 elemental 数据合成为低模 V2V 顶点缓存。

```python
def __init__(self, elemental, operator, source) -> None   # propagators.py:189
```

| 属性 | 含义 |
|---|---|
| `elemental` | elemental 加载器（`load(key)` 返回 shape `(num_disp, num_mom, Lt, Ne, Ne)`；也兼容 `CurrentElementalV2V` 的 `_View`，`preset.py:741-788`） |
| `operator` | 算符对象，须有 `.parts`（结构见 §4.1） |
| `dagger` | **由构造参数 `source` 直接赋值**（`propagators.py:193`；`test/test_propagator_with_current.py:270` 验证） |
| `outward/inward` | 均为 1（`propagators.py:194-195`） |
| `smeared` | `True`（`propagators.py:196`；`Current` 子类改为 False） |
| `cache` | dict 键为 `(derivative_idx, momentum_idx)` 元组（`propagators.py:277-286`） |

| 方法 | 位置 | 签名与说明 |
|---|---|---|
| `load` | `propagators.py:239-259` | `load(self, key, usedNe: int = None)`。加载 elemental → `_available_extent(..., "Ne", (-2,-1))` → `_normalize_extent` → identity = `(key, normalized_ne, _loader_signature, _data_signature, _operator_signature, bool(self.dagger))`；identity 变化才释放并 `_make_cache()` |
| `get` | `propagators.py:302-324` | `get(self, t)`。t 为 int → `[Ns, Ns, Ne, Ne]`；t 为数组 → `[t_len, Ns, Ns, Ne, Ne]`。实现 `contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])`（数组 t 为 `"xij,xtab->tijab"`）。**dagger 与非 dagger 分支的 contraction 字符串逐字相同——dagger 语义完全由 `_make_cache` 决定，`get` 里没有额外共轭**（`propagators.py:313-324`） |
| `release` / `__enter__/__exit__/__del__` | `propagators.py:216-230` | `_release_resources` 清空 `elemental_data`、`cache`、identity，并 `gc.collect()` + 后端内存池 `free_all_blocks()`（若有，`propagators.py:204-215`） |

### 2.4 `Current(Meson)`（`propagators.py:327-739`）

在 `Meson` 的低模 V2V 之外，围绕一个插入 current 组合 V2P/P2V/P2P 数据；P2P 是稀疏数据、按需（per-t）从文件加载。

```python
def __init__(self, elemental, operator, source,
             v2p_data: "CurrentElementalV2P" = None,
             p2v_data: "CurrentElementalP2V" = None,
             p2p_data: "CurrentElementalP2P" = None,
             debug: bool = False) -> None   # propagators.py:327-363
```

- `smeared = False`；`v2p_data/p2v_data/p2p_data` 为 preset 加载器（V2P shape `[Lt, num_disp, Ne, Np, Nc]`，`preset.py:813-836`；P2V shape `[Lt, num_disp, Np, Nc, Ne]`，`preset.py:838-861`；P2P 为稀疏 loader，`preset.py:863-930`）。
- `vsp_cache/psv_cache/psp_cache` 等属性在 `Current` 里初始化但主要被 `PropagatorWithCurrent` 写入——`Current` 自身只填 `cache（即 cache_v2v）/cache_v2p/cache_p2v/cache_p2p/p2p_loaded`（grep 确认写入点）。

| 方法 | 位置 | 说明 |
|---|---|---|
| `load` | `propagators.py:389-423` | `load(self, key, usedNe: int = None, usedNp: int = None)`。强制 `p2v_data` 存在（否则 `ValueError("p2v_data must be provided for Current integration")`）且其 Ne 与 elemental 一致（`propagators.py:396-404`）；`available_np = self.p2v_data.Np`；`self.Lt = elemental_data.shape[2]`（`propagators.py:411`）；identity 含 elemental 与 p2v/p2p 的 loader/数据签名、算符签名、dagger |
| `get_v2p` | `propagators.py:610-633` | `get_v2p(self, t)` → `[Ns, Ns, Ne, Np, Nc]`（int t）或 `[t_len, …]`。实现 `contract("xij,xepa->ijepa", cache[0], cache[2][:, t])` |
| `get_p2v` | `propagators.py:635-656` | → `[Ns, Ns, Np, Nc, Ne]`。实现 `contract("xij,xpae->ijpae", cache[0], cache[3][:, t])` |
| `get_p2p` | `propagators.py:658-739` | → `[Ns, Ns, Np, Nc, Np, Nc]`（仅 int t；数组 t 抛 `NotImplementedError`，`propagators.py:737-739`）。稀疏→稠密展开见 §4.3 |

### 2.5 `Propagator`（`propagators.py:742-861`）

标准 perambulator（VSV）的两时间访问器：以一个整数时间为锚缓存 perambulator 时间片与其 gamma(15) dagger 版本，返回相对时间片。

```python
def __init__(self, perambulator, Lt) -> None   # propagators.py:743
def load(self, key, usedNe: int = None)        # propagators.py:783-800
def get(self, t_source, t_sink)                # propagators.py:823-861
```

- `perambulator`：`Perambulator` 系加载器（`load(key)` → `[Lt, Lt, Ns, Ns, Ne, Ne]`，`preset.py:200-231`；timeslice 版 `[Lt(Δt), Ns, Ns, Ne, Ne]`×Lt 拼装，`preset.py:553-572`）。
- `get(t_source, t_sink)`：t_source/t_sink 至少一个为 int，否则 `ValueError("At least t_source or t_sink should be int")`（`propagators.py:859-862`）。以 int 时间 t 缓存 `cache = perambulator_data[t, :, :, :, :usedNe, :usedNe]`（shape `[Dt, Ns, Ns, Ne, Ne]`）并构造 `cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), cache.conj(), gamma(15))`（`propagators.py:836-841`）。锚是 t_source 时返回 `cache[(t_sink - t_source) % Lt]`；锚是 t_sink 时返回 `cache_dagger[(t_source - t_sink) % Lt]`（反向传播）——同一缓存槽在锚时间上双向复用（`propagators.py:842-848`）。
- `_relative_time_index(cache, t_from, t_to)`（`propagators.py:802-821`）：强制 `rel = (t_to - t_from) % Lt` 落在存储的 `Dt = cache.shape[0]` 内，越界抛 `IndexError`（错误信息说明 timeslice 文件只存 `[0, Dt)` 的 Δt；GPU 数组先 `.get()` 再检查）。

### 2.6 `PropagatorLocal(Propagator)`（`propagators.py:863-919`）

局部（同点源-汇）特化：`load` 后 `_make_cache` 取每个 t_source 的零相对时间片 `cache[t_source] = perambulator_data[t_source, 0, :, :, :usedNe, :usedNe]`（`propagators.py:906-911`）；`get(t_source, t_sink)` 断言 `t_source == t_sink`（int 或数组一致，`propagators.py:913-918`，断言消息 "You cannot use PropagatorLocal here"），返回 `cache[t_source]`。

### 2.7 `PropagatorWithCurrent(Propagator)`（`propagators.py:921-2344`）

组合 VSV/VSP/PSV/PSP + OverlapMatrix 五种输入，统一两时间访问接口与三个高模投影版本；仅给 vsv 时行为与 `Propagator` 完全一致（`propagators.py:921-938` docstring）。

```python
def __init__(self, vsv: Perambulator = None, vsp: PropagatorVSP = None,
             psv: PropagatorPSV = None, psp: PropagatorPSP = None,
             overlap_matrix: "OverlapMatrix" = None, Lt: int = None,
             debug: bool = False)   # propagators.py:939-1008
```

- vsv 为 None 时以 dummy 调 `super().__init__(None, Lt)`（仍可只用 VSP/PSV/PSP 图）。
- 两时间缓存命名约定（注释 `propagators.py:995-1000`）：`vsp_cache`（VSP 排序）与 `vsp_dagger`（=PSV 排序）；`psv_cache`（PSV 排序）与 `psv_dagger`（=VSP 排序）；`psp_cache` 与 `psp_dagger`（点轴互换）。高模缓存：`tilde_S_vsp_cache(+dagger)`、`tilde_S_psv_cache(+dagger)`、`tilde_S_psp_cache(+dagger)` 与 `_last_psp_highmode`（`propagators.py:1000-1008`）。

| 方法 | 位置 | 签名与输出形状 |
|---|---|---|
| `load` | `propagators.py:1096-1264` | `load(self, key, usedNe: int = None, usedNp: int = None)`。先 `_validate_current_extents`（`propagators.py:1070-1094`）；identity = `(key, normalized_ne, normalized_np, 五个 loader 的 _loader_signature 元组)`（`propagators.py:1109-1114`）；identity 变化时逐项 try/except 装载；`overlap_matrix` 为 None 直接 `ValueError("overlap_matrix must be provided")`（`propagators.py:1206-1207`）；全部成功后才写 `self.key/usedNe/usedNp/_cache_identity`（`propagators.py:1227-1231`） |
| `get` | `propagators.py:1266-1292` | VSV；`perambulator_data` 存在则委托 `super().get`，否则 `ValueError("VSV propagator not provided but is required for this diagram")` |
| `get_VSP` | `propagators.py:1361-1511` | `get_VSP(self, t_source, t_sink, cache=True)` → `[Ns, Ns, Ne, Np, Nc]`（int/int）或 `[t, Ns, Ns, …]`；dagger 时 PSV 序 |
| `get_PSV` | `propagators.py:1513-1662` | → `[Ns, Ns, Np, Nc, Ne]` |
| `get_PSP` | `propagators.py:1664-1803` | → `[Ns, Ns, Np_snk, Nc, Np_src, Nc]` |
| `get_VSP_highmode` | `propagators.py:1805-1951` | `get_VSP_highmode(self, t_source, t_sink, usedNe_source=None, *, usedNe_sink=None)` |
| `get_PSV_highmode` | `propagators.py:1953-2114` | `get_PSV_highmode(self, t_source, t_sink, usedNe_sink=None, *, usedNe_source=None)` |
| `get_PSP_highmode` | `propagators.py:2116-2344` | `get_PSP_highmode(self, t_source, t_sink, usedNe_sink=None, usedNe_source=None)`；场景级 memo 见 §4.6 |
| dagger 变换 | `propagators.py:1305-1359` | `_apply_gamma_on_spin(arr)`（前 3 轴 `[Lt, Ns, Ns, …]` 做 gamma15·conj(arr)·gamma15）；`_dagger_vsp`（尾部 `(Ne,Np,Nc) → (Np,Nc,Ne)`，transpose `0,2,1,4,5,3`）；`_dagger_psv`（对称）；`_dagger_psp`（交换两个点轴，transpose `0,1,2,5,4,3,6`） |

两个 Ne 参数语义（docstring `propagators.py:1814-1816` 明说）：VSP 版的 `usedNe_source` 是减法（VSV 的 j）端、输出 Ne 轴属于 sink（"VSP's output eigen index belongs to the sink; its subtraction index belongs to the source"）；PSV 版相反。

### 2.8 `current_elemental.py` 公开 API（`__all__`，`current_elemental.py:711-719`）

#### Schema 常量（`current_elemental.py:14-16`）

| 常量 | 值 | 含义 |
|---|---|---|
| `CURRENT_V2V_ARTIFACT_SCHEMA` | `"lattice.current.directed-v2v-artifact/v1"` | 磁盘工件 manifest 的 schema 标识；写入 manifest["schema"]（`current_elemental.py:136`），读取时强校验且 version==1（`current_elemental.py:184-186`） |
| `CURRENT_V2V_CONTRACTION_SCHEMA` | `"lattice.current.v2v-termwise-contraction/v1"` | termwise 收缩结果 dict 的 schema 标识（`current_elemental.py:367`）；同时作为 manifest["consumer"]["schema"] 写入（`current_elemental.py:144`）并在加载时逐字段比对（`current_elemental.py:277-289`） |
| `CURRENT_V2V_PAIR_CONTRACTION_SCHEMA` | `"lattice.current.v2v-term-pair-contraction/v1"` | 顶点对收缩结果 dict 的 schema 标识（`current_elemental.py:486`） |

manifest 必须恰好包含 `_ARTIFACT_KEYS = {schema, version, configuration, data, raw_contract, momenta, sources, consumer, artifact_identity}`（`current_elemental.py:17-28`；读取时 `set(manifest) != _ARTIFACT_KEYS` → ValueError，`current_elemental.py:185-186`）。

#### `save_directed_current_v2v`（`current_elemental.py:78-167`）

```python
def save_directed_current_v2v(directory, raw, contract, *, configuration,
                              momenta, gauge_source, eigenvector_source,
                              overwrite=False) -> Path
```

| 参数 | 含义 |
|---|---|
| `raw` | 必须是仅含 `"v2v"` 的 Mapping，转 host 连续数组；经 `validate_current_raw_contract(raw, contract, require_temporal=True)`（legacy spatial 契约被直接拒绝，`current_elemental.py:117` 附近；`insertion/current.py:277-279`） |
| `contract` | raw 契约（由 `insertion/current.py` 的 `build_current_raw_contract` 定义，`current.py:112-151`） |
| `configuration` | 非空字符串 |
| `momenta` | `len == shape[2]` 的三整数向量列表（`_momenta`，`current_elemental.py:51-73`） |
| `gauge_source` / `eigenvector_source` | 输入文件，各记录绝对 posix path + 整文件 sha256（`_source_record`，`current_elemental.py:40-45`；不存在抛 `FileNotFoundError`） |
| `overwrite` | True 时 `os.replace`，否则 `os.link` 硬链接（存在则 `FileExistsError`）+ finally 清理临时文件（`current_elemental.py:153-166`） |

流程：原子写数据（临时文件 `.v2v-{uuid}.tmp` → `np.save` + fsync → 按 sha256 命名 `v2v-{digest}.npy`；同名文件已存在则校验其哈希一致（内容寻址幂等）否则 `ValueError("existing content-addressed V2V file is corrupt")`，`current_elemental.py:117-134`）→ manifest（`artifact_identity = sha256(canonical_json(manifest 去 identity 字段))`，`_manifest_identity`，`current_elemental.py:32-33`）→ 原子发布。返回 manifest 路径。

#### `load_directed_current_v2v`（`current_elemental.py:201-291`）

```python
def load_directed_current_v2v(path, *, expected_configuration=None,
                              expected_gauge_sha256=None,
                              expected_eigenvector_sha256=None,
                              verify_sources=True, mmap_mode="r") -> dict
```

返回 `{"raw": {"v2v": ndarray}, "contract": validated, "manifest": manifest, "manifest_path": …, "verified_files": {"manifest"/"data"/"sources": {path, sha256}}}`。校验序列（`current_elemental.py:174-290`）：类型检查 → manifest 读取（严格字段集、schema/version、artifact_identity 重算防篡改）→ configuration 匹配 → data 元数据（format=="npy"、allow_pickle is False、文件名 `v2v-{digest}.npy` 内容寻址互锁、路径无目录成分）→ 数据哈希 → sources（绝对路径、哈希；`verify_sources=True` 时实读源文件比对）→ `expected_*_sha256` 可选绑定 → 用磁盘上的 raw_contract 重新 `validate_current_raw_contract(require_temporal=True)` → momenta 数量复核 → consumer 块逐字段比对（`current_elemental.py:277-289`）→ TOCTOU 双读（末尾重读 manifest/数据/源的 sha256，`current_elemental.py:286-290`）。`mmap_mode` 只允许 None 或 "r"。

#### `contract_directed_current_v2v`（`current_elemental.py:311-398`）

```python
def contract_directed_current_v2v(current_or_terms, raw, raw_contract,
                                  incoming_vsv, outgoing_vsv, *,
                                  source_time, sink_time, anchor_time,
                                  current_source_ne, current_sink_ne,
                                  momentum=0) -> dict
```

对每个 term 计算 `S(field→sink) J(bar←field) S(source→bar)`，然后乘 term 系数求和（docstring）。

| 参数 | 校验 |
|---|---|
| `current_or_terms` | `Current` 实例（取 `.terms`）或 term 序列；空 → `ValueError` |
| `incoming_vsv` / `outgoing_vsv` | 必须已加载且提供可调用的 `get(t_source, t_sink)`，否则 `TypeError`；**本函数绝不调用 load 或 propagator 生成器**（返回值 `"uses_preloaded_vsv_only": True`，`current_elemental.py:397`） |
| `source_time/sink_time/anchor_time` | 整数；`0 <= source_time, sink_time < extent`（extent = raw 的 bar_time 轴长） |
| `current_source_ne/current_sink_ne` | 裁剪 VSV block 前校验容量（`current_elemental.py:352-357, 466-477`） |

返回 dict：`{schema, value, axes, source_time, sink_time, anchor_time, boundary, momentum, terms: (per-term 记录…), raw_cache_identity, uses_preloaded_vsv_only: True}`；per-term 记录含 endpoints、系数、两个 accessor 的实际 get 调用参数、raw provenance。核心收缩见 §4.4。

#### `contract_directed_current_pair_v2v`（`current_elemental.py:401-520`）

```python
def contract_directed_current_pair_v2v(first_current_or_terms, first_raw,
                                       first_raw_contract,
                                       second_current_or_terms, second_raw,
                                       second_raw_contract, vsv, *,
                                       first_anchor_time, second_anchor_time,
                                       first_field_ne, first_bar_ne,
                                       second_field_ne, second_bar_ne,
                                       first_momentum=0, second_momentum=0,
                                       array_backend=None) -> dict
```

对每对 (A term, B term) 计算有序、无味道、无符号的 connected trace；两个 propagator 是 `S(field_A → bar_B)` 与 `S(field_B → bar_A)`；**不隐含 Wick 符号、味道因子、体积归一、共轭、实部选择或源平均**（"operation" 字段逐字声明，`current_elemental.py:489-492`）。

- 两个 raw 的 temporal extent 与 boundary 必须一致（`current_elemental.py:413-417`）；`vsv` 单一 accessor 供两条腿使用；`array_backend` 默认 np，须提供 asarray/einsum（`current_elemental.py:408-411`）；四个 Ne 计数必须为正整数（`current_elemental.py:445-453`）。
- 返回：`{schema: PAIR_SCHEMA, value(标量), axes: (), operation, anchors, boundary, momenta, ne, term_pairs, raw_cache_identities, uses_preloaded_vsv_only: True}`。

## ③ 数据结构与数组形状约定

### 3.1 `operator.parts` 结构（Meson/Current 共用）

`_make_cache`（`propagators.py:261-300`）遍历 `parts`：`len(parts)//2` 个"算符行"，第 i 行由 `parts[2i]`（gamma 名称，喂给 `lattice.insertion.gamma.gamma()`）与 `parts[2i+1]`（elemental 项列表）组成；每个 elemental 项是 4 元组 `(elemental_coeff, derivative_idx, momentum_idx, profile)`（`propagators.py:282-284`）。同一行内多 terms 线性叠加：`ret_elemental[-1] += elemental_coeff * cache[(deriv, mom)]`。elemental 数据切片约定：`elemental_data[derivative_idx, momentum_idx, :, :usedNe, :usedNe]`（`propagators.py:280-282`），即磁盘 `(num_disp, num_mom, Lt, Ne, Ne)` → 缓存切片 `[Lt, Ne, Ne]`。

### 3.2 缓存元组与输出形状

| 类 | 缓存内容 | 输出接口与形状 |
|---|---|---|
| `Meson` | `(gammas, elementals)`：`gammas` shape `[n_parts, Ns, Ns]`，`elementals` shape `[n_parts, Lt, Ne, Ne]` | `get(t)` → `[Ns,Ns,Ne,Ne]`（int t） / `[t_len,Ns,Ns,Ne,Ne]` |
| `Current` | 5 元组（`propagators.py:592-608`）：`gammas [n_parts,Ns,Ns]`、`v2v [n_parts,Lt,Ne,Ne]`、`v2p [n_parts,Lt,Ne,Np,Nc]`（预转置）、`p2v [n_parts,Lt,Np,Nc,Ne]`、`p2p` 为惰性指令列表 `list[list[(disp_idx, mom_idx, coeff)]]` | `get` / `get_v2p` → `[Ns,Ns,Ne,Np,Nc]` / `get_p2v` → `[Ns,Ns,Np,Nc,Ne]` / `get_p2p` → `[Ns,Ns,Np,Nc,Np,Nc]` |
| `Propagator` | `cache [Dt,Ns,Ns,Ne,Ne]`（VSV 序）+ `cache_dagger`（gamma15 共轭） | `get(t_source, t_sink)` → 相对时间片 `[Ns,Ns,Ne,Ne]` |
| `PropagatorWithCurrent` | `vsp_cache/vsp_dagger`、`psv_cache/psv_dagger`、`psp_cache/psp_dagger`、`tilde_S_*` 高模缓存、`_last_psp_highmode` | `get_VSP` → `[Ns,Ns,Ne,Np,Nc]`；`get_PSV` → `[Ns,Ns,Np,Nc,Ne]`；`get_PSP` → `[Ns,Ns,Np_snk,Nc,Np_src,Nc]` |

### 3.3 directed current V2V raw 契约

工件里的 `raw = {"v2v": ndarray}`，契约由 `insertion/current.py` 的 `build_current_raw_contract` 定义（`current.py:112-151`）：

- shape 恰为 `(8, Lt, momentum_count, used_ne, used_ne)`，轴名 `["direction", "time", "momentum", "sink_ne", "source_ne"]`；
- direction 轴 8 个值来自 `DirectedCurrentBasis.DIRECTIONS`（`gauge_link.py:137-146`）：0..7 = +x, +y, +z, -x, -y, -z, +t, -t（后四个带 dagger 标志）；
- dtype 必须 complex；`ne` 元数据要求 `source == sink == used <= available` 且 `raw_generator_used_ne_is_symmetric=True`；
- `cache_identity` 是契约（除 identity 外字段）的规范化 JSON 指纹，读取时重算比对（`current.py:104-106`）。

manifest 的 consumer 块把 raw 轴命名为 `["direction","bar_time","momentum","bar_ne","field_ne"]`（`current_elemental.py:146-152`）——raw 的时间轴锚定在 bar 端点（`resolve_directed_current_raw` 取 `raw_anchor_time = endpoints["bar_time"]`，`current.py:348-361`）。

VSV block 约定（`_vsv_block`，`current_elemental.py:296-310`）：shape 恰为 `(sink_spin, source_spin, sink_ne, source_ne)`，前两轴 (4,4)，complex、全部有限值；GPU 数组先 `.get()`。

### 3.4 termwise 收缩输出的轴语义

- `contract_directed_current_v2v` 输出轴 `(external_sink_spin a, external_source_spin c, external_sink_ne A, external_source_ne C)`（返回值 "axes" 字段，`current_elemental.py:368-373`）；
- `contract_directed_current_pair_v2v` 输出为标量（`axes: ()`）。

## ④ 算法与流程

### 4.1 `Meson._make_cache`（`propagators.py:261-300`）

产出 `(gammas, elementals)`：

- 非 dagger：`(backend.asarray(ret_gamma), backend.asarray(ret_elemental))`；
- dagger（`propagators.py:290-295`）：`gammas → contract("ik,xlk,lj->xij", gamma(8), ret_gamma.conj(), gamma(8))`（即 gamma(8)·conj(Γ)·gamma(8)）；`elementals → contract("xtba->xtab", ret_elemental.conj())`（最后两轴转置取共轭）。

`get(t)` 的收缩 `"xij,xab->ijab"` 把 gamma 部分（轴 i,j）与 elemental 部分（轴 a,b）合成为顶点 `V_{ijab}(t) = sum_x Gamma^x_{ij} E^x_{ab}(t)`；输出轴序 `(sink_spin i, source_spin j, sink_ne a, source_ne b)`。

### 4.2 `Current._make_cache` 与 v2p 对称性（`propagators.py:425-608`）

`_build_displacement_reversal_map`（`propagators.py:425-455`）：对每个 `disp_idx` 取 `GaugeLink(disp_idx).displacement`，用 `gauge_link.conjugate()`（反转 gauge_list）找到反位移的 `reverse_idx`。docstring 对称性 `v2p(disp) ~ p2v(-disp).transpose(Ne, Np)`——代码在 `_make_cache` 中确实 `v2p_data_slice = p2v_full[:, reverse_gaugelink_idx, ...].transpose(0, 3, 1, 2)`（轴 `(Lt,Np,Nc,Ne) → (Lt,Ne,Np,Nc)`，`propagators.py:540-551`），即 **v2p 从不单独读盘，完全由 p2v 的反位移切片转置而来**。

对每个算符行遍历 elemental 项 `(elemental_coeff, gaugelink_idx, momentum_idx, _profile)`，同时填 4 个累积器（`propagators.py:457-608`）：

| 累积器 | 切片 | 说明 |
|---|---|---|
| `ret_elemental_v2v` | `elemental_data[gaugelink_idx, momentum_idx, :, :usedNe, :usedNe]` | 键 `("v2v", idx, mom)` |
| `ret_elemental_p2v` | `p2v_full[:, gaugelink_idx, :usedNp, :, :usedNe]`，shape `[Lt, Np, Nc, Ne]` | **p2v 忽略 momentum_idx**（注释 "momentum-independent"，`propagators.py:521-524`） |
| `ret_elemental_v2p` | `p2v_full[:, reverse_idx, :usedNp, :, :usedNe].transpose(0,3,1,2)` → `[Lt, Ne, Np, Nc]` | 由 p2v 导出 |
| `ret_elemental_p2p` | 惰性指令列表 `[(gaugelink_idx, momentum_idx, coeff), …]` per part | 按需加载 |

dagger 版：前两元与 Meson 相同地做 gamma(8) 共轭变换；v2p/p2v 两元不做共轭（只原样 asarray），p2p 指令原样保留。

### 4.3 `Current.get_p2p` 的稀疏→稠密展开（`propagators.py:658-739`）

按 t 懒加载：`p2p_sparse_list = self.p2p_data.load(self.key, t, num_momentum=…)`（长度 = num_disp × num_momentum，元素为 dict）；对每个 part 建稠密 `zeros((usedNp, Nc, usedNp, Nc), "<c16")`：

- `sparse_data["type"] == "identity"`：`dense[p,c,p,c] += coeff`（单位矩阵 delta）；
- `"sparse"`：`dense[l, :, r, :] += coeff * values[i]`（indices `[N,2]` 的 (l,r) 对、values `[N,3,3]`），仅当 `l < usedNp and r < usedNp`（`propagators.py:706-712`）。

结果按 t 缓存于 `self.p2p_loaded[t]`；最后 `contract("xij,xpwqz->ijpwqz", cache[0], elemental_stack)`。索引展开：`idx = gaugelink_idx * num_momentum + momentum_idx`（`propagators.py:703`）。

### 4.4 termwise V2V 收缩（`contract_directed_current_v2v`，`current_elemental.py:335-366`）

逐 term 流程：

1. `endpoints = resolve_current_term_endpoints(term, anchor_time, temporal_extent=extent, boundary=…)`——按 term 的 bar/field/link_origin 偏移算三个时间点（`current.py:594-624`；boundary=="periodic" 取模、"open" 越界抛 IndexError）；
2. `raw_resolver` → `resolve_directed_current_raw`：取 `raw["v2v"][direction, bar_time, momentum, :sink_ne, :source_ne]`，direction 由 `DirectedCurrentBasis.index_for_term(term.direction, term.link)` 决定（`current.py:321-377`）；
3. `resolve_current_term_spin`：把 spinless raw 值 `(sink_ne, source_ne)` 乘以单项显式 spin 矩阵（gamma 矩阵，或 linked 项的 `wilson_r*I -/+ Gamma`，`current.py:626-644`），得到 `vertex`，轴 `(sink_spin, source_spin, sink_ne, source_ne) = (a,b,j,i)`（j=bar_ne, i=field_ne）；
4. `outgoing = _vsv_block(outgoing_vsv.get(endpoints["field_time"], sink_time))`，轴 `(a, f, A, i)`；裁 `[..., :current_source_ne]`；
5. `incoming = _vsv_block(incoming_vsv.get(source_time, endpoints["bar_time"]))`，轴 `(b, c, j, C)`；裁 `[..., :current_sink_ne, :]`；
6. 核心收缩：

   ```python
   term_value = np.einsum("afAi,bfji,bcjC->acAC", outgoing, vertex, incoming, optimize=True)
   ```

   即 `V_{aC,Ac} = sum_{f,i,j} S_out[a,f,A,i] J[b,f,j,i] S_in[b,c,j,C]`（对 b 求和）；
7. 权重：`weighted = coefficient * normalization * term_value`（term_mapping 的 coefficient 必填、normalization 默认 1，`current_elemental.py:357-361`）；逐项 shape 一致性校验（"term-wise VSV contractions must have identical external shapes"，`current_elemental.py:353-355`）。

权重应用一次且仅一次：raw resolver 层无权重（"Current weighting remains in the assembler"，`current.py:322`），spin 层无权重（`current.py:687-693`），本模块统一乘 `coefficient*normalization`。

### 4.5 pair trace 收缩（`contract_directed_current_pair_v2v`，`current_elemental.py:445-520`）

双层循环（first_terms × second_terms）：各自 resolve endpoints + spin vertex；`first_to_second = vsv.get(field_A, bar_B)`（轴 `(sink_spin, source_spin, sink_ne, source_ne)`，sink=bar_B、source=field_A），裁 `[..., :second_bar_ne, :first_field_ne]`；`second_to_first = vsv.get(field_B, bar_A)` 对称处理。trace 收缩：

```python
pair_value = backend.einsum("bfji,ackl,afki,bcjl->", first_to_second, second_to_first,
                            first_vertex, second_vertex, optimize=True)
```

轴名对齐：first_to_second `b f j i`；second_to_first `a c k l`；first_vertex `a f k i`；second_vertex `b c j l`。全部指标收缩 → 标量。权重 `first_weight * second_weight * pair_value`，`weight = coefficient * normalization`（`current_elemental.py:493-497`）。

### 4.6 `PropagatorWithCurrent` 的两时间取数与高模投影

统一取数模式（以 `get_VSP`，`propagators.py:1361-1511` 为例）：

| 时间参数情形 | 行为 |
|---|---|
| `t_source`、`t_sink` 都是 int | `vsp_cached_time == t_source` → `vsp_cache[(t_sink-t_source) % Lt]`；elif `psv_cached_time == t_sink` → `psv_dagger[(t_source-t_sink) % Lt]`；否则从 `vsp_data[t_source, …, :usedNe, :usedNp, :]` 切片，`cache=True` 时存 `vsp_cache` 并预计算 `vsp_dagger` |
| 只 `t_source` 为 int | 同上（无 dagger 分支） |
| 只 `t_sink` 为 int | 需要反向 ⇒ 用 PSV 数据：`psv_data[t_sink, …, :usedNp, :, :usedNe]` → `psv_dagger`（VSP 排序输出）；psv_data 缺失则 `ValueError` |

`get_PSV`（`propagators.py:1513-1662`）完全镜像：正向用 psv_data，t_sink 为 int 时反向用 vsp_data 的 dagger。`get_PSP`（`propagators.py:1664-1803`）：正向锚 t_source 时 `psp_data[t_source]` 再按 usedNp 切 `[..., :usedNp, :, :usedNp, :]`；t_sink 为 int 时用 `psp_dagger`。所有 get 均接受 `cache=True`；`cache=False` 时直接在局部切片上取片不写缓存。

VSP 切片顺序 `[…, :usedNe, :usedNp, :]` 与 PSV 切片顺序 `[…, :usedNp, :, :usedNe]` 各自匹配 §1.3 的尾部轴序。

高模投影（low-mode subtraction），记 M = overlap matrix（`overlap_matrix_data` shape `[Lt, Ne, Np, Nc]`）：

| 接口 | 公式（docstring 与 einsum 实现一致） | 特殊行为 |
|---|---|---|
| `get_VSP_highmode`（`propagators.py:1805-1951`） | `tilde S_{i,xa} = S_{i,xa} − Σ_j S_{ij} · conj(M_{jx,a})`；实现 `contract("abij,jxc->abixc", S_vsv, M_conj_t)`（多时间 `"tabij,…"） | `usedNe_source == 0` 时退化为 `get_VSP(...)[..., :usedNe_sink, :, :]`（`propagators.py:1827-1830`）；单时间（两个 int）一律不缓存高模、缓存未投影版（`propagators.py:1838-1846`）；多时间且 t_sink 为 int 时结果按 PSV-dagger 排序缓存（`propagators.py:1930-1937`），供后续 PSV 正向请求命中 |
| `get_PSV_highmode`（`propagators.py:1953-2114`） | `tilde S_{xa,i} = S_{xa,i} − Σ_j M_{xj,a} · S_{j,i}`；实现 `contract("jxc,abji->abxci", M_t, S_vsv)` | `usedNe_sink == 0` 时返回 `get_PSV(...)[..., :usedNe_source]`；缓存与 VSP 版对称（`propagators.py:2090-2110`） |
| `get_PSP_highmode`（`propagators.py:2116-2344`） | `tilde S_{xa,yb} = S_{xa,yb} − Σ_i M_{xi,a} · tilde S_{i,yb} − Σ_j S_{xa,j} · M_{jy,b}`；实现 `_compute_PSP_highmode`（`propagators.py:2133-2344`）：`term2 = contract("ixc,abiyd->abxcyd", M_sink_t, S_vsp_tilde)`、`term3 = contract("abxcj,jyd->abxcyd", S_psv, M_source_t)`（`M_source = M_full[:usedNe_source].conj()`）、`tilde_S = term1 − term2 − term3`；usedNe 为 0 时对应 term = 0；多时间版插入 t 轴 | 依赖内部调用 `get_PSP`、`get_PSV`、`get_VSP_highmode`（PSP 高模继承 VSP 高模的投影）；场景级 memo：`_psp_highmode_cache_key`（times + 两个 Ne + self.usedNe/usedNp），命中 `_last_psp_highmode` 直接返回（`propagators.py:1038-1056`）——注释说明 "scene expansion 每个 scene 以相同 (t_src, t_snk) 调一次" |

## ⑤ 不变量与校验

| # | 不变量 | 出处 |
|---|---|---|
| 1 | 所有 `load()` 的 identity 机制：key、usedNe/usedNp、loader 签名（含 id、文件 stat）、数据签名（含 ndarray SHA-256）、算符签名、dagger 任一变化 ⇒ 全量释放重建 | `propagators.py:239-259, 389-423, 783-800, 1096-1264` |
| 2 | `_normalize_extent`：`0 <= used <= available` 且为非 bool 整数 | `propagators.py:46-56` |
| 3 | 五输入 Ne/Np 一致性（`PropagatorWithCurrent`）；`Current.load` 要求 p2v_data 且 Ne 与 elemental 一致 | `propagators.py:1070-1094, 396-404` |
| 4 | 相对时间越界检查 `_relative_time_index`；所有 `get_*` 至少一个时间为 int | `propagators.py:802-821, 859-862` |
| 5 | `PropagatorLocal.get` 断言 `t_source == t_sink` | `propagators.py:913-918` |
| 6 | `Current.get_p2p` 不支持数组 t（`NotImplementedError`） | `propagators.py:737-739` |
| 7 | overlap_matrix 对 `PropagatorWithCurrent.load` 必需 | `propagators.py:1206-1207` |
| 8 | `Meson.get` 的 dagger/非 dagger 分支 contraction 字符串逐字相同——语义差别全在缓存构造 | `propagators.py:313-324` |
| 9 | `propagators.py:30` 导入的 `lattice.constant.Nc` 未被使用；`Current.get_p2p` 内硬编码 `Nc = 3`（常量来源不一致，重构时注意） | `propagators.py:30, 664` |
| 10 | manifest 字段集恰为 `_ARTIFACT_KEYS`；identity 为去 identity 字段的 canonical JSON sha256 | `current_elemental.py:32-33, 185-189` |
| 11 | 数据文件内容寻址命名 `v2v-{sha256}.npy`，文件名与哈希互锁 | `current_elemental.py:246-249` |
| 12 | raw 必须满足 directed 契约（`require_temporal=True`），legacy spatial 契约被显式拒绝 | `current_elemental.py:117` 附近；`current.py:277-279` |
| 13 | 收缩函数的 accessor 只读（`get`），绝不加载 propagator（`uses_preloaded_vsv_only: True`） | `current_elemental.py:397, 519` |
| 14 | 加载全程 TOCTOU 防护 | `current_elemental.py:286-290` |
| 15 | 权重应用一次且仅一次 | `current.py:322, 687-693`；`current_elemental.py:357-361` |

## ⑥ 相关测试

| 测试文件 | 被测对象 | 确认的行为（出处均为 test/ 下文件：行号） |
|---|---|---|
| `test/test_propagator_psv.py` | preset 的 PSV loader（非 propagators.py 的类） | 懒加载契约（`load()` 返回 lazy handle 而非 ndarray，`[:]` 物化；timeslice 版 suffix 必须含 `.t???` 占位符）(:1-33)；shape `[Lt, Ns, Ns, Np, Ne]` 保真 (:34-120) |
| `test/test_propagator_with_current.py` | 本模块（经 `lattice.quark_diagram` 导入，:11-13） | 最小参数构造 (:35)；PSV/VSP 高模公式的数值回归（mock 数据上复算 einsum 公式）(:106-192)；`usedNe==0` 退化 (:194)；缓存初始化与 load 清缓存 (:204-245)；Meson dagger=source (:270)；Current 需要 p2v/p2p (:295-342)；context manager (:404)；usedNe/usedNp 限制切片 (:427, 455) |
| `test/test_dt0_gamma5_vsp_reconstruction.py` | `_dagger_psv`/`_dagger_vsp` | 与 `einsum("ia,tbaxce,bj->tijexc", gamma5, psv.conj(), gamma5)` 逐元素一致（gamma15 共轭 + spin 轴交换 + 尾轴重排）(:12-40) |
| `test/test_current_v2v_contraction.py` | `contract_directed_current_v2v(_pair)` | `ExistingVSV` 假 accessor 断言绝不 `load` (:23-31)；`ConservedVectorCurrent(wilson_r=1.25).terms[6:8]`（±t 项）与手算 `1.25*I -/+ gamma8` spin 矩阵回归 termwise 收缩 (:32-80+)；pair 版（PAIR_SCHEMA）同文件覆盖 |
| `test/test_current_v2v_persistence.py` | `save/load_directed_current_v2v` | round trip + source 哈希绑定 (:32-66)；数据/manifest 篡改拒绝 (:68-122)；拒绝覆盖与要求真实源文件 (:124-) |
| `test/test_conserved_charge_v2v.py` | `contract_directed_current_v2v` | conserved charge 投影（dual projection 无隐藏共轭、ratio 形状/零除、endpoint-aware 时间 current 包装）(:14, 27-109) |
| `test/test_current_elemental.py` | preset 的 V2P/P2V/P2P/OverlapMatrix loader | shape/一致性/错误处理（相关但非本模块） |
| `test/test_ne_provenance.py` | `Meson/Propagator/PropagatorLocal/PropagatorWithCurrent` | 经 `lattice.quark_diagram` re-export 的句柄的 Ne/usedNe 出身校验 (:12) |
| `test/test_ported_production_hooks.py` | `Propagator`、`PropagatorWithCurrent` | 生产 hook 行为 (:14) |

最小验证入口：高模投影公式 `pytest test/test_propagator_with_current.py -k "highmode" -x`；dagger 变换 `pytest test/test_dt0_gamma5_vsp_reconstruction.py -x`；V2V 收缩与工件 `pytest test/test_current_v2v_contraction.py test/test_current_v2v_persistence.py -x`。
