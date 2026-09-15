# 测试组织与运行方式（Testing）

> 定位：本文档描述 EasyDistillation 的 pytest 测试体系——配置与 markers、`test/conftest.py` 的自动 skip 行为、测试-模块覆盖对照表、`test/` 下的测试数据文件清单，以及当前测试体系的薄弱点。所有出处均以 `path:line` 形式指向仓库真实代码。

## ① 模块概览与依赖关系

测试体系由三部分组成：

| 组成 | 位置 | 作用 |
|---|---|---|
| pytest 全局配置 | `pyproject.toml:60-71` | 指定收集路径、`pythonpath`、排除目录、注册 markers |
| 共享 conftest | `test/conftest.py:1-69` | 注册 markers（与 pyproject 重复一份）、按环境自动 skip gpu/mpi 测试、GPU 缺失时强制回退 numpy 后端 |
| 测试文件 | `test/test_*.py`（39 个）+ `test/scripts/`（11 个探索脚本） | 单元/集成测试与脚本式冒烟代码 |

依赖关系：

- pytest 配置依赖仓库以 `pythonpath = ["."]` 方式把根目录加入 `sys.path`，因此 `import lattice` 无需安装包（`pyproject.toml:62`）。
- `test/conftest.py:67-69` 依赖 `lattice.set_backend`：在 GPU 不可用时把整个会话强制回退到 numpy 后端。
- 带数据文件的测试依赖 `test/` 目录下的一套 `weak_field.*` 小样本（见 ③）。
- gpu 标记测试额外依赖 CuPy；mpi 标记测试额外依赖 mpi4py + PyQuda（`test/conftest.py:6-27` 的探测逻辑）。
- `test/scripts/` 下的探索脚本不在 pytest 体系内（见 ④）。

## ② 公开配置与 conftest API 参考

### 2.1 pytest 配置项（`pyproject.toml:60-71`）

| 配置项 | 值 | 含义 |
|---|---|---|
| `testpaths` | `["test"]`（`pyproject.toml:61`） | 不带参数运行 pytest 时只收集 `test/` |
| `pythonpath` | `["."]`（`pyproject.toml:62`） | 把仓库根目录加入 `sys.path`，使 `import lattice` 生效 |
| `norecursedirs` | `["test/scripts"]`（`pyproject.toml:65`） | 排除 `test/scripts/`——该目录下的脚本以 `test_` 开头但不是 pytest 测试（见 ④.3） |
| `markers` | gpu / mpi / slow / integration（`pyproject.toml:66-71`） | 四个自定义 marker 的声明与描述 |

### 2.2 marker 约定

| marker | 含义 | 实际使用者 |
|---|---|---|
| `gpu` | 需要 GPU（CuPy） | `test/test_eigenvector.py:5`、`test/test_gauge_links_methods.py:4`、`test/test_meson_spectrum.py:6` |
| `mpi` | 需要 MPI（mpi4py/PyQuda） | `test/test_perambulator.py:5`、`test/test_perambulator_mpi.py:5`、`test/test_perambulator_phase3.py:19` |
| `slow` | 耗时 >10 秒 | 已注册（`pyproject.toml:66-71`、`test/conftest.py:39-42`）但**当前没有任何测试使用**（grep `test/*.py` 无 `pytest.mark.slow`） |
| `integration` | 需要外部数据文件 | 仅 `test/test_current_contraction.py:15`（`pytestmark = pytest.mark.integration`） |

标记方式：上述文件均用模块级 `pytestmark = pytest.mark.<name>` 整体标注；其余测试文件不带任何 marker。

### 2.3 `test/conftest.py` 公开钩子与辅助函数

| 符号 | 签名 | 参数 | 返回值 / 行为 |
|---|---|---|---|
| `_cupy_available` | `_cupy_available() -> bool`（`test/conftest.py:6-17`） | 无 | CuPy 可 import **且** `cupy.cuda.runtime.getDeviceCount()` 不抛异常时返回 `True`；仅 import 成功但无 GPU 驱动时返回 `False` |
| `_mpi_available` | `_mpi_available() -> bool`（`test/conftest.py:20-27`） | 无 | `mpi4py` 与 `pyquda` 均可 import 时返回 `True`，任一 `ImportError` 返回 `False` |
| `pytest_configure` | `pytest_configure(config)`（`test/conftest.py:30-46`） | `config`：pytest 配置对象 | 通过 `config.addinivalue_line` 注册 gpu/mpi/slow/integration 四个 marker（与 `pyproject.toml:66-71` 重复声明一份） |
| `pytest_collection_modifyitems` | `pytest_collection_modifyitems(config, items)`（`test/conftest.py:49-69`） | `config`：pytest 配置对象；`items`：收集到的测试项列表 | 见 ④.2 的两条自动行为 |

## ③ 测试数据文件清单（`test/` 下）

### 3.1 weak_field 数据集

`test/` 下保存了一套由弱场规范组态（cfg 名 `"weak_field"`）衍生的完整流水线小样本，供无 GPU/MPI 环境下的小规模数值验证使用：

| 文件 | 类型 | 使用者 | 出处 |
|---|---|---|---|
| `weak_field.npy` | gauge 场 ndarray | test_load（与 ILDG/binary 读法逐位对比） | `test/test_load.py:16` |
| `weak_field.bin` | gauge 场 raw binary（`<c16`） | test_load | `test/test_load.py:20-21` |
| `weak_field.lime` | gauge 场 ILDG (Lime) 格式 | test_load、test_eigenvector、test_elemental、test_displacement_elemental、test_perambulator* | `test/test_load.py:24`、`test/test_eigenvector.py:31`、`test/test_elemental.py:18`、`test/test_displacement_elemental.py:18`、`test/test_perambulator_phase3.py:42` |
| `weak_field.eigenvector.input.npy` | Laplace eigenvector 输入 | test_elemental、test_displacement_elemental、test_perambulator* | `test/test_elemental.py:19`、`test/test_displacement_elemental.py:19` |
| `weak_field.eigenvector.ref.npy` | eigenvector 参考结果 | test_eigenvector（对比生成结果） | `test/test_eigenvector.py:39` |
| `weak_field.eigenvalue.npy` | eigenvalue 参考结果 | test_eigenvector | `test/test_eigenvector.py:40` |
| `weak_field.elemental.npy` | elemental 参考结果 | test_meson_spectrum（作为 `ElementalNpy` 数据，shape `[13, 6, Lt, Ne, Ne]`） | `test/test_meson_spectrum.py:39` |
| `weak_field.displacement_elemental.npy` | displacement elemental 参考结果 | test_displacement_elemental（输出后缀约定，`out_suffix = ".displacement_elemental.npy"`） | `test/test_displacement_elemental.py:25` |
| `weak_field.perambulator.npy` | perambulator 数据 | test_meson_spectrum（`PerambulatorNpy`，shape `[Lt, Lt, 4, 4, Ne, Ne]`） | `test/test_meson_spectrum.py:41-42` |

小样本晶格参数以 `latt_size = [4, 4, 4, 8]`（`Lx=Ly=Lz=4, Lt=8`）、`Ne=20` 为代表（`test/test_eigenvector.py:26-28`、`test/test_perambulator_phase3.py:37-38`）。

### 3.2 无引用的遗留数据文件

| 文件 | 状态 |
|---|---|
| `test/irrep_vertices_t2pp.pkl` | **当前没有任何 `test/test_*.py` 引用**（grep `t2pp` 在 test/ 顶层测试文件中无命中）；仅 `test/scripts/t2pp_skeleton.py` 名号相关 |
| `test/irrep_vertices_t2pp.pkl.meta.pkl` | 同上，成对存在的元数据文件 |

二者为历史/探索遗留数据的候选（见 ⑥.2）。

## ④ 算法与流程

### 4.1 运行方式

```bash
# 全量收集运行（无 GPU/MPI 环境下，gpu/mpi 测试自动 skip，见 4.2）
python -m pytest

# 无 GPU/MPI 即可通过的最小子集
python -m pytest test/test_gamma.py test/test_sampling_weight.py \
    test/test_vertex_type.py test/test_sparsened_point.py test/test_symmetry.py -q

# 仅运行某个 marker
python -m pytest -m integration
```

收集范围由 `testpaths = ["test"]`（`pyproject.toml:61`）决定；`norecursedirs = ["test/scripts"]`（`pyproject.toml:65`）保证 `test/scripts/` 下以 `test_` 开头的 11 个探索脚本（如 `test/scripts/test_calc_all_comparison.py`）不会被收集。

### 4.2 conftest 的两条非显然行为

1. **依赖缺失自动 skip**：收集修改钩子在收集完成后遍历所有测试项，GPU 不可用时给带 `gpu` marker 的项追加 `skip`（reason "CuPy/GPU not available"），MPI 不可用时对 `mpi` 项同理（`test/conftest.py:49-61`）。测试文件自身无需编写任何环境判断。
2. **GPU 缺失时强制回退 numpy 后端**：某些遗留 GPU 模块在 import 期就会选定 CuPy 后端；若此时才发现本机无驱动，后续普通测试会因收集顺序偶然 import 了这些模块而失败。因此钩子在 GPU 不可用时对整个会话调用 `set_backend("numpy")`（`test/conftest.py:63-69`）。

此外，部分 gpu 标记的脚本式测试还自带模块级双保险：如 `test/test_eigenvector.py:8-25` 在 `set_backend("cupy")` 失败或 CUDA 分配探针失败时 `pytest.skip(..., allow_module_level=True)`；`test/test_gauge_links_methods.py:6-14` 有类似的 pre-flight 检查。

### 4.3 脚本式测试的收集行为

test_load、test_elemental、test_eigenvector、test_displacement_elemental、test_perambulator、test_perambulator_mpi、test_gauge_links_methods 七个文件**不含任何 `def test_` 函数，也不含 assert**（逐文件 grep 计数均为 0）。pytest 收集时 import 这些模块会执行全部模块级代码（真正生成/加载/对比数据），但：

- 收集期 import 失败会报 collection error；运行期数值不一致若无 assert 则**静默通过**（这些文件只能提供"收集期不崩溃"的烟雾信号）；
- 这些文件需要对应的 `weak_field.*` 数据存在于 `test/` 下，否则收集即报错；
- 它们大多未标 integration/slow，导致无 GPU/MPI 时也会被收集执行（依赖 conftest 的 skip 或自身的 `allow_module_level` skip 保护）。

这些文件实质是"可执行文档/冒烟脚本"性质。

## ⑤ 不变量与校验

与测试体系直接相关的横切不变量（各测试所覆盖的领域不变量详见各域文档）：

| 不变量 | 覆盖测试 |
|---|---|
| `QuarkDiagram` 默认 `validate=True`：校验每条 quark 线有归属（顶点为孤立/介子/重子端点） | `test/test_quark_diagram_contraction.py`、`test/test_current_contraction.py` |
| Ne/usedNe 归一：越界/非 int/负值被拒（`True` 因 bool 是 int 子类同样被拒） | `test/test_ne_provenance.py:82-93` |
| 缓存身份（cache identity）由 (key, usedNe, loader 签名, 数据签名, operator 签名, dagger) 构成；Ne 或 loader 版本变化即重建缓存 | `test/test_ne_provenance.py:61-137` |
| directed current v2v artifact 的 schema 字符串与 sha256 manifest：防篡改、拒绝覆盖已存在文件 | `test/test_current_v2v_persistence.py`（全文） |
| gauge 场三种读法（ndarray / raw binary / ILDG）逐位一致 | `test/test_load.py:16-24`（注意：该文件为脚本式，无 assert） |
| 数据文件缺失/shape 不匹配时的报错路径（FileNotFoundError、shape 校验） | `test/test_current_elemental.py`、`test/test_propagator_psv.py`（tmp_path 自造数据验证） |

## ⑥ 测试-模块覆盖对照表与薄弱点

### 6.1 覆盖对照表（39 个测试文件）

| 测试文件 | 覆盖的模块 / 行为 | markers | 外部数据 |
|---|---|---|---|
| test_calc_diagram_bind.py | `quark_diagram._CalcDiagramPrepared`/`calc_diagram_bind`：irrep 骨架顶点被 `vertex_map` 替换、绑定幂等性、timing 键 | 无 | 无（MagicMock） |
| test_conserved_charge_v2v.py | `correlator.conserved_charge`：显式对偶投影、ratio 轴对齐、零除拒绝；交叉使用 `current_elemental` 与 `insertion.current` | 无 | 无 |
| test_current.py | `insertion.current` 综合：`CurrentTerm` schema、spatial assembler、endpoint 边界、legacy adapter 拒绝、`CurrentElementalGenerator.compute_elemental`、局域/conserved 流、WT 恒等式与 PCAC、`lattice_divergence` | 无 | 无 |
| test_current_consumption.py | current 消费端：spin-aware consumer 回归、显式单次 γ 应用、spin resolver 拒绝 | 无 | 无 |
| test_current_contraction.py | `QuarkDiagram` current 顶点展开 + `compute_diagrams(_multitime)`：4 种 propagator 类型、subscripts/shape 匹配 | `integration` | 无（图手工构造） |
| test_current_elemental.py | preset 的 `CurrentElementalV2P/P2V/P2P`、`OverlapMatrixNpy`：shape 定义、数据检索、v2p↔p2v 一致性、缺文件/shape 不匹配处理 | 无 | 无（tmp_path 自造） |
| test_current_v2v_contraction.py | `current_elemental.contract_directed_current_v2v`/`_pair_v2v`：endpoint 轴与权重、两顶点循环一致、不兼容输入拒绝、开放边界拒绝 | 无 | 无 |
| test_current_v2v_persistence.py | `save/load_directed_current_v2v`：artifact 往返、sha256 manifest 防篡改、拒绝覆盖 | 无 | 无（tmp_path） |
| test_current_vertex.py | `QuarkDiagram` current 顶点展开：(v/p)×(v/p) 组合、双 current、无 vertex_list 图 | 无 | 无 |
| test_diagram.py | `Diagram`、`diagram_simplify`、`remove_disconneted_diagram`（源码拼写）+ spatial_structure/group_projection/`genLittleGroupIrrep` | 无 | 无 |
| test_displacement_elemental.py | `DisplacementElementalGenerator` 全流程 | 无 | `weak_field.lime`、`.eigenvector.input.npy`、参考 `.displacement_elemental.npy` |
| test_dt0_gamma5_vsp_reconstruction.py | `PropagatorWithCurrent` 的 VSP↔PSV 经 γ5 dagger 的自旋轴交换重构（dt=0） | 无 | 无 |
| test_eigenvector.py | `EigenvectorGenerator`：LapEig 本征向量/值与参考对比 | `gpu` | `weak_field.lime`、`.eigenvector.input.npy`、`.eigenvector.ref.npy`、`.eigenvalue.npy` |
| test_elemental.py | `ElementalGenerator`：gauge+eigenvector → elemental | 无 | `weak_field.lime`、`.eigenvector.input.npy`、参考 `.elemental.npy` |
| test_flavor_structure.py | `base_types.{Tag,Flavor}`、`flavor_structure`、`quark_diagram.quark_contract`：Tag 不可变、介子/重子收缩、共轭 | 无 | 无 |
| test_gamma.py | `insertion.gamma`：`gamma(n)` 数组性质、`GammaName`、scheme/group 等返回类型、`gamma_transform` | 无 | 无 |
| test_gauge_link.py | `insertion.gauge_link`：`GaugeLink` 初始化等价、C4z/复合变换、idx 编码往返与唯一性、`nmax_generator`、`gen_insertion_dict` | 无 | 无 |
| test_gauge_links_methods.py | `CurrentElementalGenerator` gauge-link 内部方法一致性（`_gauge_links_product` vs `_apply_gauge_links_to_points`） | `gpu` | **集群绝对路径**数据（`test/test_gauge_links_methods.py:55-68`），仓库内不存在 |
| test_group_projection.py | `group_projection.{hadron_little_group_projection, operator_transform}` + `hadron.{gen_correlator, operator_conjugate}` | 无 | 无 |
| test_insertion_simple.py | 对 `insertion` 的结构/元测试（Mock/patch）：类定义、docstring/注解/错误处理存在性、模块规模 | 无 | 无 |
| test_load.py | `GaugeFieldIldg` 读取与 `.npy`/raw binary 逐位一致 | 无 | `weak_field.npy`、`.bin`、`.lime` |
| test_meson_spectrum.py | `correlator.one_particle.{twopoint, twopoint_matrix}`：单算符/多算符/2×2 矩阵 2pt | `gpu` | `weak_field.elemental.npy`、`.perambulator.npy` |
| test_ne_provenance.py | `result_provenance`（manifest/篡改拒绝/worktree 哈希）+ Ne 缓存身份、Ne 越界拒绝、loader 版本重建缓存 | 无 | 无（tmp_path） |
| test_perambulator.py | `PerambulatorGenerator`（PyQuda）单卡生成 | `mpi` | `weak_field.lime`、`.eigenvector.input.npy` |
| test_perambulator_mpi.py | 多进程 MPI 网格分域 `PerambulatorGenerator` + `pyquda_utils.core` | `mpi` | `weak_field.lime`、`.eigenvector.input.npy` |
| test_perambulator_phase3.py | `calc_old` vs `calc_new` 的 VSV/PSV shape/dtype/数值一致、性能、确定性、time-slice 独立性；perambulator 系列唯一有真实 assert 的测试 | `mpi` | `weak_field.lime`、`.eigenvector.input.npy` |
| test_ported_production_hooks.py | `relative_time_index` 越界拒绝、PSP highmode 缓存复用、release 清缓存 | 无 | 无 |
| test_propagator_psv.py | `PropagatorPSVNpy`（惰性 FileData）、`PropagatorPSVTimeslicesNpy`（通配 `.t???.npy`）、缺占位符报错、拒绝裸 int cfg | 无 | 无（tmp_path 自造） |
| test_propagator_with_current.py | `PropagatorWithCurrent` Mock 单元测试：highmode 公式、Ne=0 退化、缓存清理、`Meson`/`Current` 初始化、`usedNe`/`usedNp` 上限 | 无 | 无 |
| test_quark_contract.py | 与 test_flavor_structure.py 核心部分近乎重复（同 import 集、测试项一一对应） | 无 | 无 |
| test_quark_diagram_contraction.py | `QuarkDiagram` 收缩管线：scene 枚举与权重、einsum shape、multitime rolling、相同子图合并、usedNe 支持 | 无 | 无（Mock+小数组） |
| test_quark_draw.py | `quark_draw`：`_vertex_attributes_from_diagram`、dagger→side、自环与对齐、quark 着色、baryon 绘制 | 无 | 无 |
| test_sampling_weight.py | 纯数学：整数分拆、`calculate_sampling_weight` 性质、VSP/PSV/PSP highmode 投影公式、真实晶格参数数值例 | 无 | 无 |
| test_simplify.py | `diagram_simplify`：顶点排序、冗余顶点删除、分量拆分（与 test_diagram.py 部分重叠） | 无 | 无 |
| test_sparsened_point.py | `generator.generate_sparsened_points`：shape/dtype/坐标范围/唯一性、seed 可复现、参数化多尺寸、错误输入拒绝 | 无 | 无 |
| test_spatial_structure.py | `spatial_structure.{HadronIrrepRow, HadronIrrep}`：创建/相等/`transform`/`conjugate` | 无 | 无 |
| test_symmetry.py | `symmetry`：Oh 群公理、generators/irreps 一致性、little group 闭包、reduction map | 无 | 无 |
| test_temporal_current_elemental.py | `DirectedCurrentBasis` + directed raw 生成：周期回绕、开放边界置零、contract JSON 往返与缓存、SU(3) 投影、伪造拒绝 | 无 | 无 |
| test_vertex_type.py | `quark_diagram.vertex_type_from_matrix`：与 `quark_contract` 构造一致的邻接矩阵判定（meson/baryon/标量） | 无 | 无 |

### 6.2 当前测试体系的薄弱点（如实陈述）

1. **七个脚本式测试断言力为零**：test_load、test_elemental、test_eigenvector、test_displacement_elemental、test_perambulator、test_perambulator_mpi、test_gauge_links_methods 均不含 `def test_` 与 assert（grep 计数 = 0）。它们只在收集期执行一遍模块级代码，数值不一致会**静默通过**；是否补 assert 属维护决策。
2. **无引用的遗留数据**：`test/irrep_vertices_t2pp.pkl(.meta.pkl)` 无任何现行测试引用，仅 `test/scripts/t2pp_skeleton.py` 名号相关——候选遗留文件，删除前需确认脚本用途。
3. **断言薄弱却标 integration**：`test/test_current_contraction.py:15` 标 `integration`，但测试体以 print + `return True` 为主（该文件 grep assert 计数为 0），实际校验力有限。
4. **集群路径硬编码**：`test/test_gauge_links_methods.py:55-68` 的数据路径（`/public/home/siyangchen/...`、`/public/share/weiwang/...`）硬编码为集群绝对路径，仓库内不存在对应数据；本地无 GPU 时靠模块级 skip 逃过执行。
5. **test_quark_contract.py 与 test_flavor_structure.py 大面积重复**：两者 import 集与测试项一一对应，重复维护可能产生漂移。
6. **slow marker 注册但从未使用**：`pyproject.toml:66-71` 与 `test/conftest.py:39-42` 声明了 slow，但 grep 全部测试文件无 `pytest.mark.slow`，长耗时测试与快速测试目前无法按 marker 分层筛选。
7. **元测试校验力偏弱**：`test/test_insertion_simple.py` 的断言多为"代码文本存在性"（docstring/注解是否出现）而非物理有效性。
8. **参考数值占位符**：`test/test_perambulator_phase3.py:11,151` 的容差写为 `[PLACEHOLDER]`（docstring 注明"待填入真实参考数据"），相关数值一致性判断尚未闭环。

### 6.3 验证现状

在仓库根目录、无 GPU/MPI 的环境中验证：

- 最小子集 `pytest test/test_gamma.py test/test_sampling_weight.py test/test_vertex_type.py test/test_sparsened_point.py test/test_symmetry.py -q`：**159 passed**。
- 全量收集 `pytest --collect-only -q`：**489 tests collected**，无 collection error——确认 `norecursedirs` 生效、脚本式模块在无 GPU 环境下可安全完成收集期 import（依赖 conftest/模块级 skip 保护）。
