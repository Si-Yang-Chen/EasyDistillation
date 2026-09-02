# Current API v1.2.0

## Responsibility boundary

The Current API describes composable current terms and provides the single lattice-flow assembler for those terms. It does not generate a basis or a fundamental elemental tensor. Existing `ElementalGenerator`/`CurrentElementalGenerator` and contraction data paths remain responsible for raw elemental data.

`localized-blending` and other callers may call this API, but must not duplicate its weighting or temporal-endpoint contract. A caller-supplied resolver applies each term's gamma matrix, displacement, gauge link, dagger operation, offsets, and any basis-specific interpretation.

**CONTRACT-001:** Current constructs `lattice.current.term/v1` mappings; an external resolver converts each mapping into an already-applied elemental value; `assemble_current_terms` is the only component that multiplies by `coefficient * normalization` and sums terms. Current never claims that it generated the resolver's fundamental tensor.

## Public schemas

- API version: `1.2.0`.
- Term schema: `lattice.current.term/v1` (unchanged from v1.0).
- Assembler schema: `lattice.current.assembler/v1`.
- Raw adapter schema: `lattice.current.raw-spatial-displacement-basis/v1`.
- Directed temporal raw schema: `lattice.current.raw-directed-one-link-basis/v1`.

`CurrentTerm.as_dict()` has a fixed 15-key mapping:

1. `schema`
2. `coefficient`
3. `direction`
4. `displacement`
5. `gamma_index`
6. `link`
7. `wilson_r`
8. `spin_structure`
9. `normalization`
10. `bar_offset`
11. `field_offset`
12. `link_origin_offset`
13. `link_dagger`
14. `boundary_policy`
15. `temporal_point_split`

The original v1 fields retain their meanings. New fields have backward-compatible defaults: `normalization=1`, all offsets `(0, 0, 0, 0)`, `link_dagger=False`, `boundary_policy="caller-supplied"`, and `temporal_point_split=False`.

`coefficient`, `normalization`, and non-null `wilson_r` must be finite numeric scalars. `direction` is an integer from `-1` through `3`; all displacements and offsets are four-integer tuples; `link` is `none`, `forward`, or `backward`. Temporal endpoint offsets are valid only for `direction=3`, where `temporal_point_split` must accurately indicate a nonzero temporal endpoint offset.

## Defined terms

Local vector, local axial, and pseudoscalar terms have `bar_offset=field_offset=(0,0,0,0)`, no link, and are not temporally point split.

For each Wilson conserved direction `mu`, let `+mu` be the unit four-vector:

| Term | Coefficient | Bar offset | Field offset | Link origin | Link | Dagger |
| --- | ---: | --- | --- | --- | --- | --- |
| Forward | `-1/2` | `0` | `+mu` | `0` | `forward` | `False` |
| Backward | `+1/2` | `+mu` | `0` | `0` | `backward` | `True` |

The legacy `displacement`, `link`, gamma, spin structure, and Wilson-r values remain compatible: forward displacement is `+mu`, backward displacement is `-mu`. Only `mu=3` terms set `temporal_point_split=True`; spatial terms never change time.

## Endpoint resolution

`resolve_current_term_endpoints(term, *, anchor_time, temporal_extent=None, boundary)` accepts a `CurrentTerm` or a v1 mapping and returns:

- `bar_time`
- `field_time`
- `link_origin_time`
- `temporal_point_split`
- `boundary`

Each raw time is `anchor_time + offset[3]`. Supported policies are:

- `unbounded`: returns raw integer times; no extent is required.
- `periodic`: requires a positive integer extent and reduces every endpoint modulo that extent.
- `open`: requires a positive integer extent and raises `IndexError` if any endpoint is outside `[0, temporal_extent)`; it never wraps.

The term's `boundary_policy` remains `caller-supplied`: the term does not select its own physical boundary condition.

## Resolver and assembler

`assemble_current_terms` calls the supplied resolver exactly once per term as:

```python
resolver(term_mapping, endpoints=endpoints, source_ne=source_ne, sink_ne=sink_ne)
```

The resolver must return a mapping containing:

- `value`: a non-empty, finite numeric array-like value with the same shape for every term.
- `source_ne`: exactly the requested source count.
- `sink_ne`: exactly the requested sink count.
- `provenance`: optional resolver-defined provenance.

The resolver, not the assembler, applies gamma, link orientation/dagger, displacement, and endpoint semantics. The assembler applies only the unique formula

```text
value = sum(term.coefficient * term.normalization * resolved.value)
```

Available and used source/sink Ne are validated independently. Counts must be integers (not booleans), nonnegative, and used counts cannot exceed their respective available counts. Zero is valid. Source and sink used Ne may differ.

The result records assembler schema/version, assembled value, term count, independent source/sink Ne provenance, boundary, anchor time, and for every term its schema, resolved endpoints, and optional resolver provenance.

## Spin-aware current bridge and consumer boundary

`resolve_current_term_spin(term, raw_resolver, ...)` is the authoritative spin-aware resolver for a single `CurrentTerm`. Its callback receives the normal term mapping/endpoints/independent Ne counts and returns an **unweighted**, finite complex spinless V2V matrix with exact shape `(sink_ne, source_ne)`. The bridge constructs and applies exactly one explicit `(4,4)` matrix on leading `(sink_spin, source_spin)` axes:

- `LocalVectorCurrent`: `gamma(term.gamma_index)` for all four components.
- `LocalAxialCurrent`: the explicit `gamma_mu gamma_5` matrix represented by the term gamma index (its established term coefficient remains assembler-owned).
- `PseudoScalarDensity`: explicit `gamma_5`. Its `spin_structure="gamma_5"`; `direction=-1` denotes a density, not a lattice axis.
- `ConservedVectorCurrent`: forward `r - gamma_mu`, backward `r + gamma_mu`; the raw callback remains responsible for the directed link/displacement selection only.

`make_spin_aware_current_resolver(raw_resolver)` is composable with `assemble_current_terms`. `assemble_spin_aware_current(current_or_terms, raw_resolver, ...)` supplies that composition. Thus only `assemble_current_terms` applies `coefficient * normalization`, exactly once; neither raw callback nor spin bridge applies either factor. Raw contributions must be finite complex arrays and must have the exact independent sink/source Ne shape. The returned assembled value has shape `(4, 4, sink_ne, source_ne)`.

`spin_aware_current_adapter(assembled)` provides eager transport:

```python
{
  "schema": "lattice.current.assembler/v1",
  "vertex": value,  # (sink_spin, source_spin, sink_ne, source_ne)
  "axes": ("sink_spin", "source_spin", "sink_ne", "source_ne"),
  "ne": assembled["ne"],
  "term_count": assembled["term_count"],
}
```

`legacy_current_vertex_adapter({time: adapter, ...})` 只服务 **equal-time** 已组装顶点。它返回 `lattice.quark_diagram.CurrentVertexAdapter`，并把 Current 轴顺序 `(sink_spin, source_spin, sink_ne, source_ne)` 转成旧收缩顺序 `(source_spin, sink_spin, source_ne, sink_ne)`。若任一 term 的 `bar_time != field_time`，该 adapter 会拒绝输入，防止把 point-split Current 错误压成单时间顶点。

Temporal/cross-time V2V 使用 `lattice.current_elemental.contract_directed_current_v2v()`。它逐 term 解析端点并调用两个**已加载** VSV accessor：`outgoing.get(field_time, sink_time)` 与 `incoming.get(source_time, bar_time)`；随后计算 `afAi,bfji,bcjC->acAC`，最后才乘 `coefficient * normalization` 并求和。该函数不会调用 `load()`、传播子生成器或反演代码。反向 raw channel 已包含 dagger，消费端不会再次共轭或转置。

`save_directed_current_v2v()` / `load_directed_current_v2v()` 提供生产文件协议：内容寻址 `.npy`、最后原子发布的 `manifest.json`、严格 raw contract、配置号、动量、gauge/eigenvector 来源文件 SHA-256、数据 SHA-256 和消费公式版本。加载默认重新读取并校验来源文件与数据文件；只有显式 `verify_sources=False` 才允许在来源文件未挂载的离线搬运场景仅检查 manifest/data，此模式不能作为来源审计通过证据。

`consume_spin_aware_current(adapter, sink_spin, source_spin)` 仍是最小 eager CPU consumer。以上生产桥目前仅覆盖 V2V；V2P/P2V/P2P 仍需要各自的 point elemental 与 VSP/PSV/PSP 语义。

## Synthetic CPU precheck

`test/current_conservation_cpu_precheck.py` is a standalone, deterministic periodic free-field precheck. It resolves and assembles the real conserved forward/backward terms through the public spin bridge using a unit raw V2V basis, evaluates backward divergence and the zero-momentum temporal charge diagnostic, and writes JSON to stdout (or `--output`). Its pass result is a synthetic algebra/transport check only: it is **not** a real-gauge, propagator-artifact, or remote physics experiment. The JSON contains placeholders for those provenance artifacts and a deliberately perturbed failing diagnostic. A real conservation plot remains a remote-experiment deliverable.


`GaugeLink` remains a spatial-only arbitrary-path encoding. Its six digits, base-five continuation index, three-vector displacement, conjugation modulo six, and cubic transforms are persisted/consumed by legacy elemental paths and are not extended.

`CurrentElementalGenerator.calc_directed_current_raw(boundary="periodic")` instead produces an in-memory v1 **V2V-only** one-link basis with shape `(8, Lt, momentum, sink_ne, source_ne)`, complex dtype, and a required JSON-native contract. Direction order is fixed: `0..7 = +x, +y, +z, -x, -y, -z, +t, -t`; vectors are `(±x, ±y, ±z, ±t)`. The contract declares `channels=["v2v-one-link"]`, schema/version, names/vectors, axes, dtype/shape, boundary, symmetric raw Ne provenance, and a SHA-256 cache identity calculated from all semantic contract fields.

Gauge loader data are interpreted as `(gauge_axis, time, z, y, x, color_row, color_column)`. For a raw anchor `(t,x)`, entry `6` uses the forward link `U_3(t,x)` and the source eigenvector at `t+1`; entry `7` uses `U_3(t-1,x)^dagger` and the source eigenvector at `t-1`. Periodic boundaries wrap `Lt-1 -> 0` and `0 -> Lt-1`; open boundaries set unavailable crossing-link elemental entries to zero. Spatial entries preserve the legacy one-link results.

A Current term anchor and a raw-basis anchor are distinct for backward terms. The forward term at `x` resolves raw `+mu` at its bar endpoint `x`. The backward term at `x` has bar endpoint `x+mu` and resolves raw `-mu` there, producing `U_mu(x)^dagger`; it must not use the term's `link_origin=x` as the raw anchor. Under periodic endpoint resolution this bar endpoint wraps before raw selection. The same relabelling selects the integrated spatial `-x/-y/-z` channel at its bar endpoint; that is consistent with the legacy midpoint phase because the raw V2V construction assigns the half-displacement phase to the selected directed channel.

`resolve_directed_current_raw(raw, contract, term, ...)` validates the exact endpoint fields/types/temporal flag and requires endpoint boundary to equal the raw-contract boundary before selecting raw direction/time/momentum/sink/source. It returns no Current coefficient or normalization. Pass it to `assemble_current_terms`; the assembler remains the sole application of `coefficient * normalization`. The raw generator has symmetric `usedNe`; the resolver can slice to independently requested source/sink counts only within that symmetric generated extent.

Legacy `raw-spatial-displacement-basis/v1` metadata is accepted only by `validate_legacy_spatial_current_raw(raw, contract)`, which is also dispatched by `validate_current_raw_contract(..., require_temporal=False)`. It requires the full v1.1/v1.2 adapter contract emitted by `Current.compute_elemental()`: all four raw channels (`v2v`, `v2p`, `p2v`, `p2p`), `representation="raw-spatial-displacement-basis"`, false term-combination/temporal-support flags, external term application, matching term/assembler schemas, exact axes, actual channel shapes, symmetric available/used/source/sink Ne provenance, requested-Ne bounds, and Np bounds. Numeric V2V/V2P/P2V arrays must be finite complex arrays. P2P remains honestly sparse/list-based: the validator checks identity entries or sparse integer `(N,2)` indices with finite complex `(N,3,3)` values, but does not prove physical completeness of a sparse point set. Schema-only, missing/unknown/forged metadata, wrong shapes/dtypes/nonfinite arrays, or any temporal requirement are rejected.


`Current.compute_elemental()` remains a backward-compatible raw adapter around `generator.calc_all(t)`. It does not apply Current terms and its contract says:

- `combined_with_current_terms=False`
- `term_application="external-resolver-required"`
- `assembler_schema="lattice.current.assembler/v1"`
- `supports_temporal_point_split=False`
- `raw_generator_used_ne_is_symmetric=True`

A single generator output uses the same `generator.usedNe` on source and sink axes. Optional `used_source_ne` and `used_sink_ne` arguments are validated against that actual raw `generator.usedNe` and recorded only as requested provenance; they cannot request modes absent from the raw tensors and do not slice or reinterpret those tensors. Different source/sink Ne are handled by `assemble_current_terms` and its external resolver.

The strict NumPy transport tests use non-Hermitian complex links and nontrivial eigenvectors. With `rtol=atol=1e-12`, they directly assert `E_+(t) = Σ_{x,a,b} V_e(t,x,a)^* U_3(t,x)_{ab} V_f(t+1,x,b)` and `E_-(t) = Σ_{x,a,b} V_e(t,x,a)^* U_3(t-1,x)^†_{ab} V_f(t-1,x,b)`, including interior and periodic wrap times. The backward reference is explicitly distinguished from using `U`, `U.T`, or `U.conj()`.


The unit tests verify schema validation, synthetic NumPy resolver values, assembler weighting, independent Ne propagation, endpoint wrapping/rejection, atomic artifact publication, data/source hash binding, tamper rejection, exact term-wise VSV axes, and the guarantee that the bridge never loads or generates a propagator. They do not validate real gauge configurations, real precomputed VSV files, physical normalization, DCU/GPU execution, MPI/Slurm execution, or the external `localized-blending` integration. Those remain caller/integration validation boundaries.
