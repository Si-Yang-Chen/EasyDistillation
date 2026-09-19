---
name: easy-distillation-correlator-workflow
description: End-to-end workflow for building and evaluating any EasyDistillation correlation function — multi-hadron, multi-point, with or without a current/density insertion. Use this whenever the user wants to compute a correlator, "算关联函数", build interpolating operators for a lattice irrep, set up a two- or three-point function, decide symmetry/isospin/C-parity projections, map abstract vertices onto elemental and perambulator data, or run a per-configuration contraction job. Also use it when the user asks what information they need to supply, or which construction choices are still open, before a correlator run can start — even if they never say "skill" or name the library.
---

# EasyDistillation correlator workflow

## Submission gate

Before constructing outputs, launching a contraction, starting MPI, creating a
Dispatch queue, or submitting a scheduler job, present one final summary and
wait for explicit approval. Include the particle/flavor split, irrep and row,
momentum and C/parity choices, current sector, exact cfg list, prerequisite
files and parameter hashes, time map, output paths, resources, and the chosen
array or Dispatch strategy. A supplied parameter or output path is not
submission approval. Reconfirm after any physics, input, output, or resource
change.

### Backend feasibility gate

Resolve the execution backend from the actual target host before final approval.
A backend that imports successfully is not necessarily suitable for the selected
diagram size. A large contraction should use the available GPU/CuPy path when
the code and host provide it, or the agent must propose an explicitly reduced
mode/time run. Do not start a full-Lt NumPy evaluation merely because it parses.
Record backend, device, estimated contraction size, and an early liveness check;
an interrupted or output-less evaluation is `failed`, not a completed cfg.

## Ensemble-dependent parameters

Never adopt a physics or solver parameter merely because a driver gives it a
default. This includes quark masses such as `ml`, Wilson/clover coefficients,
temporal boundary conditions, stout parameters, solver tolerances, iteration
limits, and momentum or time conventions. Resolve each value from the actual
ensemble metadata, the exact prerequisite-product provenance, or a user-approved
production configuration. Record the source and whether it is a discovered
fact, inherited value, or proposal. If a value cannot be resolved, stop before
building or submitting the correlator and ask the user; do not substitute the
script default. A script default may be used only for a non-physical plumbing
option after explicitly labeling it as a proposal and obtaining confirmation.

## Remote project and data-layout resolution

For CSNS work, do not assume the local checkout or a same-named remote copy is
the code of record. Starting from the user data path, inspect candidate remote
EasyDistillation roots and select the one whose scripts, input prefixes, lattice
layout, ensemble parameters, and prerequisite products match that data. Record
the selected remote project root, relevant entry script, repository revision or
worktree state, and the evidence linking it to the ensemble. If multiple roots
match, show them and ask the user; if none matches, stop before submission.
An existing ensemble-specific driver may be used as the production entry point,
but its physics defaults still require metadata/user confirmation under the
ensemble-parameter rule above.

### Reference-driver and oracle resolution

When the user names or supplies a reference driver, that exact file is an
authoritative read-only input even when it lives outside the working code copy.
The main agent may inspect it to resolve `PROJECT_ROOT`, reference directories,
save calls, filename templates, axis conventions, and parameter mappings. A
delegated implementation agent must not receive the reference driver or its
expected output values unless the user explicitly releases them.

Do not infer the reference location from a smoke output location. Parse the
driver's actual save call and resolve every path variable first, including
project-level directories outside the gauge-data root. A result is comparable
only if its channel, operator set, axes, time/source range, and mode truncation
match. A one-operator smoke array is not a validation of a full driver whose
output has additional operator, momentum, or source-time axes. A shape mismatch
must be reported as `incomparable`, followed by reproduction of the reference
output contract; it must never be called numerical validation.

## Configuration list

Use a `conf_list` or `conf_text` file when the user supplies one. For a
one-cfg smoke test, an unambiguous cfg discovered from the data is enough; if
several cfgs remain, show them and ask which one to use. List files contain one
cfg label per non-empty, non-`#` line. Use `--cfg-list`/`Dispatch` for multiple
cfgs and keep outputs distinct; never invent labels.

One pipeline for every correlator in this library. The shape of the pipeline does not
change when you add a particle, when you go from two-point to multi-point, or when one
vertex is a current insertion instead of a meson: the same five stages always run, and
only the operator content of the stages changes.

```
1 symmetry particles -> irreps -> projected multi-particle operator rows
2 correlator Hadron(rows, flavor) -> gen_correlator -> Diagram expression
3 data bind Elemental / Perambulator loaders + vertex_map / propagator_map
4 per-cfg loop Dispatch -> prepare -> bind -> eval(t_src) -> accumulate
5 save one array per cfg + a provenance sidecar
```

Two rules keep this from becoming a pile of special cases:

- **The vertex is the unit of abstraction.** A vertex is whatever object answers
 `get(t)` — a `Meson` (or `Current`) wrapping an elemental, or a propagator block.
 Adding a current insertion does not add a stage; it adds vertex data.
- **Time is a mapping, not a loop variable.** `calc_diagram_eval` takes a symbolic
 `{time_symbol: actual_time}` map. "Scan the sink time" is `numpy.arange(Lt)` in that
 map, not a Python loop.

## 0. Before anything runs: minimal choices

Ask only for choices that cannot be safely discovered or inferred:

- particles and flavor wavefunctions;
- source/sink or multi-point split;
- target irrep, row, momentum, and gamma/derivative content;
- data location and loader layout (`usedNe`/`usedNp` if truncated); and
- current type and Wilson `r` only when a current is present.

For a simple one-cfg 2pt smoke test, do not ask about current sectors,
sampling compensation, leg offsets, or point data. Introduce those questions
only when the requested correlator uses them. A wrong flavor or irrep is still
material: list legal candidates when the user has not supplied one rather than
silently choosing physics.

The first delegated prompt must contain only the physical objective, explicitly
authorized working/data roots, and a request to inspect inputs and list missing
choices. It must not contain the expected operator formula, diagram count,
output shape, reference path, or expected values. Release those items
progressively after the agent asks or proposes them. Keep the reference-driver
and oracle notes with the main agent for later validation.

## 1. Symmetry construction

### 1.1 Flavor wavefunctions — `HadronFlavorStructure`

Conventions (`lattice/flavor_structure.py`): a 2-character string is a meson with
`flavor_str[0]` the antiquark and `flavor_str[1]` the quark; a 3-character string is a
baryon; `"bar{uds}"` is an antibaryon.

```python
import sympy as sp
from sympy import S
from lattice.flavor_structure import HadronFlavorStructure

# single-flavor building blocks
charmed = [HadronFlavorStructure("dc"), HadronFlavorStructure("uc")]
charmed_bar = [HadronFlavorStructure("cu"), -HadronFlavorStructure("cd")]
hidden_charm = HadronFlavorStructure("cc")
strange = HadronFlavorStructure("ss")
light = HadronFlavorStructure("ll")
iso_scalar = (HadronFlavorStructure("uu") + HadronFlavorStructure("dd")) / sp.sqrt(2)
```

Multi-particle flavor structures are **hand-written**; there is no library helper
that derives them:

```python
# DDbar / DbarD isoscalar combinations
DDbar = charmed[0] * charmed_bar[1] - charmed[1] * charmed_bar[0]
DbarD = -charmed_bar[0] * charmed[1] + charmed_bar[1] * charmed[0]
charm_light = hidden_charm * iso_scalar
```

Prefer exact sympy rationals (`S(2)`, `sp.sqrt(2)`) over Python floats: sympy must keep
the expression in a canonical form, and float coefficients can defeat the simplification
inside `quark_contract`.

### 1.1.1 Charge conjugation factorises — derive the wavefunction factor from the target

The overall C-parity of a two-particle operator is a **product of three factors**, and
only the last one is yours to choose:

```
C_total = C_intrinsic(particle 1) * C_intrinsic(particle 2) * C_wavefunction
```

`C_intrinsic` is **fixed the moment you pick the particle**, because it follows from that
particle's gamma and derivative structure; the library reports it directly:

```python
from lattice.insertion import Insertion, GammaName, DerivativeName, ProjectionName
ins = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, mom_dict)
ins.charge_conjugation # +1 or -1, the intrinsic C of one hadron
```

Measured values for the common channels:

| channel | J^PC | intrinsic C |
|---|---|---|
| `PI` (pseudoscalar) | 0-+ | **+1** |
| `RHO` (vector) | 1-- | **-1** |
| `A0` | 0++ | +1 |
| `B1` | 1+- | -1 |
| `A1` | 1++ | +1 |
| `B0` | 0+- | -1 |

So the workflow runs **backwards from what is natural to write down**: start from the
target channel's C-parity, divide out the two intrinsic factors, and *that* quotient is
the wavefunction C-parity the flavor structure must carry:

```
C_wavefunction = C_target * C_intrinsic(1) * C_intrinsic(2) # C is self-inverse
```

For a fermion-antifermion pair the wavefunction factor is the familiar
`C_wf = (-1)^(L+S)` (two spin-0 bosons reduce to `(-1)^L`). Worked checks against known
physics:

| system | intrinsic product | C_wf | C_total | cross-check |
|---|---|---|---|---|
| `rho -> pi pi` (P wave, L=1) | `(+1)(+1) = +1` | `(-1)^1 = -1` | **-1** | equals rho's intrinsic C, as it must |
| `D* Dbar`, target C=+ | `(-1)(+1) = -1` | **-1** | +1 | needs a C-odd wavefunction |
| `D* Dbar`, target C=- | `(-1)(+1) = -1` | **+1** | -1 | needs a C-even wavefunction |
| `D Dbar` / `D* Dbar*`, target C=± | `(+1)(+1) = +1` | `±1` | `±1` | wavefunction carries the sign |

Once the required wavefunction parity is known, build the C eigenstate with the
projector `(1 -/+ C)/2`:

```python
I0_Cm = (DDbar - C(DDbar)) * (1 / S(2)) # wavefunction C = -1
I0_Cp = (DDbar + C(DDbar)) * (1 / S(2)) # wavefunction C = +1
```

**Writing `C` yourself is the trap.** There is no flavor-level charge-conjugation helper
in the library; the one function whose name suggests otherwise is
`lattice.hadron.operator_conjugate`, and **it is the dagger, not C** -- it drives the
sink-side conjugation and, because `HadronFlavorStructure` is non-commutative, it
reverse-orders the factors (`(AB)dagger = Bdagger Adagger`). Using it as C is wrong, and
wrong in a way that looks fine: both C eigenstates come back dagger-even, so the C-odd
combination is silently lost. Write C as an order-preserving map that conjugates each
meson and leaves the coefficients alone:

```python
def C(expr):
    """Order-preserving charge conjugation: C(A*B) = C(A)*C(B)."""
    out = S.Zero
    for term in Add.make_args(sp.expand(expr)):
        coeff, factors = S.One, []
        for factor in Mul.make_args(term):
            if isinstance(factor, HadronFlavorStructure):
                factors.append(factor.conjugate())   # meson -> its antiparticle
            elif factor.is_number:
                coeff *= factor
            else:
                coeff *= C(factor)
        out += coeff * Mul(*factors)
    return sp.expand(out)
```

Verified properties of this `C`, all checked on the expressions above:
`C(DDbar) == DbarD`, `C` is linear (`C(I*x) == I*C(x)`), and `C` is an involution
(`C(C(x)) == x`) -- the last two are what make `(1 -/+ C)/2` a genuine projector.

Because the mesons do not commute, a C-odd isoscalar built from one flavour pair is
really a **commutator of each charge-conjugate pair**, not a symmetrisation. Writing
`[A,B] = AB - BA` and naming the mesons physically (`D+ = c dbar`, `D- = cbar d`,
`D0 = c ubar`, `Dbar0 = cbar u`), the `I0_Cm` above expands to

```
I0_Cm = (1/2) ([D-, D+] + [Dbar0, D0])
```

which is manifestly C-odd: `D- <-> D+` with the order reversed flips each commutator's
sign. This is why factor order in a hand-written flavor structure matters and cannot be
reordered freely -- and note the library's `charmed_bar` stores the minus sign inside the
entry (`-HadronFlavorStructure("cd")`) rather than in the flavor string, so expand the
expression before reading off which meson is which.

### 1.2 Spatial wavefunctions — `HadronIrrep` and its rows

```python
from lattice.base_types import Tag
from lattice.spatial_structure import HadronIrrep

# name, momentum, little-group irrep at that momentum, parity, tag
D_star = HadronIrrep("D_star", [0, 0, 0], "T_1", -1, Tag(0, 0))
row0 = D_star[0] # HadronIrrepRow
rows = list(range(D_star.lenth)) # 1 for A/E... see the note below
```

`HadronIrrep.lenth` is derived from the **name prefix** (`lattice/spatial_structure.py`):
`T*` → 3, `G*`/`E*` → 2, `H*` → 4, anything else → 1. That rule agrees with the actual
dimension of every irrep in the little-group tables below (`T_1`→3, `E`→2, `G_1`→2,
`H`→4, `F_1`→1, `B_1`→1), so use `lenth` rather than hardcoding a table.
`parity=None` means "no parity tracking" and is only legal for moving momenta.

The irreps actually available, from the hardcoded little-group tables
(`lattice/symmetry/hardcoded_rep.py`, surfaced via `genLittleGroupIrrep`):

| reference momentum | irreps | dimensions |
|---|---|---|
| `[0,0,0]` | `A_1 A_2 E T_1 T_2 G_1 G_2 H` | 1 1 2 3 3 2 2 4 |
| `[0,0,1]` | `A_1 A_2 B_1 B_2 E G_1 G_2` | 1 1 1 1 2 2 2 |
| `[0,1,1]` | `A_1 A_2 B_1 B_2 G` | 1 1 1 1 2 |
| `[1,1,1]` | `A_1 A_2 E F_1 F_2 G` | 1 1 2 1 1 2 |
| `[0,1,2]`, `[2,1,1]` | `A_1 A_2 F_1 F_2` | 1 1 1 1 |

Asking for an irrep outside this table (e.g. `T_1` at `[0,0,1]`) fails inside the
little-group machinery — check the table before requesting.

### 1.3 Multi-particle operator rows — projection and row selection

For a **single** hadron, its own rows are the operators. For a **pair**, the row product
must be projected onto the target irrep:

```python
from lattice.group_projection import hadron_little_group_projection

rows = hadron_little_group_projection(
    [D_star_irrep, D_irrep],   # HadronIrrep list, in the order they appear in the correlator
    "E",                       # target irrep
    0,                         # target row
    parity=None,               # None for moving momenta
)
```

This returns a **list of independent projections**, and the library does not tell you how
many are physically meaningful. Enumerate them for the user and let them choose — this is
question 4 in §0. Measured behaviour on real inputs:

| pair (momentum, irrep) | target | # independent projections |
|---|---|---|
| `D*[0,0,1]A_1 × D[0,0,-1]A_2` | `A_1` | 1 |
| `D*[0,0,1]A_1 × D[0,0,-1]A_2` | `E` | 1 |
| `D*[1,1,0]A_1 × D[0,-1,-1]A_2` | `A_1` | 1 |
| `rho[0,0,0]T_1 × pi[0,0,0]A_1` | `E` | **0 → rejected** |

`hadron_little_group_projection` computes the little group of the **total** momentum, so
a moving pair gets the moving little group automatically. The reference momenta in
`refRotateDict` are the ones all momenta are mapped onto; a momentum outside the covered
set raises `NotImplementedError`.

**Guard every projection like this** (the at-rest case returns a numeric constant, which
later becomes a vertex with zero quark lines):

```python
import sympy as sp
from lattice.spatial_structure import HadronIrrepRow

def keep_valid_rows(rows, n_hadrons):
    out = []
    for expr in rows:
        terms = sp.Add.make_args(sp.sympify(expr).expand())
        if terms and all(
            sum(1 for f in sp.Mul.make_args(t) if isinstance(f, HadronIrrepRow)) == n_hadrons
            for t in terms
        ):
            out.append(expr)
    if not out:
        raise ValueError("projection is trivially zero: use non-zero single-hadron momenta")
    return out
```

### 1.4 Package as `Hadron`

```python
from lattice.hadron import Hadron

hadrons = [Hadron(row, flavor) for row, flavor in zip(rows, flavor_wavefunctions)]
```

`Hadron` carries exactly two things: an irrep-row expression and a flavor expression.
Time and dagger are applied later by `gen_correlator`.

## 2. The correlator expression

```python
from lattice.hadron import gen_correlator

correlator = gen_correlator(
    [src_hadrons, snk_hadrons],   # one list per "half"; two halves for a 2pt
    time_slice_list=[0, 1],       # symbolic time handed to each half
    dagger_list=[False, True],    # the sink half is conjugated
)
```

`gen_correlator` (`lattice/hadron.py`) returns a **numpy object array** whose shape is
`(len(half_0), len(half_1),...)`, one entry per operator combination; each entry is a
sympy expression over `Diagram` symbols. For three halves you get a rank-3 array — this
is the `all_correlator` structure the user described, and it is what
`calc_diagram_prepare` and `_replace_diagrams` handle natively.

Since `gen_correlator` takes one list per half, **any partition of the particles into two
groups works**, which is the general rule: for scattering use source/sink; otherwise
split so that the insertion sits alone on one side.

Drop diagrams whose quark lines carry a tag the user wants excluded — typically the
disconnected local line:

```python
from lattice.quark_diagram import remove_disconneted_diagram
correlator = remove_disconneted_diagram(correlator, [r"S^q_\mathrm{local}"])
```

The tag vocabulary is generated by `quark_contract` (`lattice/quark_diagram.py`):
`S^<flavor>` for a non-local line, `S^<flavor>_\mathrm{local}` when both ends share a time,
and degenerate `u`/`d` collapse to flavor `q`. Observed for a few flavor contents:

| flavor | tags produced |
|---|---|
| `ud` / `du` | `S^q` |
| `ds` | `S^q`, `S^s` |
| `cc` | `S^c`, `S^c_\mathrm{local}` |

So **the tags follow from the flavor wavefunction, not from a fixed list**: read them off
`{d.propagator_list for d in expr.atoms(Diagram)}` before writing the `propagator_map`.

Symmetry averaging: transform the whole expression once per group element and keep the
results as a list; average the numbers after eval.

```python
from lattice.group_projection import operator_transform
correlator_list = [operator_transform(correlator, ge) for ge in group_elements]
```

Group element names are the 96 $O_h^D$ elements in `lattice/symmetry/hardcoded_rep.py`
(`iden`, `c4x`, `c4y`, `c4z`,...). Each transform can rotate momenta, so every rotated
momentum must exist in the `mom_dict`/elemental, or `InsertionRow.__call__` raises.

## 3. Mapping abstract vertices onto data

This is the stage users most often get wrong, because it is where "a hadron" becomes "a
specific file plus a specific operator content".

### 3.1 The three maps

| map | keys | values | built when |
|---|---|---|---|
| `propagator_map` | quark-line tags, e.g. `"S^q"`, `"S^c_\mathrm{local}"` | `Propagator` / `PropagatorLocal` / `PropagatorWithCurrent` | once, before `prepare` |
| `vertex_map` | a `HadronIrrepRow` skeleton | loaded `Meson` / `Current` | per cfg, before `bind` |
| `time_map` | symbolic time ints from `gen_correlator` | actual time (int, or `numpy.arange(Lt)`) | per eval |

`propagator_map` is cfg-independent in structure but its objects hold per-cfg caches, so
you build it once and call `.load(cfg, usedNe)` inside the cfg loop.

### 3.1.1 Explicit inputs the pipeline will not guess

Three things cannot be read off the expression or the data, so they are constructor
arguments. Getting them wrong is silent, which is why they are worth stating plainly.

**Which legs split in time (`leg_offsets`).** One entry per `vertex_list` element: `None`
for a vertex that keeps both legs on its anchor, or its offset classes as
`offset_classes` returns them, `[(left_delta, right_delta), [term indices]], ...`.

```python
from lattice.quark_diagram import offset_classes

# Ask the vertex which classes it offers rather than hard-coding them:
# a plain meson yields [((0, 0), [None])]; the temporal conserved current yields
# [((0, 1), [0]), ((1, 0), [1])]; the spatial one collapses to [((0, 0), [0, 1])].
classes = offset_classes(vertex)

diagram = QuarkDiagram(adjacency, vertex_list=..., L=..., usedNp=...,
                       leg_offsets=[None, classes])
```

Two consequences worth knowing. The vertex must also be **marked current-capable** in
`vertex_list`, because that marking is what drives the state expansion the classes live
in; declaring classes for an unmarked vertex expands nothing. And the classes are settled
when the skeleton is built, not at evaluation, because they decide the pool partition and
therefore the scene enumeration -- which is also why the cost is the **sum** of scene
counts over class combinations, not their product.

**Which compensation scheme (`compensation`).** Coefficients are solved from
`Z^T c = w`; the scheme chooses `w`.

| scheme | denominator | character |
|---|---|---|
| `CompensationScheme.EXPECTED` (default) | expected k-tuple count | design-unbiased for a fixed gauge field |
| `CompensationScheme.OBSERVED` | observed k-tuple count | a ratio, so biased in general; does not assume equal-probability sampling |

Do not compute the coefficients yourself: the library derives them as exact rationals,
and they depend on `(M, Np, r)`. Schemes must not be mixed within one expression.

**Where observed counts come from (`count_source`).** Required by the observed scheme
only. A callable `(positions, blocks, M, Np) -> int` giving how many k-tuples of distinct
coordinates the data really holds for that scene. The pipeline cannot know it -- only the
data does -- and a zero count raises rather than returning a coefficient, because the term
did not enter the sample and its weight is undefined.

### 3.2 Vertex data: `Elemental` → `Insertion` → `Operator` → `Meson`

```python
from lattice import preset
from lattice.insertion import Insertion, Operator, OperatorDisplacement
from lattice.insertion import GammaName, DerivativeName, ProjectionName
from lattice.insertion.mom_dict import momDict_mom1, momDict_mom3, momDict_mom9
from lattice.quark_diagram import Meson, Current

elemental = preset.ElementalNpy(prefix, suffix, shape=[num_disp, num_mom, Lt, Ne, Ne], tot_ne=Ne)

ins = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom1)
ins = ins.little_group_projection([0,0,1], "A_1") # no-op at zero momentum
row = ins[0](0, 0, 1) # InsertionRowMom (momentum by dict lookup)
operator = Operator("pi", [row], [1.0])

meson = Meson(elemental, operator, source=True) # source=True -> dagger
meson.load(cfg_key, usedNe)
```

Key details, all verified against the code:

- `InsertionRow.__call__(npx, npy, npz)` resolves the momentum by **index into the
 dict's values**, not by arithmetic (`lattice/insertion/__init__.py`). The dict must
 be the one the elemental was generated with. `momDict_mom9` starts at `"0 0 -3"`, not
 `"0 0 0"`; `momDict_mom1` is the single `"0 0 0"` entry; `momDict_mom3` covers all 27
 momenta with `|n_i| ≤ 1`. Use `momDict_mom1` for at-rest work.
- Enums are spelled `GammaName.PI`, `DerivativeName.IDEN`, `ProjectionName.A1`
 (**no underscore** in the enum member: `A1 A2 E T1 T2`).
- Gamma members: `A0 B0 PI PI_2 RHO RHO_2 A1 B1`. Derivative members:
 `IDEN NABLA B D E`, where `IDEN→0`, `NABLA→dx,dy,dz`, `B→dydz-dzdy,...`,
 `D→dydz+dzdy,...`, `E→dxdx-dydy,...`.
- `OperatorDisplacement(name, rows, coeffs, distances)` is the variant for a displacement
 elemental; it asserts the row's derivative index is 0 and rewrites it
 (`lattice/insertion/__init__.py`).
- `source`/`dagger`: the sink-side vertex is built with `dagger=True`. In the workflow
 driver this comes from the row skeleton's own `dagger` flag, so the `vertex_map` never
 has to guess.

### 3.3 Which propagator data a current insertion needs — the short rule

A **current insertion is a vertex whose two legs may be low-mode (`v`) or sampled point
(`p`)**. That gives four sectors, and the required data follows mechanically:

| sector | what the graph asks for | data you must supply |
|---|---|---|
| `vv` | `Current.get(t)` | `Elemental` (V2V) |
| `vp` | `Current.get_v2p(t)` + `PropagatorWithCurrent.get_PSV_highmode` | V2V + `CurrentElementalV2P` |
| `pv` | `Current.get_p2v(t)` + `PropagatorWithCurrent.get_VSP_highmode` | V2V + `CurrentElementalP2V` |
| `pp` | `Current.get_p2p(t)` + `PropagatorWithCurrent.get_PSP_highmode` | V2V + `CurrentElementalP2P` + `OverlapMatrix` |

Read the sector off the **propagator type**, which the graph derives from the vertex
states (`lattice/quark_diagram.py`):

- prop type `VSV` ⇒ both ends low-mode
- `PSV` ⇒ **sink end is a point** (`S_{point,low}`)
- `VSP` ⇒ **source end is a point** (`S_{low,point}`)
- `PSP` ⇒ both ends points

So the "do I need VSP/PSV/PSP?" question reduces to: *are any of the current's legs
point-sampled, and on which side?* If you only ever want the low-mode matrix element, the
`vv` sector needs just the V2V elemental and a plain `Propagator` — no `PropagatorWithCurrent`
at all:

```python
# vv-only: works today
cur = Current(v2v_elemental, operator, source, v2p_data=None, p2v_data=None, p2p_data=None)
```

`Current.load` requires `p2v_data` unconditionally (`lattice/propagators.py`), so even
the `vv`-only construction must pass *something* as `p2v_data`; a plain V2V-only route is
better expressed as a `Meson` when the vertex content is just a Dirac structure.

`PropagatorWithCurrent` composes the pieces:

```python
from lattice.quark_diagram import PropagatorWithCurrent
prop = PropagatorWithCurrent(
    vsv=perambulator,                # Perambulator loader
    vsp=None, psv=None, psp=None,    # optional PropagatorVSP/PSV/PSP loaders
    overlap_matrix=overlap,          # required for high-mode projection
    Lt=Lt,
)
```

`PropagatorWithCurrent.load(cfg, usedNe, usedNp)` requires consistent `Ne`/`Np` across all
supplied loaders and an `overlap_matrix` (`lattice/propagators.py`). High-mode
projection is applied by `get_VSP_highmode` / `get_PSV_highmode` / `get_PSP_highmode`,
which are what the graph calls for the mixed sectors — never the raw blocks.

For a **conserved** current, get the term list from the library rather than writing gamma
matrices by hand, and use the $r$ the perambulators were generated with:

```python
from lattice.insertion.current import (ConservedVectorCurrent, LocalVectorCurrent, LocalAxialCurrent, PseudoScalarDensity)
terms = ConservedVectorCurrent(wilson_r=r).terms[6:8] # temporal, the two point-split terms
terms = ConservedVectorCurrent(wilson_r=r).terms[0:6] # spatial, equal-time
```

### 3.4 File-format reference

| object | class | disk shape |
|---|---|---|
| meson elemental | `ElementalNpy` (`preset.py`) | `[num_disp, num_mom, Lt, Ne, Ne]` |
| current elemental V2V | `CurrentElementalV2V` (`preset.py`) | `[Lt, num_disp, num_mom, Ne, Ne]`, reindexed to `(disp, mom, t, e1, e2)` |
| current elemental V2P | `CurrentElementalV2P` (`preset.py`) | `[Lt, num_disp, Ne, Np, Nc]` |
| current elemental P2V | `CurrentElementalP2V` (`preset.py`) | `[Lt, num_disp, Np, Nc, Ne]` |
| current elemental P2P | `CurrentElementalP2P` (`preset.py`) | sparse, per time slice; `type='identity'` or `'sparse'` |
| perambulator | `PerambulatorNpy` (`preset.py`) | `[Lt, Lt, Ns, Ns, Ne, Ne]` |
| overlap matrix | `OverlapMatrixNpy` (`preset.py`) | `[Lt, Ne, Np, Nc]` |
| VSP / PSV / PSP | `PropagatorVSPNpy` / `PropagatorPSVNpy` / `PropagatorPSPNpy` | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` / `... Np, Nc, Ne` / `... Np, Nc, Np, Nc` |

Timeslice-slab variants (`*TimeslicesNpy`) exist for all of these; they index a
`{prefix}{cfg}.t{src:03d}{suffix}` file per source time.

**Verify, don't infer.** `Propagator` stores only relative times up to the stored `Dt`,
and refuses anything beyond (`lattice/propagators.py`). A "source-time-rank-slab"
layout is common and covers every `(source, sink)` pair exactly once, but read as a
stride-$n$ grid it looks like part of the time extent is missing. Check
$\gamma_5$-hermiticity against the loader before trusting a number.

## 4. The per-configuration loop

```python
from lattice.dispatch import Dispatch
from lattice.quark_diagram import calc_diagram_prepare, calc_diagram_bind, calc_diagram_eval

# once, before the cfg loop: cfg- and time-independent skeleton
prepared = calc_diagram_prepare(
    correlator_list,              # one entry per group element
    propagator_map=propagator_map,
    vertex_map=None,
    timing=timing_dict,
)

for cfg_key in Dispatch("cfglist.txt", "run_20260918"):   # or a plain list
    for handle in propagator_map.values():
        handle.load(cfg_key, usedNe)
    calc_diagram_bind(prepared, vertex_map_for_cfg(cfg_key))   # loads vertex caches

    acc = None
    for t_src in range(Lt):
        time_map = {sym: (t_src if role == "src" else t_snk_arange)
                    for sym, role in zip(times, roles)}
        values = calc_diagram_eval(prepared, time_map)
        # values: one entry per group element; average them, roll, accumulate
```

Division of labour, exactly as the library intends:

| call | depends on | run |
|---|---|---|
| `calc_diagram_prepare` (`quark_diagram.py`) | expression + propagator handles | **once** |
| `calc_diagram_bind` (`quark_diagram.py`) | cfg (vertex caches) | **once per cfg** |
| `calc_diagram_eval` (`quark_diagram.py`) | `time_map` | **once per t_src** |

`Dispatch` (`lattice/dispatch.py`) hands out one cfg line at a time under a file lock,
broadcasting the same line to all MPI ranks in the job, so a job array and MPI ranks both
work. It leaves a `<cfglist>.<rand>.tmp` queue file next to the list; keep it while any
job is live.

`calc_diagram_eval` returns a nested structure with the **same shape as the input
`all_correlator`**, with `Diagram` symbols replaced by numbers. Converting it to a plain
complex array is the one helper worth writing:

```python
def to_numeric(obj, lt):
    out = backend.zeros(obj.shape + (lt,), dtype=backend.complex128)
    for idx in numpy.ndindex(obj.shape):
        v = backend.asarray(obj[idx])
        out[idx] = complex(v) if backend.ndim(v) == 0 else v
    return out
```

## 5. Saving

One file per cfg plus a provenance sidecar. The library's manifest machinery
(`lattice/result_provenance.py`) is available if the run needs auditable hashes: it
records the git commit, a worktree hash, input file hashes and the parameter set, and
`load_result_manifest` verifies outputs against their hashes. For lighter runs a
`.meta.json` sidecar carrying `usedNe`, `Nop`, group elements, branch summary, diagram
counts and timings is enough — and is what the reference driver writes.

For production, keep a persistent manifest beside the results with parameter
hash, cfg, job/array ID, Dispatch suffix, output paths, status
(`planned/submitted/running/failed/needs-validation/validated`), attempts, and
validation metrics. Choose either explicit Slurm array cfg selection or one
uniquely suffixed MPI `Dispatch` queue; do not combine them accidentally. Inspect
active jobs and matching `.tmp` queues before reuse, retain a live queue, and
call `clear()` only after all workers stop. Resume only after checking scheduler
state and existing products; do not silently alter operators, data mappings, or
resources after failure.

For every validation run, record both a `smoke_output_contract` and a
`reference_output_contract` containing shape, axes, dtype, cfg, time/source
range, and operator set. Mark a run `validated` only when these contracts
agree or a documented matching subset is compared numerically. A successful
command with a different contract is `generated/incomparable`, not
`validated`.

### Delegated remote-edit and Dispatch gate

Before delegating implementation, verify that the delegated agent's execution
host can read and write the selected remote code root. A path visible only to
the main agent is not an editable project copy. If it is not writable by the
agent, transfer a content-addressed task copy or provide a patch handoff and
record the resulting fingerprint; do not silently have the main agent finish
the implementation while reporting delegated completion.

For a one-cfg test, require explicit `--cfg` mode and verify that no Dispatch
queue or `.tmp` file is created. For a Dispatch-capable driver, run a separate
queue preflight with a one-line temporary cfg list and a unique suffix, inspect
the queue before and after, and remove only that exact stale test queue after
the worker exits. Never let a smoke run consume a production queue.

Treat a group-element axis as labeled data. Record the exact ordered group list
in the sidecar and align reference arrays by names before comparing numbers;
"twelve rotations" or a set with the same cardinality is insufficient. A
changed rotation representative can preserve shape while changing selected
rows, so an unlabeled positional comparison is invalid.

Mode counts are object-specific, not necessarily one global `usedNe`. A mixed
flavor meson may use an elemental loaded with the light-sector truncation while
its charm propagator retains its own larger truncation; the contraction code
determines which leading mode block is consumed. Validate the tuple
`(vertex type, elemental usedNe, each propagator usedNe)` against the actual
loader and contraction code before declaring it incompatible. Do not reject a
mapping merely because the elemental and all propagator mode counts differ;
also do not assume the mapping without checking the driver or loader contract.

## Worked minimal case (verified end to end, weak_field data, Lt=8, Ne=20)

```python
# 2pt pion: symmetry -> correlator -> prepare -> bind -> eval
set_backend("numpy")
row = HadronIrrep("pi", [0,0,0], "A_1", -1, Tag(0,0))[0]
corr = gen_correlator([[Hadron(row, HadronFlavorStructure("ud"))]] * 2, [0,1], [False, True])

propagator_map = {"S^q":                    Propagator(peram, Lt),
                  r"S^q_\mathrm{local}":    PropagatorLocal(peram, Lt)}
prepared = calc_diagram_prepare([operator_transform(corr, "iden")],
                                propagator_map=propagator_map)
# -> n_diagrams=1, n_irrep_vertices=2

ins = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom1)
def vertex_map(v):
    m = Meson(elem, Operator(v.hadron_name, [ins[0](0,0,0)], [1.0]), v.dagger)
    m.load("weak_field", 20); return m

for h in propagator_map.values(): h.load("weak_field", 20)
calc_diagram_bind(prepared, vertex_map)
out = calc_diagram_eval(prepared, {0: 0, 1: numpy.arange(Lt)})
# -> list of 1 group element; out[0][0,0] has shape (8)
```

**Multi-point with an equal-time insertion works today**: replacing the middle half's
flavor with `"dd"` (so the $d$ line is the one carrying the insertion) gives a genuine
three-half `gen_correlator` call, and a `Meson`/`Current` vertex at the middle vertex
evaluates — I measured `(8)` output for
`<pi_src(du) · cur(dd) · pi_snk(du)>` with `time_map = {0:0, 1:2, 2:arange(8)}`. Note the
flavor split is what puts the insertion on a specific quark line; $\bar u d \cdot \bar d d
\cdot \bar d u$ keeps the $u$ spectator connected and the $d$ line inserted.

## Pitfalls

- **At-rest two-hadron projections are trivial.** `rho[0,0,0]T_1 × pi[0,0,0]A_1 → E`
 returns **zero** independent projections, and the library's own projection can also
 return a numeric constant. Both become "a vertex with zero quark lines" and later crash
 `quark_contract`. Use moving momenta and validate the rows (§1.3).
- **`t_snk` must be a `numpy` array.** Scanning is detected with `isinstance(time, np.integer)`,
 so a backend array is not recognised as a scan.
- **One scanned time per run.** Two array-valued entries in `time_map` is a semantic
 conflict (each element would need its own per-term offsets), and it raises with a
 reason. Scan one vertex; fix the others.
- **Point sectors need a vertex that can serve them, and a leg that can serve them.**
 A graph expanding into point sectors asks the *vertex* for `get_v2p`/`get_p2v`/`get_p2p`
 and the *propagator* for `get_*_highmode`. A plain `Meson` vertex fails with
 `AttributeError`; a plain `Propagator` fails the same way on the leg. Build the marked
 diagram first and read off the sectors it will request, then pick handles that cover them.
- **The high-mode complement belongs to the leg, not the vertex block.** A point-ended leg
 replaces low-mode content the `vv` sector already carries, so it must be projected --
 and only the propagator holds the overlap matrix needed for that. Do not try to project
 a vertex block: it is the operator's matrix element, and the object does not even hold
 the matrix.
- **Array times work for a scanned sink, not a scanned source.** For VSP and PSV a scalar
 source with an array sink matches the pointwise calls exactly, which is the direction the
 driver uses. Scanning the source, and either PSP array direction, are unverified and
 raise with a reason rather than returning a number -- loop the scanned time if you need
 them.
- **`Current.load` demands `p2v_data` even for the `vv` sector.** A vertex meant only for
 the low-mode sector is better expressed as a `Meson`.
- **Baryon and meson diagrams travel the same path.** Nested baryon adjacency entries are
 remapped recursively, so a baryon-source/baryon-sink matrix reaches the skeleton like any
 other.
- **A connected contraction group holds at most six quark lines.** Each propagator consumes
 two of the 13 spin slots, so four-, five- and six-line rings work and a seven-line ring
 exceeds the alphabet. It raises an error naming the limit.
- **`InsertionRow` momentum lookup is positional.** See §3.2.
- **Flavor powers break sympy.** A two-hadron flavor like `ud*ud` becomes `du(1)**2` after
 conjugation and `simplify` inside `quark_contract` dies in `HadronFlavorStructure.__new__`.
 Keep conjugated factors distinct (`ud*du`), or use summed structures.
- **`PropagatorLocal` requires `t_source == t_sink`**: local lines only appear at one
 symbolic time. Keep them on the same side, or remove them.
- **`operator_transform` rotates momenta symbolically**, so every rotated momentum must
 exist in the `mom_dict`/elemental. With partial-momentum data restrict channels to
 momenta invariant under the chosen group elements.

## Reporting back to the user

Before running anything, report back the closed list of construction choices — particles
and their split, target irrep and row, the hand-written flavor wavefunction with its
isospin/C-parity factor, the selected projection rows, and which propagator data the
insertion needs by sector. Ambiguity here is the normal case, not an exception; surfacing
the alternatives is the job.

## Production prerequisite routing

Before production, inspect the required cfg list, gauge files, metadata, and branch-specific products. If a prerequisite is missing, stop before submission and identify the producer skill: use easy-distillation-eigensystem for Laplace eigenvectors, easy-distillation-sparsened-point for point tables, easy-distillation-elemental for ordinary/current elementals, and easy-distillation-perambulator for VSV/PSV/PSP or density/generalized blocks. Propose the missing generation step and its parameters; do not silently create random or default prerequisite data. After generation, validate and record its path, parameter provenance, and content hash in the production manifest.
