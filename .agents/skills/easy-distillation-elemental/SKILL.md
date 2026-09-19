---
name: easy-distillation-elemental
description: Generate lattice QCD distillation elemental matrix elements (ElementalGenerator V-dagger phi V and CurrentElementalGenerator v2v/v2p/p2v/p2p blocks) with the EasyDistillation framework, including the full parameter branch matrix, on-disk output contract, and measured runtimes. Use when you need to produce elemental.npy/.meson/.npz files from gauge configurations and Laplace eigenvectors, or when writing driver scripts for the 03/04 elemental production stages.
---

# EasyDistillation elemental generation

## Branch selection

The default branch is `ElementalGenerator` for ordinary elemental matrices.
Enter the `CurrentElementalGenerator` branch only when the user explicitly
requests current elementals, point sources, `v2v`, `v2p`, `p2v`, `p2p`, or all
current blocks. When that branch is selected, read
`references/current-point-blocks.md`; do not load its point-source parameters
for an ordinary elemental request. The current branch is opt-in and must never
be inferred merely because point-source files happen to exist.

## 1. Required interaction

Use the same inspect-then-confirm spirit as the eigensystem workflow, but keep
the interaction adaptive rather than treating it as a fixed form. The user may
give exact files, a directory, a project/experiment root, a scheduler list, or
only a description of where the data lives. If the user gives a broad folder,
search it recursively and identify likely gauge, eigenvector, point-source, and
metadata files from names, extensions, directory structure, and file contents.
Use the smallest safe inspection first; do not load whole large arrays merely to
discover their names or shapes.

Before constructing a command, determine (or propose) the active generator
(`elemental` or `current`) and locate the matching inputs. Discover the
configuration labels, gauge/eigenvector prefixes and suffixes, and
`[Lx, Ly, Lz, Lt]` from filenames or metadata whenever possible. If a
`conf_list`/`conf_text` file is supplied, use it as the authoritative cfg list;
if it is absent, derive candidates from the discovered files. Do not replace an
explicit user selection with a guessed key. When several candidates remain,
show the relevant candidates and ask only the smallest question needed to
disambiguate them.

Inspect inputs before asking for confirmation or printing a final command. Do
not show placeholders such as `<inspect directory>` in a confirmation request.
Inferences are useful for preparing the proposal, but they are not user
approval. Label inferred values clearly and present the resulting configuration
for review.

For a temporary test, use temporary output paths and retain them until all
checks are complete; remove them only within the user's confirmed cleanup
scope. Ask for permanent output paths only for production. Do not
silently enable blending, infer a point source, change the ensemble's smearing
convention, or substitute an unresolved input pattern.

When preparing the confirmation, summarize the discovered configuration in a
compact, readable form. Include the meaning and source of important parameters,
but omit irrelevant fields and do not force the user to fill every line. Show
defaults as reviewable proposals, not as user choices that have already been
accepted. A suitable structure is:

```text
Elemental configuration
- generator: elemental/current — requested elemental or current matrix blocks [user]
- gauge directory: <value> — directory containing gauge files [user/discovered]
- eigenvector directory: <value> — directory containing Laplace eigenvectors [user/discovered]
- point-source directory/table: <value> — required only for current blocks [user/discovered]
- cfg/key: <value> — selected configuration label [discovered]
- lattice: [Lx,Ly,Lz,Lt] — spatial volume and time extent [discovered]
- gauge prefix/suffix: <value> — gauge filename construction [discovered]
- eigen prefix/suffix: <value> — eigenvector filename construction [discovered]
- backend: numpy/cupy — CPU or CUDA execution [proposal]
- num_nabla: <value> — derivative/displacement basis size; higher values cost more [proposal/user]
- momentum: <list> or mom_dict=<N> — momentum phases; current v2v only [proposal/user]
- usedNe: <value> — retained eigenvector columns; lower values reduce smoke-test cost [proposal/user]
- output format/path: <value> — elemental npy/binary or current npy/npz and destination [proposal/derived]
- time slices: <list> — slices to compute; output still contains full Lt [proposal]

Elemental-only
- calc-mode: calc_deriv/calc_disp — derivative or gauge-link displacement basis [proposal]
- stout: nstep=<value>, rho=<value> — link smearing convention [metadata/proposal]
- project-su3: on/off — SU(3) projection before contraction [proposal]
- blending: on/off — stochastic dilution contraction [proposal]
- dilution: tot=<list>, used=<list/int> — required when blending is on [user/proposal]

Current-only (show only for the explicit current/point branch)
- calc-blocks: v2v/v2p/p2v/p2p/all — requested current blocks [proposal/user]
- np: <value> — number of point sources [discovered/user]
- used-np: <value> — retained point sources [proposal/user]

For production and smoke tests alike, inspect first, show the resolved
configuration, and wait for explicit user confirmation before constructing a
command, creating output, or submitting a job. Ask only about missing values
or choices that change the physics, such as generator branch, smearing,
momentum, or `usedNe`; the user may confirm all already-resolved values with a
short reply and does not need to fill out a mandatory questionnaire.
```

Do not ask for options that are irrelevant to the selected branch. For example,
do not request dilution settings when blending is off, do not request point
data for `elemental`, and do not treat `--used-np` as meaningful for
`ElementalGenerator`. The configuration block is a communication aid, not a
mandatory questionnaire.

## 2. Test and production

For a smoke test, propose the repository's `test/weak_field.*` data when it is
available, `backend=numpy`, `usedNe=min(8, Ne)`, `num_nabla=0` or `1`, one or
two time slices such as `[0,1]`, no blending, and temporary outputs. Present
the discovered/proposed configuration and wait for explicit confirmation before
running. Check the full output shapes and dtypes before removing the temporary
directory.

For production, confirm the discovered ensemble prefixes and smearing
convention, normally compute all `Lt` slices, and run one configuration per invocation.
If a broad input folder contains many cfgs, present the discovered count and
representative labels, then process the requested subset or prepare a scheduler
array for all of them. Confirm again whenever changing the selected data,
permanent output, numerical settings, generator branch, or resource plan.

Before any solver, output creation, MPI launch, scheduler submission, or
Dispatch queue creation, present one final summary and wait for explicit
approval. It must include the exact cfg list, generator branch and blocks,
ordered momentum list and source, `num_nabla`, `totNe/usedNe`, time slices,
smearing/projection, backend, output paths and overwrite policy, execution
resources, and the chosen distribution strategy. Ask whether to submit/run with
that summary; a parameter value or output-path approval is not submission
approval. If a result-affecting or resource-affecting field changes, reconfirm.

Driver script: `scripts/gen_elem.py` (argparse over every branch below; run it on
whatever cluster/host holds your gauge data and eigenvectors). Code references are
against EasyDistillation commit `e19a399`.

## Environment (generic requirements; verify at your site)

- A Python interpreter (3.11.x was used for the reference numbers below) with numpy
 1.26.x, scipy 1.11.x, opt_einsum 3.3.x, sympy 1.11.x, cupy 13.x, and
 pyquda/pyquda_utils 0.10.x installed. CUDA must be available for the cupy backend;
 MPI must be available for the QUDA smearing fallback only.
- Repo: unpack the EasyDistillation source archive into a scratch/work directory; the
 package root (`lattice/`, `test/`) is the import root (pass it via `--repo`).
- Backend: `lattice.set_backend("numpy" | "cupy")` BEFORE constructing any generator
 (`lattice/backend.py`). cupy requires `CUDA_VISIBLE_DEVICES=0` explicitly (otherwise
 initialization can fail even on idle systems).
- GPU reference hardware: a single 32GB-class GPU. The cupy elemental path does
 NOT need MPI: `ElementalGenerator.__init__` compiles a stout-smearing CUDA kernel via
 `cupy.RawModule` (`lattice/generator/elemental.py`), and `stout_smear` uses it
 before ever touching QUDA (`lattice/generator/elemental.py`). Only the QUDA
 smearing fallback needs pyquda + MPI (an MPI launcher binary must be on PATH; on some
 modulefile-based sites it is only loadable in interactive shells).
- Batch: submit array jobs with your site's scheduler; a reasonable CPU starting point
 is 1 task with ~60G memory. Avoid queueing long-running jobs on GPU-only partitions
 (they are usually oversubscribed); prefer CPU partitions for production elementals.

## 3. Workflow

1. **Staging**: unpack the EasyDistillation source archive into a scratch dir
 (e.g. `<work-root>/<task>/`); copy `scripts/gen_elem.py` there.
2. **Smoke**: run interactively with the repo's `test/weak_field.*` data
 (4^3 x 8, Ne=20), 1-2 timeslices, small `--num-nabla`. Confirms backend, file paths,
 and output shapes in seconds.
3. **Production**: run per configuration with full `Lt` and the ensemble's real
 prefixes (Laplace eigenvector dirs, sparsened-field dirs, gauge `.lime` files).
4. **Job array**: one job per configuration, submitted with your site's scheduler.
 Example SLURM header (replace `<partition>` with your site's partition name and
 `<work-dir>` / `<python>` with your paths):

```bash
#!/bin/bash
#SBATCH -p <partition>
#SBATCH -n 1
#SBATCH --mem 60G
#SBATCH -o logs/elem_%A_%a.out
cd <work-dir>
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" cfglist.txt)
<python> gen_elem.py... --key "$CFG"
```

## 4. Usage

```
python gen_elem.py --repo <repo-root> --generator elemental --latt Lx Ly Lz Lt \
    --gauge-prefix <dir>/ --gauge-suffix .lime --eigen-prefix <dir>/ \
    --eigen-suffix .npy --tot-ne 70 --num-nabla 3 \
    --momentum 0,0,0 0,0,1 ... --out-prefix <out-dir>/ --key <cfg>
```

Full `--help` lists every option. `--mom-dict N` is an alternative to
`--momentum` and takes the ordered list from `mom_dict_to_list(N)`
(`lattice/insertion/mom_dict.py`). Prefer the repository's momlist generator or
the user's supplied generator over manually rewriting an equivalent list. The
resolved ordered list, its source, and its count must be shown during parameter
confirmation and recorded with the output metadata.

## Momentum lists and cfg Dispatch

Keep momentum enumeration separate from configuration/task distribution.
`mom_dict_to_list(N)` converts the project's `momDict_mom1`, `momDict_mom3`,
`momDict_mom9`, or `momDict_test` dictionaries into the ordered tuple list used
by `ElementalGenerator`. A project-specific momlist helper supplied by the user
takes precedence. Do not hand-write a list merely because it appears equivalent:
the dictionary order is part of the momentum-axis contract. If explicit
`--momentum` values are used, preserve their order and report them exactly.
In the current repository, `mom_dict_to_list` accepts only `N=0,1,3,9`; do not
invent `mom_dict_to_list(2)` for a requested `p^2\le2` shell. Locate the exact
project-specific generator for that shell or ask the user to confirm an
explicit list.

`lattice.dispatch.Dispatch` is a cfg/task queue, not a momentum generator. Given
`Dispatch(cfglist, suffix)`, rank 0 creates or reuses
`cfglist.<suffix-or-random>.tmp`, atomically removes one non-empty cfg line at a
time under a file lock, broadcasts that cfg to the other MPI ranks, and yields
it to all ranks in that MPI worker. The cfglist, temporary queue, code, inputs,
and outputs must be on a filesystem shared by all ranks. `clear()` removes the
queue only after every user has stopped.

For a multi-cfg elemental run, each cfg task normally computes the complete
confirmed momentum list; Dispatch does not split the momenta unless the user
explicitly requests a separate momentum-level decomposition. Choose one cfg
distribution strategy and state it: use `Dispatch` for dynamic MPI workers, or
use explicit Slurm array indices for one cfg per array element. Do not combine
both accidentally. With Dispatch, use a unique suffix, inspect matching
`.tmp` files and active jobs before reuse, and never remove a queue belonging to
an active run. With an array-only run, `SLURM_ARRAY_TASK_ID` selects the cfg and
no Dispatch `.tmp` should be created. Record the chosen strategy, cfg key, and
ordered momentum list in the run metadata.

For production, create a persistent manifest beside the outputs recording a
parameter hash, cfg, job/array ID, Dispatch suffix, output paths, status
(`planned/submitted/running/failed/needs-validation/validated`), attempts, and
validation results. Never let two active tasks write the same cfg output. Before
resuming, check scheduler state, existing outputs, and matching `.tmp` queues.
Do not silently change physics parameters or resources after a failure; propose
the change and obtain confirmation. Call `Dispatch.clear()` only after all
workers stop and the manifest is complete.

## 5. Parameter / branch matrix (verified against code)

### ElementalGenerator (`lattice/generator/elemental.py`)

| Option | Default | Behavior / constraints |
|---|---|---|
| `--calc-mode` | `calc_deriv` | `calc_deriv`: derivative expansion, per-t block `(num_derivative, num_mom, Ne, Ne)` (`elemental.py`). `calc_disp`: GaugeLink displacement encoding, per-t `(num_disp, num_mom, Ne, Ne)` (`elemental.py`). `calc(t)` dispatches (`elemental.py`). |
| `--num-nabla` | 0 | Sets `num_derivative = (3**(n+1)-1)//2` (`elemental.py`): n=0→1, 1→4, 2→13, 3→40. And `num_disp = list(GaugeLink.nmax_generator(n))[-1]` (`elemental.py`): n=0→1, 1→7, 2→37, 3→187. |
| `--momentum` | `0,0,0` | List of `(px,py,pz)`; phase `exp(i p·x)` from `MomentumPhase` (`lattice/insertion/phase.py`); a half-displacement phase `exp(i p·disp/2)` is applied inside calc_disp / v2v. |
| `--dilution-tot`, `--dilution-used` | None | Blending coefficients: tuple `(totNe_list, usedNe_list_or_int)` (`elemental.py`). Asserts `len` match, each `usedNe<=totNe`, and `sum(usedNe_list)==Ne` (`elemental.py`). |
| `--is-blending` | off | Requires dilution, else `ValueError: Dilution tuple is not defined.` (`elemental.py`). Calc then contracts with `stocastic_coeff[:usedNe,:usedNe]` (`elemental.py`). |
| `--used-ne` | all (`eigenvector.Ne`) | `elemental.py`. Subset is exact (leading axes of the Ne x Ne block). |
| `--used-np` | accepted, ignored | Kept for compatibility only in ElementalGenerator (`elemental.py`). |
| `--debug` | off | Verbose per-t/per-disp prints (`elemental.py` etc.). |
| Smearing | `--stout-nstep 0 --stout-rho 0.12` | `load` then optionally `project_SU3` (`elemental.py`, iterative U←(U+(U⁻¹)ᴴ)/2 to 1e-15) and `stout_smear(nstep, rho)` (`elemental.py`): numpy→ndarray impl; cupy→CUDA kernel, else QUDA (needs MPI), else ndarray. `--no-project-su3` skips projection. |
| `re_combine` | n/a | Class method `re_combine_auto` (`elemental.py`, auto dense/sparse) recombines displacement-basis elementals into the insertion basis; module-level `re_combine` at `elemental.py`. |

Guard: calling `calc_deriv` on a `calc_mode="calc_disp"` generator raises `ValueError`
(`elemental.py`).

### CurrentElementalGenerator (`lattice/generator/elemental.py`)

| Option | Default | Behavior / constraints |
|---|---|---|
| `--calc-blocks` | `all` | `v2v` per-t `(num_disp, num_mom, Ne, Ne)` (`elemental.py`); `v2p` per-t `(num_disp, Ne, Np, Nc)` (`elemental.py`); `p2v` per-t `(num_disp, Np, Nc, Ne)` (`elemental.py`); `p2p` list of per-disp dicts — `{"type":"identity"}` for zero displacement or `{"type":"sparse","indices":[K,2],"values":[K,Nc,Nc]}` with K possibly 0 (`elemental.py`); `all` = `calc_all(t)` computes all four reusing gauge-link products (`elemental.py`). |
| `--num-nabla` | 0 | `num_disp = nmax_generator(n)` as above (`elemental.py`). |
| `--momentum` | `0,0,0` | Used by v2v only; v2p/p2v/p2p are momentum-independent. |
| `--used-ne`, `--used-np` | all | Validated by `_bounded_count`: `ValueError` unless `0 <= used <= available` (`elemental.py`, helper at). |
| `--np`, `--point-prefix` | required for current | Point table `(Np, Lt, 3)` zero-based `(x,y,z)` ints via `PointSourceNpy` (`lattice/preset.py`). |
| `--debug` | off | Same verbose mode. |
| Extra API | n/a | `calc_directed_current_raw(boundary="periodic"\|"open")` returns `{"v2v": (8, Lt, num_mom, Ne, Ne), "contract": metadata}` for the conserved-current one-link basis (`elemental.py`); not exposed by gen_elem.py. |

Both generators: `load(key)` reads gauge (`GaugeFieldIldg`, `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]`
big-endian, `lattice/preset.py`; transposed to `(Nd,Lt,Lz,Ly,Lx,Nc,Nc)`, spatial links
`[:Nd-1]` kept) and eigenvectors (`EigenvectorNpy`, `[Lt, Ne, Lz, Ly, Lx, Nc]`,
`lattice/preset.py`).

## 6. Output contract (files produced and how they load back)

| File | On-disk shape / dtype | Loader (`lattice/preset.py`) |
|---|---|---|
| `{key}{out_suffix}` (default `{key}.elemental.npy`) | `(Nblock, num_mom, Lt, usedNe, usedNe)`, complex128; `Nblock = num_derivative` (calc_deriv) or `num_disp` (calc_disp) | `ElementalNpy` (default suffix `.stout.n20.f0.12.nev70.meson.npy`) |
| `{key}.meson` (`--out-format binary`) | same layout, raw `<c16` bytes | `ElementalBinary` (default suffix `.stout.n20.f0.12.nev70.meson`, default shape `[40, 27, 128, 70, 70]`) |
| `{key}_v2v.npy` | `(Lt, num_disp, num_mom, Ne, Ne)` complex128 | `CurrentElementalV2V` — its view re-indexes to the presented `(num_disp, num_mom, Lt, Ne, Ne)` layout; 5-index key `(disp, mom, t, e_snk, e_src)` |
| `{key}_v2p.npy` | `(Lt, num_disp, Ne, Np, Nc)` complex128 | `CurrentElementalV2P` |
| `{key}_p2v.npy` | `(Lt, num_disp, Np, Nc, Ne)` complex128 | `CurrentElementalP2V` |
| `{key}.t{t:03d}.p2p.npz` per t | entries `type_{i}` (str), `indices_{i}` `(K,2)` int32, `values_{i}` `(K,Nc,Nc)` complex128, one entry per disp | `CurrentElementalP2P`; `load(key, t, num_momentum)` expands disp-major: entry index `= disp_idx * num_momentum + momentum_idx`. HDF5 variant `{key}_p2p.h5` (`disp_{i}/t_{t}/...`) is also supported by the loader but gen_elem.py writes npz. |
| `{key}.elemental.meta.json` | provenance sidecar (args, shapes, momentum list) | informational |

A subset `--timeslices` still writes the full `Lt` axis; un-computed slices are zero-filled.

## 7. Runtime (measured; neutral reference hardware)

Reference problem: `test/weak_field.*`, lattice 4^3 x 8, Ne=20, num_nabla=2, 7 momenta,
numpy backend, one CPU core (interactive and batched runs gave the same figures):

| Operation | Measured |
|---|---|
| ElementalGenerator init + load | < 0.1 s |
| calc_deriv, per timeslice | 0.19-0.22 s (full Lt=8 ≈ 1.7 s) |
| calc_disp, per timeslice | 0.22-0.23 s |
| CurrentElementalGenerator load | ~0.01 s |
| current calc_v2v, per t | 0.01-0.03 s |
| current calc_v2p / p2v / p2p, per t | < 0.05 s |
| current calc_all, per t | ≈ sum of the four (no regression) |
| Full smoke run (elemental + current, 8 t) | ~2 s each |
| Scheduler round-trip (submit + both smokes) | ~45 s wall incl. queue wait (site-dependent) |

GPU (1x 32GB 32GB-class GPU, cupy, num_nabla=1, 2 momenta): cupy init incl. `RawModule`
kernel compile 4.1 s (one-off per process), load < 0.1 s, calc_deriv t=0 0.18 s; results
agree with numpy to 2.2e-15; `stout_smear(1, 0.12)` through the CUDA kernel finishes
without error.

Scaling note: cost grows with `num_disp` (calc mode) or the 2**len(derivative) split in
calc_deriv, the number of momenta, and usedNe²; real-ensemble elementals (Lt=72-128,
Ne=70-128, nabla=2-3) are minutes-to-tens-of-minutes per configuration on one CPU core —
budget accordingly and keep Ne subsets for smoke runs.

## 8. Pitfalls

- Set `CUDA_VISIBLE_DEVICES=0` before any cupy run; otherwise cupy initialization fails
 even though devices are idle.
- Construct generators AFTER `set_backend`; the backend is captured lazily at construction.
- `CurrentElementalGenerator.load` frees the gauge cache (`gauge_field.data = None`,
 `elemental.py`); calling `load` a second time with the same key on the same
 loader short-circuits and returns `None` data → `TypeError: 'NoneType' object is not
 subscriptable`. Create a fresh generator (or fresh loader objects) per reload.
- `calc` returns an internal reused buffer (not a copy); copy it if you keep references
 across calls.
- `ElementalNpy`/`ElementalBinary` metadata dtypes (`<c8`/`<c16`) and shapes are NOT
 validated on load — a wrong declared shape silently misindexes the mmap. Keep the
 `.meta.json` sidecar with the true shape.
- Blending requires `sum(usedNe_list) == Ne` exactly and `usedNe <= totNe` per group
 (asserts in `elemental.py`); a single int for `--dilution-used` is replicated
 across groups.
- `CurrentElementalP2P.load` is disp-major (`disp_idx * num_momentum + momentum_idx`);
 the p2p block for zero displacement is `{"type": "identity"}`, and valid-pair sets can
 be empty (`indices` shape `(0,2)`).
- An MPI launcher must be on PATH only for the QUDA smearing fallback (not for the
 cupy-kernel path); on some modulefile-based sites it is unavailable in
 non-interactive shells, so test MPI availability before relying on that fallback.
- Stage everything under your own scratch/work directories; do not write into shared or
 other users' data areas. Keep gauge/eigen inputs and outputs on storage with enough
 quota for full-`Lt` outputs.

## Production prerequisite routing

Before production, inspect the required cfg list, gauge files, metadata, and branch-specific products. If a prerequisite is missing, stop before submission and identify the producer skill: use easy-distillation-eigensystem for Laplace eigenvectors, easy-distillation-sparsened-point for point tables, easy-distillation-elemental for ordinary/current elementals, and easy-distillation-perambulator for VSV/PSV/PSP or density/generalized blocks. Propose the missing generation step and its parameters; do not silently create random or default prerequisite data. After generation, validate and record its path, parameter provenance, and content hash in the production manifest.

The CLI smearing defaults are reviewable proposals only. Resolve other ensemble-dependent or action parameters from metadata; filename suggestions are inferences and require user confirmation before execution.
