# Kunshan EasyDistillation data map for Wilson Current×Current

## Scope

This is a read-only audit of `/public/home/siyangchen/qedinf/EasyDistillation` and the paths referenced by its scripts. It records paths and compatibility evidence for the eight-configuration ensemble
`10000,13000,14000,15000,16000,17000,18000,19000` on
`beta6.20_mu-0.2770_ms-0.2400_L24x72`.

A matching filename such as `current_elemental` does not establish compatibility with the new eight-direction Wilson point-split `ConservedVectorCurrent`. Operator compatibility is stated separately below.

## Code locations that define data addresses

- `/public/home/siyangchen/qedinf/EasyDistillation/EasyDistillation/2.gen_propagator.py:37` defines sparse-point, gauge, LapH eigenvector, legacy VSV and legacy PSV paths.
- `/public/home/siyangchen/qedinf/EasyDistillation/EasyDistillation/3.gen_elemental.py:29` defines gauge/eigenvector inputs and the meson-elemental output.
- `/public/home/siyangchen/qedinf/EasyDistillation/EasyDistillation/3.gen_current_elemental_all.py:34` defines gauge/eigenvector/point inputs and legacy `v2v/v2p/p2v/p2p` current-elemental outputs.
- `/public/home/siyangchen/qedinf/EasyDistillation/EasyDistillation/4.contractionNocurrent.py:53` fixes the eight configurations and defines the legacy rho–rho C2 output.
- `/public/home/siyangchen/qedinf/EasyDistillation/EasyDistillation/4.contractionWithcurrent.py:110` defines obsolete perambulator roots; lines 146–149 build only a two-vertex meson–current diagram and then exit before contraction.
- `/public/home/siyangchen/qedinf/experiments/nonlocal-current/scripts/4.calc_localized_two_point.py` is the surviving localized two-point implementation; SHA-256 `5d17bb3814bea31026f8a415aeba016d5e3bcfba307c76fa8e7a27f0125c1811`.
- The same bytes exist at `/public/home/siyangchen/qedinf/experiments/localized-blending/scripts/4.calc_localized_two_point.py`.

## Reusable real-data inputs

All counts below were checked with exact expected filenames for the eight target configurations.

| Role | Absolute path | Expected/present | Header or manifest declaration | Compatibility |
|---|---|---:|---|---|
| Gauge | `/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_*.lime` | target cfg files exist | loader declares `[t,z,y,x,mu,color,color]` | reusable gauge input |
| Sparse points | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/01.sparsened_field/{cfg}.npy` | 8/8 target cfgs | `(216,72,3)`, `int32` | reusable; localized production uses the first declared 64 points |
| LapH eigenvectors | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/02.laplace_eigs.nev128/{cfg}.npy` | 8/8 target cfgs | `(72,128,24,24,24,3)`, `complex128`; `[t,Ne,z,y,x,color]` | reusable |
| Meson elementals | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/03.elemental.ndisp1.np0.nev128/{cfg}.npy` | 8/8 target cfgs | `(7,23,72,128,128)`, `complex128`; saved as `[disp,momentum,t,Ne,Ne]` | candidate rho input; operator equivalence still requires the measurement definition |
| Overlap matrices | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/03.overlap_matrix/{cfg}.overlap_matrix.npy` | 8/8 | `(72,128,216,3)`, `complex128` | reusable localized-blending input |
| Localized VSV | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.localized.nev128_to_nev128.fulltime.src18.np64` | 576/576 slabs | each `(18,4,4,128,128)`, `<c16`; 18 sources `0,4,...,68`; four temporal ranks; `absolute-global-source-and-sink` | reusable |
| Localized PSV | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.localized.nev128_to_np64.fulltime.src72` | 2304/2304 slabs | each `(18,4,4,64,3,128)`, `<c16`; all 72 sources; four ranks; `absolute-global-source-and-sink` | reusable |
| Localized PSP | `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.localized.np64_to_np64.fulltime.src18` | 576/576 slabs | each `(18,4,4,64,3,64,3)`, `<c16`; explicit axis order begins `t_sink_local,spin_sink,spin_source,...`; 18 sources | reusable |

Manifest hashes:

- VSV manifest: `8d8ca7498a07c85ef704ebc22b05385574d2529cb5aa5b6a41654da5798b2c11`.
- PSV manifest: `a8f7ba4f2bde2ebbdac644f5082090ebd093bc3b392fe9cf246327f40d62b2cb`.
- PSP manifest: `6f94f539910af58feecabb7c8f8209e5807818fbd0f2fcc6dad6ca90746823b1`.

The manifests explicitly declare `layout=source-time-rank-slab`, periodic-compatible inputs with `t_boundary=-1`, global lattice `(24,24,24,72)`, temporal grid size four, mass `-0.277`, clover `1.160920226`, stout steps/rho `20/0.12`, and the absolute-global source/sink convention. No VSP files are expected by this production layout; the localized code obtains the reverse channel from the supported propagator algebra rather than from a separate VSP dataset.

## Existing correlators

### Candidate C2 families

There are two distinct rho-like C2 families; they must not be treated as aliases.

1. Readiness-audited candidate:
   `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/05.correlator.nocurrent.nodisp/{cfg}.npy`
   (also reached through the `experiments/localized-blending/data/production` symlink). It has 8/8 `(72,72)` complex files. Example hashes are cfg10000 `b92fbe39bb7a2aa9ef88422acc09ab2519339976bf5aa025713fff48d6300e09` and cfg19000 `001c6ceecd3c81bbc35d3e664667b73f9e7f035f5efc095dad7d1a2894e17d36`.
2. Legacy script output:
   `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/05.correlator.nocurrent.nonlocal/{cfg}.npy`.
   It also has 8/8 `(72,72)` complex files, but different bytes (cfg10000 `f2d346b5d98b2333943b8a0ad535213ea00de4c0305d5162d6ce9c958191c3c0`; cfg19000 `9e72234768f7b18c70572a1c33af39d3cdb72fcf0377190ae0f17048bf7257eb`).

Neither family is promoted to a denominator without an operator/formula match.

### Meson–legacy-current two-point

`/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/05.correlator.vector_meson_to_vector_current/usedNe128.usedNp216/{cfg}.npy`
contains 8/8 `(72,72)` complex files. Its `RUN_NOTES.txt` records Slurm job `118684570`, 18 source times, full `usedNe=128/usedNp=216`, and a completed two-vertex meson–current run.

This is not a hadron–current–hadron C3 and is not a Wilson temporal-current Current×Current correlator.

## Legacy `current_elemental_all` compatibility

The directory
`/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/03.current_elemental_all.ndisp1.nmom0.nev128.np216`
contains all four files (`v2v`, `v2p`, `p2v`, `p2p`) for each of the eight target configurations (32/32 files). Example `v2v` shape is `(72,7,23,128,128)` complex128; the generation script declares the other layouts.

Despite the name, these are generic spatial `GaugeLink` elementals generated from six spatial directions. The legacy nonlocal endpoint is `GammaName.A0 x T_1u- -> T_1`. It is not the new eight-direction Wilson point-split basis with forward/backward coefficients `-1/2(r-gamma_mu)` and `+1/2(r+gamma_mu)`, and it does not include temporal links. Therefore these files may be used only after an explicit algebraic compatibility proof; they are not silently reused as directed Wilson-current artifacts.

## Superseded and deleted localized correlators

Eight Slurm jobs (`119414114,119414128,119414134,119414138,119414149,119414160,119414169,119414174`) historically produced three two-point topologies for every target configuration under
`05.correlator.localized.production.ne128.np64.src18`:

- local-current to local-current;
- local-current to nonlocal-current;
- local-current to meson.

Those result manifests remain under `/public/home/siyangchen/BASE/localized-production-np64/correlator-cfg*/attempt-*/result/result.json`, but the correlator files are no longer present. The authoritative final verification says they were generated before the 2026-08-22 spin-dagger/high-mode implementation fix and must be regenerated for physics outputs:

- `/public/home/siyangchen/BASE/localized-production-np64/final-verification-summary.json`
- SHA-256 `9393a9872d50d74fbd413ff7ef9795fecd741dcb1f383e6ee83cf5967878448b`

The stale cleanup inventory is
`/public/home/siyangchen/BASE/localized-production-np64/cleanup-stale-20260822.json`, SHA-256 `07ef5c5d45ace68ba44ccb188c4347127ebaa6959ffad9efa5de6c0b0c848a4c`.

Historical `status=passed` is execution lineage, not evidence that the now-missing, pre-fix arrays are reusable.

## Three-point search result

No genuine three-vertex hadron–current–hadron C3 producer or nonempty C3 result family was found in the current checkout, all Git refs searched, the isolated experiment snapshots, or shallow data/result trees. The apparent current producer in `4.contractionWithcurrent.py` has only two vertices and exits before contraction. This is a bounded negative search result, not a claim that no external/private C3 can exist.

## Actionable conclusion

1. The raw real-data inputs needed to construct a new eight-configuration localized contraction are present and complete: current-independent VSV/PSV/PSP slabs, overlap matrices, gauge/eigenvector/point inputs, and candidate C2s.
2. Existing correlators and `current_elemental_all` products use legacy two-point/spatial-current definitions and cannot establish the new Wilson temporal `J4` Current×Current observable.
3. The old eight-configuration localized current-current outputs are both implementation-superseded and absent; regenerate into a new result directory if that two-point observable is selected.
4. A Wilson Current×Current producer should reuse the audited VSV/PSV/PSP inputs but bind the new directed-current operator/term convention and write fresh hashes, axis semantics, code version and Slurm provenance.
5. A charge-normalization H–J–H C3 remains a separate observable and still needs the approved three-vertex formula; it must not be inferred from any two-point current-current or meson-current array.
