# Kunshan conserved-current measurement readiness

## Purpose

This evidence records the next stage after the successful cfg10000 Current/VSV artifact smoke. It audits whether existing Kunshan ensemble data are sufficient for a scientifically defined connected charge-normalization or Ward--Takahashi measurement. It does not promote a two-point correlator into a three-point numerator and does not invent a ratio/contact formula.

## Audited ensemble

- Ensemble label: `beta6.20_mu-0.2770_ms-0.2400_L24x72`
- Configurations: `10000, 13000, 14000, 15000, 16000, 17000, 18000, 19000`
- Temporal extent/boundary: `72`, periodic
- Inventory source v3: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/measurement-inventory-source-v3.json`
- Inventory-source-v3 SHA-256: `b45d517ce16a0f956a87640648310ae3dba606416038abb46c6d3be915beca20`
- Rebuilt inventory v3: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/measurement-inventory-v3.json`
- Inventory-v3 SHA-256: `abdd6461f46ba0ba5ef1a848ac3b8a28520aa4908eb791466948a64cb7968b92`
- Readiness report v4: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/measurement-readiness-v4.json`
- Readiness-v4 SHA-256: `b5dc0a7dabed27b32ab3ec3ca7c2a49c5247ec7fd6e68214d871c73a8ca17e95`
- Readiness-tool SHA-256: `e9eb4ee2f5d238c989326d2c105081d38e3ec694c15ecaeb60c42ceff179c566`
- Readiness v4 records `files_verified=true`, enforces role-specific time axes/shape and topology, and normalizes hash comparison. Earlier readiness v1-v3 records are superseded lineage history.
- Draft measurement contract: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/measurement-contract.draft.json`
- Draft-contract SHA-256: `b4c709afa95b825808a5191deef48b85a7356fb86c838d331a6bf26cfda4575a`

The inventory was reproduced from a declarative source with `build_measurement_inventory.py --verify-finite`; every file header and SHA-256 was checked. Inventory v3 binds the candidate C2 to its own generation script (`4.contractionNocurrent.py`, SHA-256 `ce360d0b8a7ddf7867332e6e537c37a80c90e8d5bbf949c302647a9df065c532`) and the meson-current data to its RUN_NOTES (`6a759b86922400269798134d13f98577bf7861cbdb1a0d54c17439f13afff506`). Earlier inventory v1/v2 records are superseded lineage history.

A subsequent read-only audit of the paths encoded in `~/qedinf/EasyDistillation` is recorded in `docs/kunshan-easydistillation-data-map.md`. It found complete eight-configuration localized VSV/PSV/PSP slab families and overlap matrices that can be reused as propagator inputs. It also established that `05.correlator.nocurrent.nodisp` and `05.correlator.nocurrent.nonlocal` are distinct C2 families, not aliases. The legacy `03.current_elemental_all` and meson-current products use a six-spatial-direction `GaugeLink` / `GammaName.A0` definition, not the new eight-direction Wilson point-split temporal `J4`, and therefore are not silently promoted to directed-current artifacts.

## Existing data that can be reused

### Candidate meson two-point

- Role: `candidate-two-point` (not yet an approved denominator)
- Directory: `/public/home/siyangchen/qedinf/experiments/localized-blending/data/production/05.correlator.nocurrent.nodisp`
- Eight files, each shape `(72,72)`, `complex128`, finite
- First cfg/hash: `10000` / `b92fbe39bb7a2aa9ef88422acc09ab2519339976bf5aa025713fff48d6300e09`
- Last cfg/hash: `19000` / `001c6ceecd3c81bbc35d3e664667b73f9e7f035f5efc095dad7d1a2894e17d36`

### Meson–current two-point

- Role: `meson-current-two-point` (not a hadron–current–hadron C3)
- Operator note: source `rho`; sink `GammaName.A0 x T_1u- -> T_1` nonlocal current
- Directory: `/public/home/siyangchen/qedinf/experiments/localized-blending/data/production/05.correlator.vector_meson_to_vector_current/usedNe128.usedNp216`
- Eight files, each shape `(72,72)`, `complex128`, finite
- First cfg/hash: `10000` / `23fb8b3501199580b65f720f3ce10a1548efe2c315f9491ac95317acefa97aeb`
- Last cfg/hash: `19000` / `712c258409cc48cff7b8adb7c5b06b838da62b88d8cf52f3e6fdb38cd07d4a92`
- Evidence notes SHA-256: `6a759b86922400269798134d13f98577bf7861cbdb1a0d54c17439f13afff506`

These data are reusable inputs/evidence, but neither dataset is silently promoted to the required `two-point-denominator` or `three-point-numerator` role.

### Complete localized propagator inputs

For all eight target configurations, an exact filename audit found:

- VSV: 576/576 rank slabs at `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.localized.nev128_to_nev128.fulltime.src18.np64`;
- PSV: 2304/2304 rank slabs at `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.localized.nev128_to_np64.fulltime.src72`;
- PSP: 576/576 rank slabs at `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.localized.np64_to_np64.fulltime.src18`;
- overlap matrices: 8/8 `(72,128,216,3)` complex arrays under `03.overlap_matrix`.

Their manifests explicitly declare `source-time-rank-slab` and `absolute-global-source-and-sink`. These products can support a new contraction without regenerating propagators. Historical eight-configuration localized current-current arrays cannot be reused: they were generated before the 2026-08-22 spin-dagger/high-mode fix, marked for regeneration by `final-verification-summary.json` (SHA-256 `9393a9872d50d74fbd413ff7ef9795fecd741dcb1f383e6ee83cf5967878448b`), and subsequently removed by stale cleanup.

## Machine-readable blockers

The readiness report has `ready=false` with these blockers:

1. current flavor/electric-charge weights are not defined;
2. measurement contract is not approved;
3. approval authority/time are missing;
4. C2 formula is missing;
5. C3 formula is missing;
6. ratio formula is missing;
7. current-time list is missing;
8. plateau-time list is missing;
9. source projector is missing;
10. sink projector is missing;
11. no approved three-point numerator dataset exists;
12. no approved two-point denominator dataset exists.

(The report lists approval authority/time as separate entries, yielding 13 machine-readable strings.)

## Implemented readiness and formula-neutral primitives

- `audit_measurement_readiness.py` validates strict measurement-contract and inventory schemas and reports exact missing products/conventions.
- `build_measurement_inventory.py` reproducibly hashes and validates declared NPY dataset families.
- `measurement-contract.template.json` and `measurement-inventory-source.template.json` provide deliberate `REQUIRES_*` fields rather than defaults.
- `lattice.correlator.conserved_charge.project_explicit_v2v_scalar` applies exactly a caller-supplied dual tensor with no hidden conjugation or normalization.
- `lattice.correlator.conserved_charge.build_declared_ratio` performs only the explicitly declared numerator/denominator division, with definition identity and zero-denominator checks; it performs no fit/contact/plateau operation.

## Next executable action

The EasyDistillation path audit separates two possible next observables:

- **Wilson Current×Current two-point:** raw localized VSV/PSV/PSP and overlap inputs are complete, but the historical output used a legacy spatial-current implementation and was removed after a correctness fix. A new producer can reuse the propagators, bind the directed Wilson-current terms, and write fresh ensemble outputs without waiting for an H–J–H formula.
- **Connected rho charge normalization:** still requires the approved H–J–H measurement contract below; no two-point current-current or meson-current array can replace its C3.

For charge normalization, obtain an approved measurement contract specifying the rho source/sink projector, connected topology, flavor/electric-charge weights, exact C2/C3 index formulas, ratio, source/sink/current times, contact times and plateau. Once approved:

1. promote the existing candidate C2 only if it matches the approved operator/formula;
2. generate the missing hadron–current–hadron three-point numerator on Kunshan (V2V-only if the approved topology permits it; otherwise add required point channels);
3. build `(Ncfg,Lt)` charge/WT arrays with complete hashes and provenance;
4. pass them through the existing audited producer/analyzer gates.

Until that contract exists, numerical C3/ratio production would invent physics conventions and is intentionally blocked.
