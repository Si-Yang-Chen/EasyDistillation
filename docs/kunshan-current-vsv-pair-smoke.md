# Kunshan real dual-current (J4×J4) pair smoke evidence

## Scope

This records the first real-gauge, dual point-split conserved-current contraction on Kunshan. It reuses the audited cfg10000 `Ne=1` directed-current artifact and the existing `Ne=1` VSV source-timeslice family; no propagator, eigenvector, or gauge input was regenerated. The output is an **ordered, connected, unflavored, unsigned V2V trace**: it is an artifact smoke, not a Ward--Takahashi, charge-normalization, or ensemble physics result.

## Deployment identity

- Bundle: `/public/home/siyangchen/BASE/wilson-j4-426d076.bundle`, SHA-256 `e9bd7096073c43d550c0da5a71a34235bb86c51f09d9655989c327814232960f`.
- Isolated clone: `/public/home/siyangchen/BASE/lattice-flow-wilson-j4-pair-smoke/20260902-220837/source`, branch `feature/wilson-current-j4`, HEAD `426d0764e57a83265a47d8844aa431929662fbfa`, clean worktree. The shared checkout `~/qedinf/EasyDistillation` was not modified; its state was re-verified read-only before deployment (HEAD `94f8fcdd…`, dirty `lattice/insertion/__init__.py`, pending foreign array job `120729358_[0-1]` untouched).
- Snapshot source manifest: `audit/handover-manifest.snapshot.json` inside the clone; SHA-256 `1e62cae528a17659b28c1ee7551efdf1ee03eb973912d554bc9b7359f8654b38`; identity `7ac911a7e04e3742728fe8a8b81dbb268bafc79087eebd7920cebfacc3ade512`; 65 files verified byte-exact on the clone, including all eight required pair dependencies.

## Execution record

- Path: `/public/home/siyangchen/BASE/lattice-flow-wilson-j4-pair-smoke/20260902-220837/audit/10000.execution-record-pair-login.json`
- SHA-256: `7d422bafe33c1f2e6a3dd5b64533586fc61b7325fabd4eb951b004d607be36e1`
- Identity: `d1f723cafdef190f87fa927188e776e4738ba3d1b99fc80dd027a331d7535267`
- Mode: login-node-readonly CPU artifact smoke, Job ID `none-login-readonly`; cluster/git/resources in the record matched the trusted runtime exactly.
- Bound inputs: current artifact manifest `d7f180e4…`, VSV timeslice manifest `092c0442…`, snapshot source manifest `1e62cae5…`.

## Inputs

- Directed-current artifact: `…/20260901-062612/results/current-ne1-cfg10000-attempt-02/current-artifact`; identity `bb322b7d…`, data SHA-256 `dcf3d98a…`, raw cache identity `56e62060…`.
- VSV family: `04.perambulator.smoke.nev1_to_nev1.fulltime/10000.t{source_time:03d}.rank0000.npy`, shape `(72,4,4,1,1)`, `<c16`, declared `source-relative`; hash manifest `092c0442…`, identity `d1203745…`.

## Pair contraction

- Direction: temporal `J4` (`current_direction=3`), Wilson `r=1`, both currents from the same artifact, anchors `first=second=4`, momentum `(0,0,0)`, `Ne=1`.
- Term pairs: 4 (`forward,forward`, `forward,backward`, `backward,forward`, `backward,backward`) with coefficients `(-1/2)(-1/2)`, `(-1/2)(+1/2)`, `(+1/2)(-1/2)`, `(+1/2)(+1/2)`; endpoints `bar=4,field=5` (direction 6) and `bar=5,field=4` (direction 7); backward raw channel carries the dagger, no second dagger applied.
- VSV accesses: 8, exactly `S(field_A→bar_B)` and `S(field_B→bar_A)` per pair; touched only `t004` (`b4740090…`) and `t005` (`767fae1f…`) rank files at relative indices 0/1/71; all accessed blocks finite.
- Kernel: `sum_terms bfji,ackl,afki,bcjl->`; no implicit Wick sign, flavor factor, normalization, conjugation, real-part selection, or source averaging.

## Result

- Directory: `/public/home/siyangchen/BASE/lattice-flow-wilson-j4-pair-smoke/20260902-220837/results/pair-smoke-cfg10000-v1`
- Result: rank-0 `<c16`, value `-1.2601715236706704e-05 - 0.00029053104880180804j`, finite.
- Result SHA-256: `a1cb967c0d44fc8e917a7fe4ea46ee25ea3a1548d3d39e636ce3607b365cf21d` (content-addressed filename).
- Result manifest: identity `3be081ae0127aeb567c8edff88f029e9608e6a298e8870d374ae7e43558a4fbc`, SHA-256 `36077ec515ccabd6721f9f36b465322b56710bb8bdea23a6a79766cea99983a3`.
- `DONE` SHA-256: `8904f2cb6d8b413675f9d62aa9fd743afdedf36d4f0d907f6794c4bef27f83ea`; all three artifacts independently re-hashed after publication.
- Dry-run rehearsal: `results/pair-smoke-cfg10000-dryrun` validated the same bindings without creating output.

## Boundary and notes

- The pair pytest subset was not re-executed on the login node (subprocess overhead exceeded session limits); the deployed clone is byte-identical to the locally tested tree via the snapshot manifest, and the authoritative CLI ran once end-to-end here.
- This smoke demonstrates dual-endpoint machinery, term pairing, VSV access discipline, and publication integrity for one configuration, one `Ne`, one anchor pair. Ensemble `(Ncfg,Lt)` production, flavor/charge definitions, and any Ward--Takahashi statement remain separate gated work.
