# Kunshan directed-current / existing-VSV smoke evidence

## Scope

This record documents an artifact-level smoke for configuration `10000` on Kunshan. It verifies real input files, explicit time-axis semantics, directed-current generation, endpoint-aware term contraction, hashes, and atomic output publication. It is **not** a Ward--Takahashi, charge-normalization, or statistical physics validation.

## Remote workspace

- Audit root: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612`
- Isolated source snapshot: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/work/source-final`
- No existing EasyDistillation checkout or production data file was overwritten.
- After the smoke, the isolated snapshot was updated with final audit hardening (same-byte JSON hash/parse, trusted runtime binding, in-memory consumed arrays, and full-family post-consumption rehash). Final snapshot SHA-256 values: `lattice/current_elemental.py` = `fe6e006e17e65b7c434f0b8f3fbeeb8b91a02ce98b317b4e96a4a5efd1a60367`; `contract_existing_vsv.py` = `778da123b8f79c2819939d8546622b6ce4bff914a7625440577b1c53a4272b11`. The canonical v3 result manifest retains the producer script SHA from the code actually executed for that result; v1/v2 are superseded history.

## Existing VSV audit

Existing VSV data were found and reused; no propagator was regenerated.

- Data directory: `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/04.perambulator.smoke.nev1_to_nev1.fulltime`
- Configuration: `10000`
- Dataset manifest layout: `source-time-rank-slab`
- Files: `10000.t000.rank0000.npy` through `10000.t071.rank0000.npy` (`72/72` present)
- Per-file shape/dtype: `(72, 4, 4, 1, 1)`, `complex128`
- Explicit second-time semantics: `source-relative`
- Generated hash manifest: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/10000.vsv-timeslices.manifest.json`
- Hash-manifest SHA-256: `092c04424e6781dd5e06bbbca439faaef181ace7d0043e2c75902abe46b7226f`
- Hash-manifest identity: `d12037454d8f957b62786ad0588a32a37464dbf6fe0fef2485ca0e104668fbc4`
- First-file SHA-256 (`t000`): `6d02bdcde8cced931885b71b541c7ad2704a151b2afc1eb2ca486c0d458f73f7`
- Last-file SHA-256 (`t071`): `28bec3a2b8fd38ec24056854da219ffec0f068196820955276aa18f56268b27a`

## Directed-current artifact generation

The existing VSV used `Ne=1`, but a matching directed-current artifact did not exist. A bounded DCU test generated only that missing Current artifact; it did not generate a propagator.

### Attempt 1

- Slurm Job ID: `120571753`
- Result: `FAILED`, exit `1:0`, elapsed `00:00:35`
- Classification: infrastructure-incomplete
- Cause: compute job did not load the validated module environment, so CuPy could not find `libomp.so`.
- Evidence: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/results/current-ne1-cfg10000-attempt-01/CONTINUE.json`
- No artifact result was accepted.

### Attempt 2

- Slurm Job ID: `120571967`
- Partition/account: `kshdnormal` / `ybyang`
- Resources: `1 node`, `1 DCU`, `8 CPU`, `04:00:00` limit
- Result: `COMPLETED`, exit `0:0`, elapsed `00:02:30`
- Result directory: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/results/current-ne1-cfg10000-attempt-02`
- Current artifact identity: `bb322b7d704cc3e7b551c12e22ca625de8300e529831fe94980b586f28ae280f`
- Current data SHA-256: `dcf3d98a5083f617fc6dde0b2014e25157b6025af754b4079617375c7a482adb`
- Current manifest SHA-256: `d7f180e41f60b8239b38598656f15a5e6c85cb2b0067f541562e581916a7df2e`
- Raw cache identity: `56e62060e55663da57f3aeb2221021e4ed8e866629e3d5c6ad144465f163dc84`
- Result/DONE hashes were independently re-read and matched.

### Real input lineage

- Gauge: `/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_10000.lime`
- Gauge SHA-256: `f3525aa58b745a2ec555be82659b391d28ad97adc0abcbbc8c9f38cf715ce8fc`
- Eigenvector: `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/02.laplace_eigs.nev128/10000.npy`
- Eigenvector SHA-256: `6d4c2c8f5ec7fa3ea5a94a3aad24a333bf296dad7e5fe5ec812687d60c666903`
- Point file: `/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72/01.sparsened_field/10000.npy`
- Point SHA-256: `789b0a405b719af674a5212dbcefafc3c8db48d6759001049097d1f9ff16f8d1`

## Existing-VSV smoke result

- Current component: temporal (`direction=3`), Wilson `r=1`
- Times: source `0`, sink `8`, current anchor `4`
- Boundary: periodic
- Current/VSV Ne: `1/1`
- Momentum: `(0,0,0)`
- Dry-run status: valid; sources verified; `4` actual VSV block accesses; no output directory created
- Execution record: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/10000.execution-record-login.json`
- Execution-record SHA-256: `de1856caa157e92fab5ba27f239318d3dd48e6a7e395beb683353bcc4879411e`
- Execution-record identity: `0513a7c2af51354b14078e2addd3c9a752a835ee936ecff5bd7a83d5ab7d7b8d`
- Smoke execution mode/Job ID: `login-node-readonly-artifact-smoke` / `none-login-readonly`
- Current artifact generation lineage remains Slurm Job `120571967`; it is not misrepresented as the smoke execution itself.
- Result directory: `/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/results/vsv-current-smoke-cfg10000-v3`
- Output shape/dtype: `(4,4,1,1)`, `complex128`
- Maximum absolute value: `0.009072637668193217`
- Result SHA-256: `f31f3975dbb4493d6eb0bb1c7bbcdf9ceaa7e2f30ab529d7048a3b6404880ac8`
- Result manifest identity: `39cc1604fb94856cc5acf744c0ae400fb79f42c38f3ca38c578cf05f7ee292ca`
- Result manifest SHA-256: `af80199f26dd3e68aa76a7cff304008c312ae66ae815183c62afbbbc5959c178`
- Producer script SHA-256: `778da123b8f79c2819939d8546622b6ce4bff914a7625440577b1c53a4272b11`
- Result manifest and `DONE` hashes were independently re-read and matched.
- The final v3 run enforces trusted runtime/record equality, hashes and parses each audit JSON from the same byte snapshot, copies consumed arrays in memory, and rehashes all 72 VSV source-time files before publication.
- Earlier `/results/vsv-current-smoke-cfg10000` and `-v2` results have the same contraction SHA-256 but predate one or more final audit-binding hardenings; they are retained as superseded history.

### Accessed VSV blocks

The smoke used exactly the endpoint-derived blocks below:

1. `S(field=5 -> sink=8)`: file `t005`, relative index `3`
2. `S(source=0 -> bar=4)`: file `t000`, relative index `4`
3. `S(field=4 -> sink=8)`: file `t004`, relative index `4`
4. `S(source=0 -> bar=5)`: file `t000`, relative index `5`

The backward Current term selected raw direction `7` at its bar endpoint time `5`; no second dagger was applied by the consumer.

## Remaining validation boundary

This evidence establishes a reproducible real-artifact V2V workflow for one configuration, one mode, and one time tuple. It does not establish Ward--Takahashi equality, contact terms, charge normalization, ensemble statistics, systematic errors, V2P/P2V/P2P point-split support, or production-scale performance.
