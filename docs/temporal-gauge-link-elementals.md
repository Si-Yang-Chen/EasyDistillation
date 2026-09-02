# Temporal Gauge-Link Elementals

## First-party evidence

The synchronized repository HEAD is `94f8fcdd67defdd14ebc4ff2a1a64b26b36fb28f`. `GaugeLink` is introduced by commit `89231744b1ea62a6d1e602e9fb60d9587a5af113` (`lattice/insertion/gauge_link.py`, historical blob in that commit); its current six-direction table, base-five continuation index, three-vector displacement, cubic transforms, and modulo-six conjugation are in `lattice/insertion/gauge_link.py:132`. The pre-repair manifest hashes include `current-api-v1.1.0.tsv` SHA-256 `c3019a8b7bc87ac126c72d4cfb4c27c408b2d3c97cb8c2883fb81945e7aabe7d` and handoff checksum `d535fd2a0f2b3a6e85680db8aa55b641d426b74c560b87cee719c52e107f96b3`.

Git history searches (`git log -S` for `eight direction`, `8 directions`, `GaugeLink`, and `CurrentElementalGenerator`), `git grep` across all refs, and synchronized source/docs searches found no first-party historical persisted eight-direction or four-dimensional directed-link convention. This extension therefore defines a new scoped v1 basis rather than decoding or padding old data.

The loader path in `lattice/generator/elemental.py:1081` converts external gauge data `(t,z,y,x,mu,row,col)` to internal `(mu,t,z,y,x,row,col)`, retaining all four links in `_current_U`; legacy `_U` remains `_current_U[:3]`. The legacy V2V/V2P/P2V/P2P generator path remains `calc_all()` and the corresponding loader formats are documented in `lattice/preset.py:739`. `Current.compute_elemental()` consumes only `calc_all()` mappings (`lattice/insertion/current.py:628`), while the new raw resolver consumes a standalone V2V tensor. Quark-diagram `get_v2p`, `get_p2v`, and `get_p2p` paths consume their dedicated precomputed loaders (`lattice/quark_diagram.py:1815`).

## Scoped mapping and anchors

`DirectedCurrentBasis` in `lattice/insertion/gauge_link.py:132` is deliberately separate from `GaugeLink`; global indices remain untouched because they index arbitrary spatial paths, are transformed by cubic-group maps, and label legacy persisted displacement data.

The V2V-only raw contract is `lattice.current.raw-directed-one-link-basis/v1`. Its fixed directions are `0..7 = +x,+y,+z,-x,-y,-z,+t,-t`, with four-vectors `(±x,±y,±z,±t)`. Its tensor is `(direction,time,momentum,sink_ne,source_ne)`. It does not claim V2P/P2V/P2P coverage.

Gauge input axes are `(mu,t,z,y,x,row,col)`. At raw anchor `(t,x)`, `+t` is `U_3(t,x)` to source time `t+1`; `-t` is `U_3(t-1,x)^dagger` to source time `t-1`. Periodic boundaries wrap, and open boundaries leave unavailable crossing entries zero.

Current terms use a term anchor, which differs from a backward raw anchor. Forward at `x` selects `+mu` at raw bar endpoint `x`. Backward at `x` has `bar_offset=+mu`, so it selects `-mu` at raw bar endpoint `x+mu`, yielding `U_mu(x)^dagger`; selecting its term `link_origin=x` would be wrong. The same spatial relabelling selects the integrated negative channel at its bar endpoint, with the directed V2V midpoint phase already attached to that channel.

## Serialization and cache rules

Public contract metadata is JSON-native: arrays are represented by lists, and shapes/axes/directions/Ne/schema/dtype/boundary are exact validated fields. `validate_current_raw_contract()` rejects missing, unknown, stale, or tampered metadata. `current_raw_cache_key()` recomputes the SHA-256 canonical semantic fingerprint (excluding the identity field) and rejects a copied or stale identity. The same user key cannot reuse a raw entry after changing boundary, version, directions, shape, or Ne.

Legacy `lattice.current.raw-spatial-displacement-basis/v1` data is accepted only through `validate_legacy_spatial_current_raw(raw, contract)`. It validates the full stable `Current.compute_elemental()` v1.1/v1.2 contract: four named channels, spatial representation, false temporal support, external term application, matching schemas, axes/shapes against actual arrays, finite complex V2V/V2P/P2V arrays, symmetric raw Ne with requested-Ne bounds, and Np bounds. P2P validation is intentionally structural only: identity entries or sparse integer `(N,2)` indices and finite complex `(N,3,3)` values. It cannot prove a sparse point set is physically complete. Schema-only and any temporal request are rejected.

The deterministic transport tests use distinct non-Hermitian complex 3x3 links and nontrivial eigenvectors. At `rtol=atol=1e-12` they directly evaluate `E_+(t)=Σ V(t)^† U_3(t) V(t+1)` and `E_-(t)=Σ V(t)^† U_3(t-1)^† V(t-1)` for an interior time and both periodic wraps; `U`, `U.T`, and `U.conj()` are explicitly distinguishable from the required dagger reference.

## Consumer bridge and artifact boundary

`lattice.current_elemental` supplies the temporal/cross-time V2V production seam. `save_directed_current_v2v()` writes a content-addressed NPY and publishes `manifest.json` last. The manifest binds configuration, exact raw contract, momentum list, NPY SHA-256, gauge/eigenvector source paths and SHA-256 values, and the consumer equation. `load_directed_current_v2v()` verifies the NPY and, by default, re-hashes both recorded source files; `verify_sources=False` is an explicit offline-transfer mode and is not source-audit evidence.

`contract_directed_current_v2v()` consumes only preloaded VSV objects exposing `get(t_source, t_sink)`. For every Current term it resolves the distinct bar/field endpoints, obtains `S(field->sink)` and `S(source->bar)`, contracts `afAi,bfji,bcjC->acAC`, and only then applies the term coefficient/normalization and sums. The backward raw channel already carries the dagger and bar-endpoint anchor, so no second dagger is applied. The function never calls `load()` or any propagator generator.

`legacy_current_vertex_adapter` remains available only for equal-time assembled vertices and rejects point-split endpoint provenance. A one-time `get(t)` protocol cannot represent distinct bar/field endpoints and is not the temporal production bridge.

This slice deliberately does not add V2P/P2V/P2P point-split support. Those channels still require persisted point elementals and compatible VSP/PSV/PSP propagators.

## Validation boundary

Tests use deterministic synthetic NumPy links/eigenvectors and synthetic already-loaded VSV accessors. They cover non-Hermitian dagger transport, periodic/open endpoints, independent Current Ne, exact contraction axes, coefficients/signs, atomic publication, content/source hashes, and tamper rejection. They do not validate real gauge configurations, real existing VSV files, physical normalization, GPU/DCU execution, MPI, Slurm, or full observable production.
