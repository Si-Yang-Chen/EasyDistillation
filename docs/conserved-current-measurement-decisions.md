# Conserved-current measurement contract decisions

The Kunshan readiness v4 audit is complete and reproducible. Existing data are not the remaining ambiguity: the unresolved items below define the physics measurement. This document is a decision checklist, not an approval.

## Recovered candidate defaults

These candidates come from existing Kunshan scripts/data and may be accepted or replaced:

1. Ensemble: `beta6.20_mu-0.2770_ms-0.2400_L24x72`.
2. Configurations: `10000, 13000, 14000, 15000, 16000, 17000, 18000, 19000`.
3. Source/sink interpolator candidate: zero-momentum `rho`, `GammaName.RHO x A_1g+ -> T_1`, V2V smeared.
4. Current candidate: `ConservedVectorCurrent`, temporal component `J_4` (`component=3`), Wilson `r=1`, connected V2V topology.
5. Existing source times: `0,4,...,68`.
6. Existing candidate C2: `05.correlator.nocurrent.nodisp`; it is not promoted until the exact C2 definition is approved.
7. Existing `vector_meson_to_vector_current` data are a two-vertex meson-current correlator and cannot serve as a hadron-current-hadron C3.

## Required decisions

### 1. Physics target and topology

- Confirm the target is connected rho charge normalization, or name a different hadron/observable.
- Confirm connected V2V-only topology is sufficient.
- State whether disconnected contributions are excluded by definition or merely deferred.

### 2. Flavor/electric-charge weights

Provide the exact flavor content and numerical weights, including whether the target is isovector, individual flavor, or electromagnetic. State all relative signs.

### 3. Spin/irrep projector

Provide the exact source and sink row/polarization treatment and dual contraction tensor. State whether rows are selected, summed, or averaged and whether any complex conjugation is part of the **definition**.

### 4. Exact C2 and C3 formulas

Provide complete index formulas and overall signs/factors for:

- `C2(t_src,t_snk)`;
- `C3(t_src,t_current,t_snk; J_4)`;
- source/sink/current momentum projection;
- forward/backward quark-line ordering;
- whether the existing candidate C2 is definition-equivalent.

Do not answer only `C3/C2`: the numerator and denominator definitions are separately required.

### 5. Ratio and normalization

Provide the exact charge-ratio formula, including any square-root kinematic factor, source/sink normalization, sign, complex/real component selection, and expected reference value. State whether the expected plateau is `1`, a flavor charge, or another normalization.

### 6. Time/contact/boundary policy

Provide:

- source times;
- sink separations or absolute sink times;
- allowed insertion times;
- source/sink contact times and contact-term treatment;
- excluded boundary-stencil times;
- plateau window;
- whether periodic wrapping is included in the fit.

## After approval

Once the six decisions are recorded in an approval document:

1. fill `measurement-contract.template.json` and set `status=approved`;
2. hash the approval document;
3. run `audit_measurement_readiness.py --verify-files`;
4. promote the candidate C2 only if it matches exactly;
5. generate the missing C3 on Kunshan using the approved topology;
6. build audited `(Ncfg,Lt)` charge/WT inputs and run existing ensemble gates.

Until approval, the code deliberately refuses to invent these conventions.
