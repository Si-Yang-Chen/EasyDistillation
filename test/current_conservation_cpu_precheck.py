"""Synthetic/free-field CPU current-conservation precheck (not a gauge experiment)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Permit ``python test/current_conservation_cpu_precheck.py`` from any cwd in this repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lattice import set_backend  # noqa: E402
from lattice.insertion.current import (  # noqa: E402
    CURRENT_API_VERSION,
    CURRENT_ASSEMBLER_SCHEMA,
    CURRENT_TERM_SCHEMA,
    ConservedVectorCurrent,
    assemble_spin_aware_current,
    lattice_divergence,
)


LATTICE_SHAPE = (2, 2, 2, 4)  # x, y, z, t; all directions periodic
THRESHOLD = 1.0e-12
NE = 1


def _code_hash() -> str:
    return hashlib.sha256((ROOT / "lattice" / "insertion" / "current.py").read_bytes()).hexdigest()


def _api_hash() -> str:
    semantic_api = {
        "api_version": CURRENT_API_VERSION,
        "term_schema": CURRENT_TERM_SCHEMA,
        "assembler_schema": CURRENT_ASSEMBLER_SCHEMA,
    }
    return hashlib.sha256(json.dumps(semantic_api, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _raw_resolver(term, *, endpoints, source_ne, sink_ne):
    """Free-field raw basis: unit V2V contribution at every site and endpoint."""
    del endpoints
    if source_ne != NE or sink_ne != NE:
        raise ValueError("precheck uses exactly one source and sink mode")
    # The identical raw basis makes each assembled point-split flux a constant
    # explicit r/gamma matrix.  No coefficient is applied here.
    return {
        "value": np.ones((sink_ne, source_ne), dtype=np.complex128),
        "source_ne": source_ne,
        "sink_ne": sink_ne,
        "provenance": {"basis": "synthetic-free-field-unit-v2v"},
    }


def _assembled_flux(current, mu, site):
    """Resolve/assemble one direction at one site through the public path."""
    del site  # translationally invariant synthetic raw basis
    terms = current.terms[2 * mu : 2 * mu + 2]
    result = assemble_spin_aware_current(
        terms,
        _raw_resolver,
        available_source_ne=NE,
        available_sink_ne=NE,
        used_source_ne=NE,
        used_sink_ne=NE,
        anchor_time=0,
        temporal_extent=LATTICE_SHAPE[3],
        boundary="periodic",
    )
    return result["value"][:, :, 0, 0]


def build_precheck() -> dict:
    """Run the deterministic periodic free-field identity and return JSON data."""
    set_backend("numpy")
    current = ConservedVectorCurrent(wilson_r=1.0)
    nx, ny, nz, nt = LATTICE_SHAPE
    flux = np.empty((4, 4, 4, nx, ny, nz, nt), dtype=np.complex128)
    for mu in range(4):
        matrix = _assembled_flux(current, mu, (0, 0, 0, 0))
        flux[mu] = matrix[:, :, None, None, None, None]

    # Select one explicit spin component as a scalar charge-current channel.
    scalar_flux = flux[:, 0, 2]
    divergence = lattice_divergence(scalar_flux, periodic=True)
    charge = scalar_flux[3].sum(axis=(0, 1, 2))
    charge_residual = charge - charge[0]
    residual_per_site = np.abs(divergence)
    residual_per_time = np.max(residual_per_site, axis=(0, 1, 2))
    max_residual = float(max(np.max(residual_per_site), np.max(np.abs(charge_residual))))

    # A deliberate perturbation is reported as a diagnostic, not folded into pass.
    perturbed = scalar_flux.copy()
    perturbed[0, 0, 0, 0, 0] += 0.25
    perturbed_max = float(np.max(np.abs(lattice_divergence(perturbed, periodic=True))))
    return {
        "kind": "synthetic_free_field_cpu_precheck",
        "warning": "not a real-gauge experiment; no production propagator or gauge field",
        "formula": {
            "divergence": "sum_mu (J_mu(x) - J_mu(x-mu))",
            "charge": "Q(t)=sum_xyz J_3(x,y,z,t), test Q(t)-Q(0)=0",
            "assembler": "sum_terms coefficient * normalization * spin_aware_raw_value",
        },
        "parameters": {
            "lattice_shape_x_y_z_t": LATTICE_SHAPE,
            "boundary": "periodic",
            "wilson_r": 1.0,
            "source_ne": NE,
            "sink_ne": NE,
            "spin_channel": "(sink_spin, source_spin)=(0,2)",
        },
        "residual_per_site": residual_per_site.tolist(),
        "residual_per_time": residual_per_time.tolist(),
        "charge": [[float(value.real), float(value.imag)] for value in charge],
        "charge_residual_per_time": np.abs(charge_residual).tolist(),
        "max_residual": max_residual,
        "threshold": THRESHOLD,
        "passed": bool(max_residual <= THRESHOLD),
        "perturbed_diagnostic": {
            "max_divergence_residual": perturbed_max,
            "passed": bool(perturbed_max <= THRESHOLD),
        },
        "provenance": {
            "api_version": CURRENT_API_VERSION,
            "term_schema": CURRENT_TERM_SCHEMA,
            "assembler_schema": CURRENT_ASSEMBLER_SCHEMA,
            "api_sha256": _api_hash(),
            "code_sha256": _code_hash(),
            "gauge_configuration": "none (synthetic unit raw basis)",
            "propagator_artifact": "placeholder: not available in CPU precheck",
            "experiment": "placeholder: remote real-gauge experiment not run",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON here")
    args = parser.parse_args()
    result = build_precheck()
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
