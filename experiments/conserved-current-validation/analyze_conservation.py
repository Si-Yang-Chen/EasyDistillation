"""Deterministic, CPU-only statistics for conserved-current validation."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np


def jackknife(values: np.ndarray) -> dict[str, np.ndarray | int]:
    """Return mean, jackknife error, replicas, and count along configuration axis."""
    data = np.asarray(values, dtype=float)
    if data.ndim < 1 or data.shape[0] < 2:
        raise ValueError("jackknife needs at least two configurations")
    if not np.all(np.isfinite(data)):
        raise ValueError("jackknife values must be finite")
    count = data.shape[0]
    total = data.sum(axis=0)
    replicas = (total[None, ...] - data) / (count - 1)
    replica_mean = replicas.mean(axis=0)
    error = np.sqrt((count - 1) / count * np.sum((replicas - replica_mean) ** 2, axis=0))
    return {"mean": data.mean(axis=0), "error": error, "replicas": replicas, "count": count}


def jackknife_covariance(values: np.ndarray) -> dict[str, Any]:
    """Configuration-jackknife covariance, stable rank, and pseudo-inverse."""
    summary = jackknife(values)
    replicas = np.asarray(summary["replicas"], dtype=float)
    if replicas.ndim != 2:
        raise ValueError("covariance input must have shape (configuration, observable)")
    centered = replicas - replicas.mean(axis=0)
    covariance = (replicas.shape[0] - 1) / replicas.shape[0] * centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    cutoff = np.finfo(float).eps * max(covariance.shape) * scale * 100.0
    keep = eigenvalues > cutoff
    rank = int(np.count_nonzero(keep))
    inverse = np.zeros_like(covariance)
    if rank:
        inverse = (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
    return {
        **summary,
        "covariance": covariance,
        "inverse": inverse,
        "rank": rank,
        "cutoff": cutoff,
        "valid": bool(rank > 0),
    }


def chi2_p_value(chi2: float, dof: int) -> tuple[float | None, str | None]:
    """Return scipy p-value when available; never silently approximate it."""
    if dof <= 0 or not isfinite(chi2):
        return None, "covariance rank does not support positive degrees of freedom"
    try:
        from scipy.stats import chi2 as chi2_distribution
    except ImportError:
        return None, "scipy unavailable; p-value intentionally not approximated"
    return float(chi2_distribution.sf(chi2, dof)), None


def correlated_constant_fit(values: np.ndarray) -> dict[str, Any]:
    """Fit a correlated constant with jackknife covariance and rank-aware chi2."""
    covariance = jackknife_covariance(values)
    mean = np.asarray(covariance["mean"], dtype=float)
    inverse = np.asarray(covariance["inverse"], dtype=float)
    ones = np.ones(mean.size, dtype=float)
    denominator = float(ones @ inverse @ ones)
    if covariance["rank"] < 2 or denominator <= 0:
        return {
            **covariance,
            "constant": None,
            "constant_error": None,
            "chi2": None,
            "dof": 0,
            "p_value": None,
            "p_value_reason": "covariance rank does not support a correlated constant fit",
            "fit_valid": False,
        }
    constant = float(ones @ inverse @ mean / denominator)
    residual = mean - constant
    chi2 = float(residual @ inverse @ residual)
    dof = int(covariance["rank"] - 1)
    p_value, reason = chi2_p_value(chi2, dof)
    return {
        **covariance,
        "constant": constant,
        "constant_error": float(np.sqrt(1.0 / denominator)),
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "p_value_reason": reason,
        "fit_valid": True,
    }


def jsonable(value: Any) -> Any:
    """Convert NumPy values deterministically for JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
