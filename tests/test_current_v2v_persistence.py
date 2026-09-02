from hashlib import sha256
import json

import numpy as np
import pytest

from lattice.current_elemental import (
    CURRENT_V2V_ARTIFACT_SCHEMA,
    load_directed_current_v2v,
    save_directed_current_v2v,
)
from lattice.insertion.current import build_current_raw_contract


def _raw():
    values = np.arange(8 * 3 * 2 * 2 * 2, dtype=np.float64).reshape(8, 3, 2, 2, 2)
    raw = {"v2v": values.astype(np.complex128) * (1 + 2j)}
    contract = build_current_raw_contract(
        raw,
        boundary="periodic",
        available_ne=3,
        used_ne=2,
        momentum_count=2,
    )
    return raw, contract


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def test_directed_current_v2v_artifact_round_trip_and_source_binding(tmp_path):
    raw, contract = _raw()
    gauge = tmp_path / "gauge.lime"
    eigenvector = tmp_path / "eigenvector.npy"
    gauge.write_bytes(b"audited-gauge-input")
    eigenvector.write_bytes(b"audited-eigenvector-input")

    manifest_path = save_directed_current_v2v(
        tmp_path / "artifact",
        raw,
        contract,
        configuration="cfg-001",
        momenta=[(0, 0, 0), (1, -1, 0)],
        gauge_source=gauge,
        eigenvector_source=eigenvector,
    )
    loaded = load_directed_current_v2v(
        manifest_path,
        expected_configuration="cfg-001",
        expected_gauge_sha256=_digest(gauge),
        expected_eigenvector_sha256=_digest(eigenvector),
    )

    assert loaded["manifest"]["schema"] == CURRENT_V2V_ARTIFACT_SCHEMA
    assert loaded["manifest"]["momenta"] == [[0, 0, 0], [1, -1, 0]]
    assert loaded["manifest"]["consumer"]["term_contraction"] == ("afAi,bfji,bcjC->acAC")
    assert loaded["manifest"]["sources"]["gauge"] == {
        "path": gauge.resolve().as_posix(),
        "sha256": _digest(gauge),
    }
    assert isinstance(loaded["raw"]["v2v"], np.memmap)
    np.testing.assert_array_equal(loaded["raw"]["v2v"], raw["v2v"])
    assert loaded["contract"] == contract
    assert not list((tmp_path / "artifact").glob("*.tmp"))


def test_directed_current_v2v_artifact_rejects_data_and_manifest_tampering(tmp_path):
    raw, contract = _raw()
    gauge = tmp_path / "gauge"
    eigenvector = tmp_path / "eigenvector"
    gauge.write_bytes(b"gauge")
    eigenvector.write_bytes(b"eigenvector")
    manifest_path = save_directed_current_v2v(
        tmp_path / "artifact",
        raw,
        contract,
        configuration="cfg",
        momenta=[(0, 0, 0), (1, 0, 0)],
        gauge_source=gauge,
        eigenvector_source=eigenvector,
    )
    manifest = json.loads(manifest_path.read_text())
    data_path = manifest_path.parent / manifest["data"]["filename"]
    data_path.write_bytes(data_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="data hash"):
        load_directed_current_v2v(manifest_path)

    manifest_path = save_directed_current_v2v(
        tmp_path / "artifact-2",
        raw,
        contract,
        configuration="cfg",
        momenta=[(0, 0, 0), (1, 0, 0)],
        gauge_source=gauge,
        eigenvector_source=eigenvector,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["configuration"] = "forged"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        load_directed_current_v2v(manifest_path)

    manifest_path = save_directed_current_v2v(
        tmp_path / "artifact-3",
        raw,
        contract,
        configuration="cfg",
        momenta=[(0, 0, 0), (1, 0, 0)],
        gauge_source=gauge,
        eigenvector_source=eigenvector,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["data"]["filename"] = "renamed.npy"
    semantic = {key: value for key, value in manifest.items() if key != "artifact_identity"}
    manifest["artifact_identity"] = sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content-addressed"):
        load_directed_current_v2v(manifest_path)


def test_artifact_requires_real_sources_and_refuses_overwrite(tmp_path):
    raw, contract = _raw()
    gauge = tmp_path / "gauge"
    eigenvector = tmp_path / "eigenvector"
    gauge.write_bytes(b"gauge")
    eigenvector.write_bytes(b"eigenvector")
    destination = tmp_path / "artifact"
    kwargs = {
        "configuration": "cfg",
        "momenta": [(0, 0, 0), (1, 0, 0)],
        "gauge_source": gauge,
        "eigenvector_source": eigenvector,
    }
    save_directed_current_v2v(destination, raw, contract, **kwargs)
    with pytest.raises(FileExistsError):
        save_directed_current_v2v(destination, raw, contract, **kwargs)
    with pytest.raises(FileNotFoundError, match="gauge"):
        save_directed_current_v2v(
            tmp_path / "missing-source",
            raw,
            contract,
            **{**kwargs, "gauge_source": tmp_path / "missing"},
        )

    gauge.write_bytes(b"modified-gauge")
    with pytest.raises(ValueError, match="gauge source file"):
        load_directed_current_v2v(destination)
    loaded = load_directed_current_v2v(destination, verify_sources=False)
    assert loaded["manifest"]["configuration"] == "cfg"
    with pytest.raises(ValueError, match="read-only"):
        load_directed_current_v2v(destination, verify_sources=False, mmap_mode="r+")
