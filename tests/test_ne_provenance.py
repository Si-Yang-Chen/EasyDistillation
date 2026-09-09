from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from lattice import set_backend
from lattice.quark_diagram import Meson, Propagator, PropagatorLocal, PropagatorWithCurrent
from lattice.result_provenance import (
    build_manifest,
    current_git_state,
    finalize_result,
    load_result_manifest,
    prepare_result_directory,
    sha256_bytes,
)

set_backend("numpy")


class Operator:
    parts = [0, [(1.0, 0, 0, None)]]


class ArrayLoader:
    def __init__(self, data, *, ne=None, np_=None):
        self.data = data
        self.Ne = ne
        self.Np = np_
        self.cache_version = 1
        self.load_count = 0
        self.elem = type(
            "Element",
            (),
            {"shape": list(data.shape), "dtype": str(data.dtype)},
        )()

    def load(self, key):
        self.load_count += 1
        return self.data


@pytest.fixture
def arrays():
    rng = np.random.default_rng(20260830)
    elemental = (
        rng.normal(size=(1, 1, 2, 3, 3))
        + 1j * rng.normal(size=(1, 1, 2, 3, 3))
    )
    perambulator = (
        rng.normal(size=(2, 2, 4, 4, 3, 3))
        + 1j * rng.normal(size=(2, 2, 4, 4, 3, 3))
    )
    return elemental.astype(np.complex128), perambulator.astype(np.complex128)


def test_same_key_ne_switch_rebuilds_meson_and_vsv_caches(arrays):
    elemental, perambulator = arrays
    elemental_loader = ArrayLoader(elemental, ne=3)
    perambulator_loader = ArrayLoader(perambulator, ne=3)

    meson = Meson(elemental_loader, Operator(), source=False)
    propagator = Propagator(perambulator_loader, Lt=2)
    local = PropagatorLocal(perambulator_loader, Lt=2)

    for value, shape in ((3, (3, 3)), (1, (1, 1)), (0, (0, 0)), (3, (3, 3))):
        meson.load("cfg", value)
        propagator.load("cfg", value)
        local.load("cfg", value)
        assert meson.get(0).shape[-2:] == shape
        assert propagator.get(0, 1).shape[-2:] == shape
        assert local.get(0, 0).shape[-2:] == shape

    assert elemental_loader.load_count == 4
    assert perambulator_loader.load_count == 8


@pytest.mark.parametrize("used_ne", [-1, 4, 1.5, True])
def test_ne_bounds_are_rejected(arrays, used_ne):
    elemental, perambulator = arrays
    objects = (
        Meson(ArrayLoader(elemental, ne=3), Operator(), False),
        Propagator(ArrayLoader(perambulator, ne=3), 2),
        PropagatorLocal(ArrayLoader(perambulator, ne=3), 2),
    )
    for obj in objects:
        with pytest.raises((ValueError, TypeError)):
            obj.load("cfg", used_ne)


def test_loader_version_change_rebuilds_same_key_cache(arrays):
    elemental, _ = arrays
    loader = ArrayLoader(elemental, ne=3)
    meson = Meson(loader, Operator(), False)
    meson.load("cfg", 1)
    first = meson.get(0).copy()
    loader.data = loader.data * 2
    loader.cache_version += 1
    meson.load("cfg", 1)
    second = meson.get(0)
    np.testing.assert_allclose(second, 2 * first)


def _current_propagator(total_ne=3, total_np=2):
    rng = np.random.default_rng(20260831)
    vsv = ArrayLoader(
        rng.normal(size=(2, 2, 4, 4, total_ne, total_ne)).astype(np.complex128),
        ne=total_ne,
    )
    vsp = ArrayLoader(
        rng.normal(size=(2, 2, 4, 4, total_ne, total_np, 3)).astype(np.complex128),
        ne=total_ne,
        np_=total_np,
    )
    psv = ArrayLoader(
        rng.normal(size=(2, 2, 4, 4, total_np, 3, total_ne)).astype(np.complex128),
        ne=total_ne,
        np_=total_np,
    )
    overlap = ArrayLoader(
        rng.normal(size=(2, total_ne, total_np, 3)).astype(np.complex128),
        ne=total_ne,
        np_=total_np,
    )
    return PropagatorWithCurrent(
        vsv=vsv,
        vsp=vsp,
        psv=psv,
        psp=None,
        overlap_matrix=overlap,
        Lt=2,
    )


def test_propagator_with_current_same_key_ne_switch_rebuilds_caches():
    propagator = _current_propagator()
    propagator.load("cfg", usedNe=3, usedNp=2)
    assert propagator.get_VSP(0, 1).shape[-3:] == (3, 2, 3)
    propagator.load("cfg", usedNe=1, usedNp=1)
    assert propagator.get_VSP(0, 1).shape[-3:] == (1, 1, 3)
    assert propagator.get_PSV(0, 1).shape[-3:] == (1, 3, 1)


def test_propagator_with_current_release_clears_all_derived_caches():
    propagator = _current_propagator()
    propagator.load("cfg", usedNe=2, usedNp=1)
    propagator.get_VSP(0, 1)
    assert propagator.vsp_cache is not None
    propagator.release()
    assert propagator.key is None
    assert propagator.perambulator_data is None
    assert propagator.vsp_data is None
    assert propagator.vsp_cache is None
    assert propagator.psv_cache is None
    assert propagator.psp_cache is None
    assert propagator.tilde_S_vsp_cache is None
    assert propagator.tilde_S_psv_cache is None
    assert propagator.tilde_S_psp_cache is None


def test_source_sink_asymmetric_ne_shapes_and_zero_boundary():
    propagator = _current_propagator()
    propagator.load("cfg", usedNe=3, usedNp=2)

    vsp = propagator.get_VSP_highmode(
        0, 1, usedNe_source=1, usedNe_sink=2
    )
    psv = propagator.get_PSV_highmode(
        0, 1, usedNe_sink=2, usedNe_source=1
    )
    assert vsp.shape == (4, 4, 2, 2, 3)
    assert psv.shape == (4, 4, 2, 3, 1)

    vsp_unprojected = propagator.get_VSP_highmode(
        0, 1, usedNe_source=0, usedNe_sink=2
    )
    psv_unprojected = propagator.get_PSV_highmode(
        0, 1, usedNe_sink=0, usedNe_source=1
    )
    assert vsp_unprojected.shape == (4, 4, 2, 2, 3)
    assert psv_unprojected.shape == (4, 4, 2, 3, 1)

    with pytest.raises(ValueError):
        propagator.get_VSP_highmode(0, 1, usedNe_source=4, usedNe_sink=1)
    with pytest.raises(ValueError):
        propagator.get_PSV_highmode(0, 1, usedNe_sink=1, usedNe_source=-1)


def _code_state():
    return {
        "repository": "C:/repo",
        "commit": "a" * 40,
        "dirty": True,
        "status_sha256": "b" * 64,
        "worktree_sha256": "e" * 64,
    }


def _manifest(tmp_path, source_ne, sink_ne, *, attempt=1, parameters=None):
    return build_manifest(
        output_root=tmp_path,
        logical_test_id="ne-convergence",
        configuration="cfg10000",
        source_ne=source_ne,
        sink_ne=sink_ne,
        available_ne=3,
        attempt=attempt,
        code=_code_state(),
        inputs={"eigenvectors": "c" * 64, "gauge": "d" * 64},
        parameters=parameters or {"seed": 20260830, "normalization": "none"},
    )


def test_result_directories_are_isolated_and_manifest_complete(tmp_path):
    manifests = [
        _manifest(tmp_path, 1, 1),
        _manifest(tmp_path, 1, 2),
        _manifest(tmp_path, 2, 1),
        _manifest(tmp_path, 1, 1, attempt=2),
    ]
    directories = {manifest["result_directory"] for manifest in manifests}
    assert len(directories) == len(manifests)
    for manifest in manifests:
        directory = prepare_result_directory(manifest)
        output = directory / "correlator.npy"
        np.save(output, np.array([manifest["source_ne"], manifest["sink_ne"]]))
        completed = finalize_result(manifest, {"correlator.npy": output})
        loaded = load_result_manifest(directory)
        assert loaded["verified"]
        assert loaded["outputs"] == completed["outputs"]
        required = {
            "source_ne",
            "sink_ne",
            "inputs",
            "code",
            "parameters",
            "config_sha256",
            "run_id",
        }
        assert required <= loaded.keys()


def test_concurrent_prepare_allows_exactly_one_writer(tmp_path):
    manifest = _manifest(tmp_path, 2, 2)

    def prepare():
        try:
            prepare_result_directory(manifest)
            return "created"
        except FileExistsError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: prepare(), range(2)))
    assert sorted(outcomes) == ["conflict", "created"]
    assert (Path(manifest["result_directory"]) / "manifest.json").is_file()


def test_missing_code_provenance_is_rejected(tmp_path):
    code = _code_state()
    del code["status_sha256"]
    with pytest.raises(ValueError, match="code must contain"):
        build_manifest(
            output_root=tmp_path,
            logical_test_id="ne-convergence",
            configuration="cfg10000",
            source_ne=1,
            sink_ne=1,
            available_ne=3,
            attempt=1,
            code=code,
            inputs={"gauge": "d" * 64},
            parameters={"normalization": "none"},
        )


def test_result_directory_conflict_and_tamper_are_rejected(tmp_path):
    manifest = _manifest(tmp_path, 1, 1)
    directory = prepare_result_directory(manifest)
    with pytest.raises(FileExistsError):
        prepare_result_directory(manifest)

    output = directory / "correlator.npy"
    np.save(output, np.array([1.0]))
    finalize_result(manifest, {"correlator.npy": output})
    np.save(output, np.array([2.0]))
    with pytest.raises(ValueError, match="output does not match"):
        load_result_manifest(directory)


def test_legacy_result_requires_explicit_unverified_mode(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    np.save(legacy / "10000.npy", np.ones(2))
    with pytest.raises(ValueError, match="legacy result"):
        load_result_manifest(legacy)
    loaded = load_result_manifest(legacy, allow_legacy=True)
    assert loaded["status"] == "legacy-unverified"
    assert loaded["verified"] is False


def test_manifest_content_tamper_is_rejected(tmp_path):
    manifest = _manifest(tmp_path, 1, 2)
    directory = prepare_result_directory(manifest)
    path = directory / "manifest.json"
    tampered = json.loads(path.read_text())
    tampered["sink_ne"] = 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="does not match"):
        load_result_manifest(directory)


def test_contraction_entrypoints_fail_before_gpu_without_manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("LOCALIZED_INPUT_MANIFEST", None)
    environment.pop("LOCALIZED_RESULT_ROOT", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for relative_path in ("4.contraction.py", "examples/4_contraction.py"):
        completed = subprocess.run(
            [sys.executable, str(root / relative_path)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert completed.returncode != 0
        assert "LOCALIZED_INPUT_MANIFEST and LOCALIZED_RESULT_ROOT are required" in completed.stderr
        assert "Using backend: cupy" not in completed.stdout
        source = (root / relative_path).read_text()
        assert "05.correlator.current.nonlocal" not in source
        assert "exit()" not in source


def test_staged_tracked_content_changes_worktree_identity(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            text=True,
            capture_output=True,
            check=True,
        )

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Provenance Test")
    tracked = repository / "tracked.txt"
    tracked.write_text("baseline\n")
    git("add", "tracked.txt")
    git("commit", "-m", "baseline")

    states = []
    manifests = []
    for content in ("version-a\n", "version-b\n"):
        tracked.write_text(content)
        git("add", "tracked.txt")
        state = current_git_state(repository)
        states.append(state)
        manifests.append(
            build_manifest(
                output_root=tmp_path / "results",
                logical_test_id="staged-provenance",
                configuration="cfg",
                source_ne=1,
                sink_ne=1,
                available_ne=1,
                attempt=1,
                code=state,
                inputs={"gauge": "d" * 64},
                parameters={"normalization": "none"},
            )
        )

    assert states[0]["status_sha256"] == states[1]["status_sha256"]
    assert states[0]["worktree_sha256"] != states[1]["worktree_sha256"]
    assert manifests[0]["run_id"] != manifests[1]["run_id"]


def test_worktree_content_hash_changes_run_identity(tmp_path):
    first = _manifest(tmp_path, 1, 1)
    changed_code = _code_state()
    changed_code["worktree_sha256"] = "f" * 64
    second = build_manifest(
        output_root=tmp_path,
        logical_test_id="ne-convergence",
        configuration="cfg10000",
        source_ne=1,
        sink_ne=1,
        available_ne=3,
        attempt=1,
        code=changed_code,
        inputs={"eigenvectors": "c" * 64, "gauge": "d" * 64},
        parameters={"seed": 20260830, "normalization": "none"},
    )
    assert first["run_id"] != second["run_id"]


def test_parameters_change_run_identity(tmp_path):
    first = _manifest(tmp_path, 1, 1, parameters={"normalization": "none"})
    second = _manifest(tmp_path, 1, 1, parameters={"normalization": "volume"})
    assert first["run_id"] != second["run_id"]
    assert sha256_bytes(b"none") != sha256_bytes(b"volume")
