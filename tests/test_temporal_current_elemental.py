import json

import numpy as np
import pytest

from lattice import CurrentElementalGenerator, set_backend
from lattice.insertion.current import (
    CURRENT_API_VERSION,
    CURRENT_DIRECTED_RAW_SCHEMA,
    CURRENT_ELEMENTAL_SCHEMA,
    LocalVectorCurrent,
    ConservedVectorCurrent,
    assemble_current_terms,
    assemble_spin_aware_current,
    spin_aware_current_adapter,
    build_current_raw_contract,
    current_raw_cache_key,
    resolve_current_term_endpoints,
    resolve_directed_current_raw,
    validate_legacy_spatial_current_raw,
    validate_current_raw_contract,
)
from lattice.insertion.gauge_link import DirectedCurrentBasis, GaugeLink


def _generator(used_ne=2):
    set_backend("numpy")

    class Loader:
        Ne = 2
        Np = 1

    generator = CurrentElementalGenerator(
        latt_size=[1, 1, 1, 3],
        gauge_field=Loader(),
        eigenvector=Loader(),
        point=Loader(),
        num_nabla=0,
        momentum_list=[(0, 0, 0)],
        usedNe=used_ne,
        usedNp=1,
    )
    generator._current_U = np.zeros((4, 3, 1, 1, 1, 3, 3), dtype=np.complex128)
    for axis in range(4):
        for time in range(3):
            generator._current_U[axis, time, 0, 0, 0] = np.diag(
                [10 * axis + time + 1, 10 * axis + time + 2, 10 * axis + time + 3]
            )
    generator._U = generator._current_U[:3]
    generator._eigenvector_data = np.zeros((3, 2, 1, 1, 1, 3), dtype=np.complex128)
    generator._eigenvector_data[:, :, 0, 0, 0, :2] = np.eye(2)[None, :, :]
    generator._point_data = np.zeros((1, 3, 3), dtype=np.int64)
    return generator


def _raw_contract(boundary="periodic"):
    values = np.zeros((8, 3, 1, 2, 2), dtype=np.complex128)
    for time in range(3):
        values[6, time, 0] = np.diag([100 + time, 200 + time])
        values[7, time, 0] = np.diag([300 + time, 400 + time])
    raw = {"v2v": values}
    return raw, build_current_raw_contract(
        raw,
        boundary=boundary,
        available_ne=2,
        used_ne=2,
        momentum_count=1,
    )


def _nonhermitian_generator():
    """Three-time complex transport fixture; references use atol=rtol=1e-12."""
    set_backend("numpy")

    class Loader:
        Ne = 2
        Np = 1

    generator = CurrentElementalGenerator(
        latt_size=[2, 1, 1, 3],
        gauge_field=Loader(),
        eigenvector=Loader(),
        point=Loader(),
        num_nabla=0,
        momentum_list=[(0, 0, 0)],
        usedNe=2,
        usedNp=1,
    )
    generator._current_U = np.empty((4, 3, 1, 1, 2, 3, 3), dtype=np.complex128)
    generator._eigenvector_data = np.empty((3, 2, 1, 1, 2, 3), dtype=np.complex128)
    for axis in range(4):
        for time in range(3):
            for x_coord in range(2):
                base = 1 + 17 * axis + 5 * time + 3 * x_coord
                generator._current_U[axis, time, 0, 0, x_coord] = np.array(
                    [
                        [base + 1j, 2 - 3j, -1 + 2j],
                        [4 + 2j, base + 5j, 3 - 1j],
                        [2 - 4j, -3 + 1j, base + 2 - 2j],
                    ],
                    dtype=np.complex128,
                )
    for time in range(3):
        for eigenvector in range(2):
            for x_coord in range(2):
                base = 2 + 7 * time + 4 * eigenvector + x_coord
                generator._eigenvector_data[time, eigenvector, 0, 0, x_coord] = np.array(
                    [base + 1j, 2 - base * 1j, 3 * base + 2j],
                    dtype=np.complex128,
                )
    generator._U = generator._current_U[:3]
    generator._point_data = np.zeros((1, 3, 3), dtype=np.int64)
    return generator


def _reference_v2v(left, link, right):
    return np.einsum("ezyxa,zyxac,fzyxc->ef", left.conj(), link, right)


def _legacy_spatial_raw_contract():
    raw = {
        "v2v": np.ones((2, 1, 2, 2), dtype=np.complex128),
        "v2p": np.ones((2, 2, 1, 3), dtype=np.complex128),
        "p2v": np.ones((2, 1, 3, 2), dtype=np.complex128),
        "p2p": [
            {"type": "identity"},
            {
                "type": "sparse",
                "indices": np.array([[0, 0]], dtype=np.int64),
                "values": np.ones((1, 3, 3), dtype=np.complex128),
            },
        ],
    }
    contract = {
        "schema": CURRENT_ELEMENTAL_SCHEMA,
        "representation": "raw-spatial-displacement-basis",
        "combined_with_current_terms": False,
        "supports_temporal_point_split": False,
        "term_application": "external-resolver-required",
        "term_schema": "lattice.current.term/v1",
        "assembler_schema": "lattice.current.assembler/v1",
        "ne": {
            "available": 2,
            "used": 2,
            "source": 2,
            "sink": 2,
            "requested_source": 1,
            "requested_sink": 2,
            "raw_generator_used_ne_is_symmetric": True,
        },
        "np": {"available": 1, "used": 1},
        "shapes": {key: tuple(np.shape(value)) for key, value in raw.items()},
        "axes": {
            "v2v": ("displacement", "momentum", "sink_ne", "source_ne"),
            "v2p": ("displacement", "sink_ne", "point", "color"),
            "p2v": ("displacement", "point", "color", "source_ne"),
            "p2p": "sparse-per-displacement",
        },
    }
    return raw, contract


def test_directed_basis_is_exact_json_native_and_gaugelink_remains_spatial():
    metadata = DirectedCurrentBasis.metadata()
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["schema"] == "lattice.current.directed-one-link-basis/v1"
    assert tuple(item["name"] for item in metadata["directions"]) == (
        "+x",
        "+y",
        "+z",
        "-x",
        "-y",
        "-z",
        "+t",
        "-t",
    )
    assert [item["vector"] for item in metadata["directions"]] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -1],
    ]
    assert [GaugeLink([index]).idx for index in range(6)] == [1, 2, 3, 4, 5, 6]
    assert [GaugeLink([index]).displacement for index in range(6)] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]


def test_temporal_raw_nonhermitian_transport_uses_dagger_and_periodic_wrap():
    """Atol=rtol=1e-12 for V(t)^dagger U V(t+1) and V(t)^dagger U(t-1)^dagger V(t-1)."""
    generator = _nonhermitian_generator()
    values = generator.calc_directed_current_raw("periodic")["v2v"]
    tolerance = {"rtol": 1e-12, "atol": 1e-12}

    for time in range(3):
        left = generator._eigenvector_data[time, :2]
        forward_source = generator._eigenvector_data[(time + 1) % 3, :2]
        backward_source = generator._eigenvector_data[(time - 1) % 3, :2]
        forward_link = generator._current_U[3, time]
        backward_link = generator._current_U[3, (time - 1) % 3].conj().transpose(0, 1, 2, 4, 3)
        expected_forward = _reference_v2v(left, forward_link, forward_source)
        expected_backward = _reference_v2v(left, backward_link, backward_source)
        np.testing.assert_allclose(values[6, time, 0], expected_forward, **tolerance)
        np.testing.assert_allclose(values[7, time, 0], expected_backward, **tolerance)

    interior = 1
    left = generator._eigenvector_data[interior, :2]
    wrong_link = generator._current_U[3, 0]
    source = generator._eigenvector_data[0, :2]
    for wrong_operation in (
        wrong_link,
        wrong_link.transpose(0, 1, 2, 4, 3),
        wrong_link.conj(),
    ):
        wrong = _reference_v2v(left, wrong_operation, source)
        assert not np.allclose(values[7, interior, 0], wrong, **tolerance)

    wrap_forward = _reference_v2v(
        generator._eigenvector_data[2, :2],
        generator._current_U[3, 2],
        generator._eigenvector_data[0, :2],
    )
    np.testing.assert_allclose(values[6, 2, 0], wrap_forward, **tolerance)
    wrap_backward = _reference_v2v(
        generator._eigenvector_data[0, :2],
        generator._current_U[3, 2].conj().transpose(0, 1, 2, 4, 3),
        generator._eigenvector_data[2, :2],
    )
    np.testing.assert_allclose(values[7, 0, 0], wrap_backward, **tolerance)


def test_spatial_backward_nonhermitian_transport_matches_direct_reference():
    """Atol=rtol=1e-12 for V(x)^dagger U_x(x-1)^dagger V(x-1)."""
    generator = _nonhermitian_generator()
    values = generator.calc_directed_current_raw("periodic")["v2v"]
    time = 1
    left = generator._eigenvector_data[time, :2]
    link = np.roll(generator._current_U[0, time], 1, axis=2).conj().transpose(0, 1, 2, 4, 3)
    source = np.roll(generator._eigenvector_data[time, :2], 1, axis=3)
    np.testing.assert_allclose(values[3, time, 0], _reference_v2v(left, link, source), rtol=1e-12, atol=1e-12)


def test_temporal_raw_links_obey_transport_and_periodic_wrap():
    raw = _generator().calc_directed_current_raw("periodic")
    values = raw["v2v"]
    assert values.shape == (8, 3, 1, 2, 2)
    assert values.dtype == np.dtype("<c16")
    for time in range(3):
        np.testing.assert_allclose(values[6, time, 0], np.diag([31 + time, 32 + time]))
        previous = (time - 1) % 3
        np.testing.assert_allclose(values[7, time, 0], np.diag([31 + previous, 32 + previous]))
    assert raw["contract"]["schema"] == CURRENT_DIRECTED_RAW_SCHEMA
    assert raw["contract"]["channels"] == ["v2v-one-link"]
    assert raw["contract"]["axes"]["v2v"] == [
        "direction",
        "time",
        "momentum",
        "sink_ne",
        "source_ne",
    ]


def test_spatial_directed_links_match_legacy_one_link_v2v():
    generator = _generator()
    generator.num_nabla = 1
    generator.num_disp = 7
    legacy = generator.calc_v2v(0)
    raw = generator.calc_directed_current_raw()["v2v"]
    np.testing.assert_allclose(raw[:6, 0, 0], legacy[1:7, 0], rtol=1e-14, atol=1e-14)


def test_open_temporal_boundary_zeroes_invalid_one_link_transport():
    values = _generator().calc_directed_current_raw("open")["v2v"]
    np.testing.assert_array_equal(values[6, 2], 0)
    np.testing.assert_array_equal(values[7, 0], 0)
    assert not np.all(values[6, 0] == 0)
    assert not np.all(values[7, 1] == 0)


def test_contract_json_round_trip_and_cache_identity_are_strict():
    raw, contract = _raw_contract()
    restored = json.loads(json.dumps(contract, sort_keys=True))
    assert validate_current_raw_contract(raw, restored) == contract
    assert current_raw_cache_key("cfg", restored) == current_raw_cache_key("cfg", contract)

    for field, value in (
        ("boundary", "open"),
        ("version", 2),
        ("basis", {**contract["basis"], "directions": []}),
        ("shapes", {"v2v": [8, 3, 1, 1, 1]}),
        ("ne", {**contract["ne"], "used": 1, "source": 1, "sink": 1}),
    ):
        tampered = json.loads(json.dumps(contract))
        tampered[field] = value
        tampered["cache_identity"] = contract["cache_identity"]
        with pytest.raises(ValueError, match="identity|schema/version|directions|shape|Ne"):
            current_raw_cache_key("cfg", tampered)

    changed = build_current_raw_contract(
        raw,
        boundary="open",
        available_ne=2,
        used_ne=2,
        momentum_count=1,
    )
    assert current_raw_cache_key("cfg", changed) != current_raw_cache_key("cfg", contract)
    unknown = dict(contract)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_current_raw_contract(raw, unknown)
    with pytest.raises(ValueError, match="legacy spatial"):
        validate_current_raw_contract(raw, {"schema": CURRENT_ELEMENTAL_SCHEMA}, require_temporal=True)


def test_backward_uses_bar_endpoint_as_raw_anchor_and_wraps_periodically():
    raw, contract = _raw_contract()
    forward, backward = ConservedVectorCurrent().terms[6:8]
    assert CURRENT_API_VERSION == "1.2.0"

    for anchor_time, expected_forward, expected_backward in (
        (1, 1, 2),
        (2, 2, 0),
    ):
        forward_endpoints = resolve_current_term_endpoints(
            forward,
            anchor_time=anchor_time,
            temporal_extent=3,
            boundary="periodic",
        )
        backward_endpoints = resolve_current_term_endpoints(
            backward,
            anchor_time=anchor_time,
            temporal_extent=3,
            boundary="periodic",
        )
        assert backward_endpoints["link_origin_time"] == anchor_time
        assert backward_endpoints["bar_time"] == expected_backward
        resolved_forward = resolve_directed_current_raw(
            raw,
            contract,
            forward,
            endpoints=forward_endpoints,
            source_ne=2,
            sink_ne=2,
        )
        resolved_backward = resolve_directed_current_raw(
            raw,
            contract,
            backward,
            endpoints=backward_endpoints,
            source_ne=2,
            sink_ne=2,
        )
        np.testing.assert_array_equal(resolved_forward["value"], raw["v2v"][6, expected_forward, 0])
        np.testing.assert_array_equal(resolved_backward["value"], raw["v2v"][7, expected_backward, 0])
        assert resolved_backward["provenance"]["raw_anchor"] == "bar_endpoint"

    def resolver(term, **kwargs):
        return resolve_directed_current_raw(raw, contract, term, **kwargs)

    result = assemble_current_terms(
        (forward, backward),
        resolver,
        available_source_ne=2,
        available_sink_ne=2,
        used_source_ne=1,
        used_sink_ne=2,
        anchor_time=2,
        temporal_extent=3,
        boundary="periodic",
    )
    expected = -0.5 * raw["v2v"][6, 2, 0, :2, :1] + 0.5 * raw["v2v"][7, 0, 0, :2, :1]
    np.testing.assert_array_equal(result["value"], expected)


def test_open_contract_endpoint_boundary_and_shape_validation():
    raw, contract = _raw_contract("open")
    backward = ConservedVectorCurrent().terms[7]
    endpoints = resolve_current_term_endpoints(
        backward,
        anchor_time=1,
        temporal_extent=3,
        boundary="open",
    )
    np.testing.assert_array_equal(
        resolve_directed_current_raw(raw, contract, backward, endpoints=endpoints, source_ne=1, sink_ne=1)["value"],
        raw["v2v"][7, 2, 0, :1, :1],
    )
    bad_boundary = dict(endpoints)
    bad_boundary["boundary"] = "periodic"
    with pytest.raises(ValueError, match="boundary"):
        resolve_directed_current_raw(raw, contract, backward, endpoints=bad_boundary, source_ne=1, sink_ne=1)
    invalid_endpoints = (
        {},
        {**endpoints, "extra": 1},
        {**endpoints, "bar_time": True},
        {**endpoints, "temporal_point_split": False},
    )
    for bad_endpoints in invalid_endpoints:
        with pytest.raises((TypeError, ValueError), match="endpoints|integer|temporal"):
            resolve_directed_current_raw(raw, contract, backward, endpoints=bad_endpoints, source_ne=1, sink_ne=1)


def test_load_retains_all_links_once_and_uses_loader_axis_order():
    set_backend("numpy")

    class GaugeData:
        def __init__(self, values):
            self.values = values
            self.file = "fake-gauge"

        def __getitem__(self, key):
            return self.values[key]

    class GaugeLoader:
        Ne = Np = 1

        def __init__(self, values):
            self.values = values
            self.calls = []
            self.data = object()

        def load(self, key):
            self.calls.append(key)
            return GaugeData(self.values)

    class EigenData:
        def __init__(self):
            self.requests = []
            self.values = np.ones((2, 2, 1, 1, 1, 3), dtype=np.complex128)

        def __getitem__(self, key):
            self.requests.append(key)
            return self.values[key]

    class EigenLoader:
        Ne = 2

        def __init__(self):
            self.data = EigenData()

        def load(self, key):
            return self.data

    class PointData:
        def __init__(self):
            self.requests = []
            self.values = np.zeros((2, 2, 3), dtype=np.int64)

        def __getitem__(self, key):
            self.requests.append(key)
            return self.values[key]

    class PointLoader:
        Np = 2

        def __init__(self):
            self.data = PointData()

        def load(self, key):
            return self.data

    external = np.zeros((2, 1, 1, 1, 4, 3, 3), dtype=np.complex128)
    for time in range(2):
        for axis in range(4):
            external[time, 0, 0, 0, axis] = np.eye(3) * (10 * time + axis)
    gauge = GaugeLoader(external)
    eigenvector = EigenLoader()
    point = PointLoader()
    generator = CurrentElementalGenerator(
        latt_size=[1, 1, 1, 2],
        gauge_field=gauge,
        eigenvector=eigenvector,
        point=point,
        momentum_list=[(0, 0, 0)],
        usedNe=1,
        usedNp=1,
    )
    generator.load("cfg")
    assert gauge.calls == ["cfg"]
    assert generator._current_U.shape == (4, 2, 1, 1, 1, 3, 3)
    assert generator._U.shape == (3, 2, 1, 1, 1, 3, 3)
    np.testing.assert_array_equal(generator._current_U, external.transpose(4, 0, 1, 2, 3, 5, 6))
    np.testing.assert_array_equal(generator._U, generator._current_U[:3])
    assert generator._gauge_field_path == "fake-gauge"
    assert gauge.data is None
    assert generator._eigenvector_data.shape == (2, 1, 1, 1, 1, 3)
    assert generator._point_data.shape == (1, 2, 3)
    assert eigenvector.data.requests == [(slice(None), slice(None, 1))]
    assert point.data.requests == [slice(None, 1)]
    raw = generator.calc_directed_current_raw()
    assert raw["contract"]["schema"] == CURRENT_DIRECTED_RAW_SCHEMA
    assert raw["contract"]["ne"]["available"] == 2
    assert raw["contract"]["ne"]["used"] == 1


def test_directed_generator_rejects_malformed_eigenvector_shape():
    generator = _generator()
    generator._eigenvector_data = generator._eigenvector_data[..., :2]
    with pytest.raises(ValueError, match="eigenvector data must have shape"):
        generator.calc_directed_current_raw()


def test_project_su3_updates_directed_spatial_links():
    generator = _generator()
    generator._U = generator._current_U[:3].copy() * 1.5
    generator.project_SU3()
    np.testing.assert_allclose(generator._current_U[:3], generator._U, rtol=0, atol=0)


@pytest.mark.parametrize("used_ne", [0, 1, 2])
def test_generator_current_raw_ne_bounds(used_ne):
    raw = _generator(used_ne).calc_directed_current_raw()
    assert raw["v2v"].shape[-2:] == (used_ne, used_ne)


def test_legacy_spatial_contract_accepts_compute_elemental_shaped_output():
    class Generator:
        Ne = usedNe = 2
        Np = usedNp = 1

        def calc_all(self, time):
            assert time == 4
            return {
                "v2v": np.ones((2, 1, 2, 2), dtype=np.complex128),
                "v2p": np.ones((2, 2, 1, 3), dtype=np.complex128),
                "p2v": np.ones((2, 1, 3, 2), dtype=np.complex128),
                "p2p": [
                    {"type": "identity"},
                    {
                        "type": "sparse",
                        "indices": np.array([[0, 0]], dtype=np.int64),
                        "values": np.ones((1, 3, 3), dtype=np.complex128),
                    },
                ],
            }

    adapted = LocalVectorCurrent().compute_elemental(Generator(), t=4)
    validated = validate_legacy_spatial_current_raw(
        adapted["all_elementals"],
        adapted["contract"],
    )
    assert validated["contract"] == adapted["contract"]


def test_legacy_spatial_contract_requires_full_evidenced_metadata_and_raw_channels():
    raw, contract = _legacy_spatial_raw_contract()
    validated = validate_legacy_spatial_current_raw(raw, contract)
    assert validated["schema"] == CURRENT_ELEMENTAL_SCHEMA
    assert validated["legacy_spatial_only"] is True
    assert validated["channels"] == ("v2v", "v2p", "p2v", "p2p")
    assert validate_current_raw_contract(raw, contract) == validated

    with pytest.raises(ValueError, match="missing or unknown"):
        validate_current_raw_contract(raw, {"schema": CURRENT_ELEMENTAL_SCHEMA})
    with pytest.raises(ValueError, match="legacy spatial"):
        validate_current_raw_contract(raw, contract, require_temporal=True)


@pytest.mark.parametrize(
    "channel, replacement",
    [
        ("v2v", np.ones((2, 1, 2, 1), dtype=np.complex128)),
        ("v2p", np.ones((2, 2, 1, 2), dtype=np.complex128)),
        ("p2v", np.ones((2, 1, 2, 2), dtype=np.complex128)),
        ("v2v", np.ones((2, 1, 2, 2), dtype=np.float64)),
        ("v2p", np.full((2, 2, 1, 3), np.nan + 0j, dtype=np.complex128)),
    ],
)
def test_legacy_spatial_contract_rejects_corrupt_numeric_channels(channel, replacement):
    raw, contract = _legacy_spatial_raw_contract()
    raw[channel] = replacement
    with pytest.raises((TypeError, ValueError), match="shape|dtype|finite"):
        validate_legacy_spatial_current_raw(raw, contract)


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("representation", "forged", "representation"),
        ("supports_temporal_point_split", True, "temporal"),
        (
            "ne",
            {
                "available": 2,
                "used": 2,
                "source": 1,
                "sink": 2,
                "requested_source": 1,
                "requested_sink": 2,
                "raw_generator_used_ne_is_symmetric": True,
            },
            "symmetric",
        ),
        (
            "ne",
            {
                "available": 2,
                "used": 2,
                "source": 2,
                "sink": 2,
                "requested_source": 3,
                "requested_sink": 2,
                "raw_generator_used_ne_is_symmetric": True,
            },
            "requested",
        ),
    ],
)
def test_legacy_spatial_contract_rejects_forged_metadata(field, replacement, message):
    raw, contract = _legacy_spatial_raw_contract()
    contract[field] = replacement
    with pytest.raises(ValueError, match=message):
        validate_legacy_spatial_current_raw(raw, contract)


def test_temporal_directed_raw_spin_bridge_preserves_links_weights_and_distinct_ne():
    raw, contract = _raw_contract()
    forward, backward = ConservedVectorCurrent(wilson_r=1.25).terms[6:8]

    def raw_resolver(term, **kwargs):
        return resolve_directed_current_raw(raw, contract, term, momentum=0, **kwargs)

    assembled = assemble_spin_aware_current(
        (forward, backward),
        raw_resolver,
        available_source_ne=2,
        available_sink_ne=2,
        used_source_ne=1,
        used_sink_ne=2,
        anchor_time=2,
        temporal_extent=3,
        boundary="periodic",
    )
    forward_spin = 1.25 * np.eye(4) - np.asarray(
        __import__("lattice.insertion.gamma", fromlist=["gamma"]).gamma(forward.gamma_index)
    )
    backward_spin = 1.25 * np.eye(4) + np.asarray(
        __import__("lattice.insertion.gamma", fromlist=["gamma"]).gamma(backward.gamma_index)
    )
    expected = (
        -0.5 * forward_spin[:, :, None, None] * raw["v2v"][6, 2, 0, :2, :1]
        + 0.5 * backward_spin[:, :, None, None] * raw["v2v"][7, 0, 0, :2, :1]
    )
    np.testing.assert_allclose(assembled["value"], expected, rtol=0, atol=0)
    assert assembled["value"].shape == (4, 4, 2, 1)
    assert assembled["terms"][0]["resolver_provenance"]["raw"]["direction"] == 6
    assert assembled["terms"][1]["resolver_provenance"]["raw"]["direction"] == 7
    assert assembled["terms"][1]["resolver_provenance"]["raw"]["time"] == 0
    assert spin_aware_current_adapter(assembled)["term_count"] == 2


def test_invalid_current_raw_inputs_are_rejected():
    with pytest.raises(ValueError, match="boundary"):
        _generator().calc_directed_current_raw("unbounded")
    generator = _generator()
    generator._current_U = generator._current_U[:3]
    with pytest.raises(ValueError, match="four link"):
        generator.calc_directed_current_raw()
    raw, contract = _raw_contract()
    term = ConservedVectorCurrent().terms[6]
    with pytest.raises(ValueError, match="outside"):
        resolve_directed_current_raw(
            raw,
            contract,
            term,
            endpoints={
                "bar_time": 3,
                "field_time": 1,
                "link_origin_time": 0,
                "temporal_point_split": True,
                "boundary": "periodic",
            },
            source_ne=1,
            sink_ne=1,
        )
    bad = raw["v2v"].copy()
    bad[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_current_raw_contract({"v2v": bad}, contract)
