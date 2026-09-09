import numpy as np
import pytest

from lattice import CurrentElementalGenerator, set_backend
from lattice.insertion.current import (
    CURRENT_API_VERSION,
    CURRENT_ASSEMBLER_SCHEMA,
    CURRENT_ELEMENTAL_SCHEMA,
    CURRENT_TERM_SCHEMA,
    ConservedVectorCurrent,
    CurrentTerm,
    LocalAxialCurrent,
    LocalVectorCurrent,
    PseudoScalarDensity,
    assemble_current_terms,
    lattice_divergence,
    legacy_current_vertex_adapter,
    spin_aware_current_adapter,
    resolve_current_term_endpoints,
    verify_pcac,
    verify_wt,
)
from lattice.insertion.gamma import gamma


def test_current_term_v1_compatibility_and_fixed_keys():
    legacy = CurrentTerm(2, 1, (0, 1, 0, 0), 2)
    mapping = legacy.as_dict()
    assert tuple(mapping) == (
        "schema",
        "coefficient",
        "direction",
        "displacement",
        "gamma_index",
        "link",
        "wilson_r",
        "spin_structure",
        "normalization",
        "bar_offset",
        "field_offset",
        "link_origin_offset",
        "link_dagger",
        "boundary_policy",
        "temporal_point_split",
    )
    assert mapping["schema"] == CURRENT_TERM_SCHEMA
    assert mapping["normalization"] == 1
    assert mapping["bar_offset"] == mapping["field_offset"] == (0, 0, 0, 0)
    assert mapping["link_origin_offset"] == (0, 0, 0, 0)
    assert mapping["link_dagger"] is False
    assert mapping["boundary_policy"] == "caller-supplied"
    assert mapping["temporal_point_split"] is False


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"coefficient": np.nan}, ValueError),
        ({"normalization": np.inf}, ValueError),
        ({"direction": 4}, ValueError),
        ({"direction": True}, TypeError),
        ({"displacement": (0, 0, 0)}, ValueError),
        ({"bar_offset": (0, 0, 0, 1)}, ValueError),
        ({"link": "sideways"}, ValueError),
        ({"link_dagger": 1}, TypeError),
        ({"boundary_policy": "periodic"}, ValueError),
        ({"temporal_point_split": True}, ValueError),
    ],
)
def test_current_term_rejects_invalid_schema_values(kwargs, error):
    values = {
        "coefficient": 1,
        "direction": 0,
        "displacement": (0, 0, 0, 0),
        "gamma_index": 1,
    }
    values.update(kwargs)
    with pytest.raises(error):
        CurrentTerm(**values)


def test_spatial_assembler_applies_only_coefficient_and_normalization():
    terms = (
        CurrentTerm(2, 0, (1, 0, 0, 0), 1, normalization=3),
        CurrentTerm(-0.5, 1, (0, 1, 0, 0), 2, normalization=4),
    )
    values = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[10.0, 20.0], [30.0, 40.0]]),
    )
    calls = []

    def resolver(term, **kwargs):
        calls.append((term, kwargs))
        index = len(calls) - 1
        return {
            "value": values[index],
            "source_ne": kwargs["source_ne"],
            "sink_ne": kwargs["sink_ne"],
            "provenance": {"term": index},
        }

    result = assemble_current_terms(
        terms,
        resolver,
        available_source_ne=2,
        available_sink_ne=2,
        used_source_ne=2,
        used_sink_ne=2,
        anchor_time=5,
    )
    np.testing.assert_array_equal(result["value"], 6 * values[0] - 2 * values[1])
    assert result["schema"] == CURRENT_ASSEMBLER_SCHEMA
    assert result["api_version"] == CURRENT_API_VERSION
    assert result["term_count"] == 2
    assert result["ne"] == {
        "source": {"available": 2, "used": 2},
        "sink": {"available": 2, "used": 2},
    }
    assert all(call[0]["schema"] == CURRENT_TERM_SCHEMA for call in calls)
    assert all(
        call[1]["endpoints"]
        == {
            "bar_time": 5,
            "field_time": 5,
            "link_origin_time": 5,
            "temporal_point_split": False,
            "boundary": "unbounded",
        }
        for call in calls
    )
    assert result["terms"] == (
        {
            "schema": CURRENT_TERM_SCHEMA,
            "endpoints": calls[0][1]["endpoints"],
            "resolver_provenance": {"term": 0},
        },
        {
            "schema": CURRENT_TERM_SCHEMA,
            "endpoints": calls[1][1]["endpoints"],
            "resolver_provenance": {"term": 1},
        },
    )


def test_assembler_preserves_different_source_sink_ne_and_provenance():
    def resolver(term, *, endpoints, source_ne, sink_ne):
        assert source_ne == 1
        assert sink_ne == 2
        return {
            "value": np.ones((sink_ne, source_ne)),
            "source_ne": source_ne,
            "sink_ne": sink_ne,
            "provenance": "external-elemental",
        }

    result = assemble_current_terms(
        LocalVectorCurrent().terms[:1],
        resolver,
        available_source_ne=3,
        available_sink_ne=4,
        used_source_ne=1,
        used_sink_ne=2,
    )
    assert result["value"].shape == (2, 1)
    assert result["ne"] == {
        "source": {"available": 3, "used": 1},
        "sink": {"available": 4, "used": 2},
    }
    assert result["terms"][0]["resolver_provenance"] == "external-elemental"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"available_source_ne": True}, TypeError),
        ({"available_sink_ne": 1.5}, TypeError),
        ({"available_source_ne": -1}, ValueError),
        ({"used_source_ne": -1}, ValueError),
        ({"used_sink_ne": True}, TypeError),
        ({"used_source_ne": 3}, ValueError),
        ({"used_sink_ne": 3}, ValueError),
    ],
)
def test_assembler_rejects_invalid_ne_bounds(kwargs, error):
    counts = {"available_source_ne": 2, "available_sink_ne": 2}
    counts.update(kwargs)
    with pytest.raises(error):
        assemble_current_terms(
            LocalVectorCurrent().terms[:1],
            lambda term, **call: {
                "value": np.ones(1),
                "source_ne": call["source_ne"],
                "sink_ne": call["sink_ne"],
            },
            **counts,
        )


def test_assembler_rejects_resolver_ne_mismatch():
    with pytest.raises(ValueError, match="exactly match"):
        assemble_current_terms(
            LocalVectorCurrent().terms[:1],
            lambda term, **call: {
                "value": np.ones(1),
                "source_ne": 2,
                "sink_ne": 1,
            },
            available_source_ne=2,
            available_sink_ne=2,
            used_source_ne=1,
            used_sink_ne=1,
        )


def test_temporal_endpoints_periodic_open_and_spatial():
    forward, backward = ConservedVectorCurrent().terms[6:8]
    assert resolve_current_term_endpoints(
        forward,
        anchor_time=7,
        temporal_extent=8,
        boundary="periodic",
    ) == {
        "bar_time": 7,
        "field_time": 0,
        "link_origin_time": 7,
        "temporal_point_split": True,
        "boundary": "periodic",
    }
    assert resolve_current_term_endpoints(
        backward.as_dict(),
        anchor_time=7,
        temporal_extent=8,
        boundary="periodic",
    ) == {
        "bar_time": 0,
        "field_time": 7,
        "link_origin_time": 7,
        "temporal_point_split": True,
        "boundary": "periodic",
    }
    for term in (forward, backward):
        with pytest.raises(IndexError, match="open temporal boundary"):
            resolve_current_term_endpoints(
                term,
                anchor_time=7,
                temporal_extent=8,
                boundary="open",
            )
    spatial = ConservedVectorCurrent().terms[0]
    assert (
        resolve_current_term_endpoints(
            spatial,
            anchor_time=7,
            temporal_extent=8,
            boundary="periodic",
        )["field_time"]
        == 7
    )


def test_legacy_vertex_adapter_rejects_point_split_endpoint_loss():
    term = ConservedVectorCurrent().terms[6]

    def resolver(term, **call):
        return {
            "value": np.ones((1, 1), dtype=np.complex128),
            "source_ne": call["source_ne"],
            "sink_ne": call["sink_ne"],
        }

    from lattice.insertion.current import assemble_spin_aware_current

    assembled = assemble_spin_aware_current(
        (term,),
        resolver,
        available_source_ne=1,
        available_sink_ne=1,
        anchor_time=0,
        temporal_extent=2,
        boundary="periodic",
    )
    adapter = spin_aware_current_adapter(assembled)
    with pytest.raises(ValueError, match="point-split"):
        legacy_current_vertex_adapter({0: adapter})

    degenerate = {
        **adapter,
        "terms": (
            {
                "endpoints": {
                    "bar_time": 0,
                    "field_time": 0,
                    "link_origin_time": 0,
                    "temporal_point_split": True,
                    "boundary": "periodic",
                }
            },
        ),
    }
    with pytest.raises(ValueError, match="point-split"):
        legacy_current_vertex_adapter({0: degenerate})

    for invalid_endpoints in (
        {},
        {
            "bar_time": None,
            "field_time": None,
            "link_origin_time": 0,
            "temporal_point_split": True,
            "boundary": "periodic",
        },
        {
            "bar_time": 0,
            "field_time": 0,
            "link_origin_time": 0,
            "temporal_point_split": False,
            "boundary": "periodic",
            "unknown": 1,
        },
    ):
        bad = {**adapter, "terms": ({"endpoints": invalid_endpoints},)}
        with pytest.raises(TypeError, match="endpoint"):
            legacy_current_vertex_adapter({0: bad})


def test_legacy_vertex_adapter_rejects_invalid_ne_provenance():
    term = LocalVectorCurrent().terms[0]

    def resolver(term, **call):
        return {
            "value": np.ones((1, 1), dtype=np.complex128),
            "source_ne": call["source_ne"],
            "sink_ne": call["sink_ne"],
        }

    from lattice.insertion.current import assemble_spin_aware_current

    assembled = assemble_spin_aware_current(
        (term,),
        resolver,
        available_source_ne=1,
        available_sink_ne=1,
    )
    adapter = spin_aware_current_adapter(assembled)
    for invalid in (
        True,
        1.5,
    ):
        bad = {**adapter, "ne": {**adapter["ne"]}}
        bad["ne"]["source"] = {"available": 1, "used": invalid}
        with pytest.raises(TypeError, match="Ne provenance"):
            legacy_current_vertex_adapter({0: bad})
    bad = {**adapter, "ne": {**adapter["ne"]}}
    bad["ne"]["source"] = {"available": 0, "used": 1}
    with pytest.raises(ValueError, match="Ne bounds"):
        legacy_current_vertex_adapter({0: bad})


def test_conserved_temporal_and_local_endpoint_schema():
    temporal_forward, temporal_backward = ConservedVectorCurrent().terms[6:8]
    assert temporal_forward.bar_offset == (0, 0, 0, 0)
    assert temporal_forward.field_offset == (0, 0, 0, 1)
    assert temporal_forward.link_origin_offset == (0, 0, 0, 0)
    assert temporal_forward.link_dagger is False
    assert temporal_forward.temporal_point_split is True
    assert temporal_backward.bar_offset == (0, 0, 0, 1)
    assert temporal_backward.field_offset == (0, 0, 0, 0)
    assert temporal_backward.link_origin_offset == (0, 0, 0, 0)
    assert temporal_backward.link_dagger is True
    assert temporal_backward.temporal_point_split is True
    for term in LocalVectorCurrent().terms + LocalAxialCurrent().terms + PseudoScalarDensity().terms:
        assert term.bar_offset == term.field_offset == (0, 0, 0, 0)
        assert term.link == "none"
        assert term.temporal_point_split is False


def test_endpoint_and_assembler_contract_rejections():
    term = LocalVectorCurrent().terms[0]
    for boundary in ("invalid", None):
        with pytest.raises(ValueError, match="boundary"):
            resolve_current_term_endpoints(term, anchor_time=0, boundary=boundary)
    for boundary in ("periodic", "open"):
        for extent in (None, 0, True, 1.5):
            with pytest.raises((TypeError, ValueError), match="temporal_extent"):
                resolve_current_term_endpoints(
                    term,
                    anchor_time=0,
                    temporal_extent=extent,
                    boundary=boundary,
                )
    with pytest.raises(ValueError, match="schema"):
        resolve_current_term_endpoints(
            {**term.as_dict(), "schema": "wrong"},
            anchor_time=0,
            boundary="unbounded",
        )
    schema_less = dict(term.as_dict())
    schema_less.pop("schema")
    with pytest.raises(ValueError, match="schema"):
        resolve_current_term_endpoints(
            schema_less,
            anchor_time=0,
            boundary="unbounded",
        )
    assert (
        resolve_current_term_endpoints(
            term,
            anchor_time=2,
            temporal_extent="ignored",
            boundary="unbounded",
        )["bar_time"]
        == 2
    )
    with pytest.raises(ValueError, match="non-empty"):
        assemble_current_terms(
            [],
            lambda term, **kwargs: {},
            available_source_ne=0,
            available_sink_ne=0,
        )
    with pytest.raises(TypeError, match="callable"):
        assemble_current_terms(
            [term],
            None,
            available_source_ne=0,
            available_sink_ne=0,
        )


@pytest.mark.parametrize(
    "resolver, error, message",
    [
        (lambda term, **kwargs: np.ones(1), TypeError, "mapping"),
        (lambda term, **kwargs: {"value": np.ones(1)}, TypeError, "missing"),
        (
            lambda term, **kwargs: {
                "value": np.array([]),
                "source_ne": 1,
                "sink_ne": 1,
            },
            ValueError,
            "non-empty",
        ),
        (
            lambda term, **kwargs: {
                "value": np.array([np.inf]),
                "source_ne": 1,
                "sink_ne": 1,
            },
            ValueError,
            "finite",
        ),
        (
            lambda term, **kwargs: {
                "value": np.array(["x"]),
                "source_ne": 1,
                "sink_ne": 1,
            },
            TypeError,
            "numeric",
        ),
    ],
)
def test_assembler_rejects_invalid_resolver_results(resolver, error, message):
    with pytest.raises(error, match=message):
        assemble_current_terms(
            LocalVectorCurrent().terms[:1],
            resolver,
            available_source_ne=1,
            available_sink_ne=1,
        )


def test_assembler_rejects_mismatched_term_shapes():
    values = iter((np.ones((1, 2)), np.ones((2, 1))))

    def resolver(term, **kwargs):
        return {
            "value": next(values),
            "source_ne": kwargs["source_ne"],
            "sink_ne": kwargs["sink_ne"],
        }

    with pytest.raises(ValueError, match="does not match"):
        assemble_current_terms(
            LocalVectorCurrent().terms[:2],
            resolver,
            available_source_ne=1,
            available_sink_ne=1,
        )


def test_raw_adapter_records_requested_ne_without_slicing():
    class Generator:
        Ne = usedNe = 2
        Np = usedNp = 1

        def calc_all(self, t):
            return {
                "v2v": np.ones((1, 1, 2, 2)),
                "v2p": np.ones((1, 2, 1, 3)),
                "p2v": np.ones((1, 1, 3, 2)),
                "p2p": [{}],
            }

    result = LocalVectorCurrent().compute_elemental(
        Generator(),
        t=0,
        used_source_ne=1,
        used_sink_ne=2,
    )
    assert result["elemental"].shape[-2:] == (2, 2)
    assert result["contract"]["ne"] == {
        "available": 2,
        "used": 2,
        "source": 2,
        "sink": 2,
        "requested_source": 1,
        "requested_sink": 2,
        "raw_generator_used_ne_is_symmetric": True,
    }

    Generator.usedNe = 1
    with pytest.raises(ValueError, match="used_source_ne.*available_source_ne"):
        LocalVectorCurrent().compute_elemental(
            Generator(),
            t=0,
            used_source_ne=2,
        )


def test_compute_elemental_with_numpy_current_elemental_generator():
    set_backend("numpy")

    class Loader:
        Ne = 1
        Np = 1

    loader = Loader()
    generator = CurrentElementalGenerator(
        latt_size=[1, 1, 1, 1],
        gauge_field=loader,
        eigenvector=loader,
        point=loader,
        num_nabla=0,
        momentum_list=[(0, 0, 0)],
        usedNe=1,
        usedNp=1,
    )
    generator._U = np.broadcast_to(np.eye(3, dtype=np.complex128), (3, 1, 1, 1, 1, 3, 3)).copy()
    generator._eigenvector_data = np.ones((1, 1, 1, 1, 1, 3), dtype=np.complex128)
    generator._point_data = np.zeros((1, 1, 3), dtype=int)

    result = LocalVectorCurrent().compute_elemental(generator, t=0)
    all_elementals = result["all_elementals"]

    assert result["elemental"].shape == (1, 1, 1, 1)
    assert set(all_elementals) == {"v2v", "v2p", "p2v", "p2p"}
    assert all_elementals["v2p"].shape == (1, 1, 1, 3)
    assert all_elementals["p2v"].shape == (1, 1, 3, 1)
    assert len(all_elementals["p2p"]) == 1
    assert result["api_version"] == CURRENT_API_VERSION == "1.2.0"
    assert result["schema"] == CURRENT_ELEMENTAL_SCHEMA
    assert result["contract"] == {
        "schema": CURRENT_ELEMENTAL_SCHEMA,
        "representation": "raw-spatial-displacement-basis",
        "combined_with_current_terms": False,
        "supports_temporal_point_split": False,
        "term_application": "external-resolver-required",
        "term_schema": CURRENT_TERM_SCHEMA,
        "assembler_schema": CURRENT_ASSEMBLER_SCHEMA,
        "ne": {
            "available": 1,
            "used": 1,
            "source": 1,
            "sink": 1,
            "requested_source": 1,
            "requested_sink": 1,
            "raw_generator_used_ne_is_symmetric": True,
        },
        "np": {"available": 1, "used": 1},
        "shapes": {
            "v2v": (1, 1, 1, 1),
            "v2p": (1, 1, 1, 3),
            "p2v": (1, 1, 3, 1),
            "p2p": (1,),
        },
        "axes": {
            "v2v": ("displacement", "momentum", "sink_ne", "source_ne"),
            "v2p": ("displacement", "sink_ne", "point", "color"),
            "p2v": ("displacement", "point", "color", "source_ne"),
            "p2p": "sparse-per-displacement",
        },
    }
    assert all(term["schema"] == CURRENT_TERM_SCHEMA for term in result["terms"])


def test_current_elemental_generator_ne_bounds_and_provenance():
    set_backend("numpy")

    class Loader:
        Ne = 2
        Np = 1

    def build(used_ne):
        generator = CurrentElementalGenerator(
            latt_size=[1, 1, 1, 1],
            gauge_field=Loader(),
            eigenvector=Loader(),
            point=Loader(),
            num_nabla=0,
            momentum_list=[(0, 0, 0)],
            usedNe=used_ne,
            usedNp=1,
        )
        generator._U = np.broadcast_to(np.eye(3, dtype=np.complex128), (3, 1, 1, 1, 1, 3, 3)).copy()
        generator._eigenvector_data = np.arange(6, dtype=np.complex128).reshape(1, 2, 1, 1, 1, 3)
        generator._point_data = np.zeros((1, 1, 3), dtype=int)
        return generator

    results = {}
    for used_ne in (0, 1, 2):
        result = LocalVectorCurrent().compute_elemental(build(used_ne), t=0)
        results[used_ne] = result
        assert result["contract"]["ne"] == {
            "available": 2,
            "used": used_ne,
            "source": used_ne,
            "sink": used_ne,
            "requested_source": used_ne,
            "requested_sink": used_ne,
            "raw_generator_used_ne_is_symmetric": True,
        }
        assert result["contract"]["shapes"]["v2v"][-2:] == (used_ne, used_ne)

    np.testing.assert_allclose(
        results[1]["all_elementals"]["v2v"],
        results[2]["all_elementals"]["v2v"][..., :1, :1],
    )
    np.testing.assert_allclose(
        results[1]["all_elementals"]["v2p"],
        results[2]["all_elementals"]["v2p"][..., :1, :, :],
    )
    np.testing.assert_allclose(
        results[1]["all_elementals"]["p2v"],
        results[2]["all_elementals"]["p2v"][..., :, :, :1],
    )
    for smaller, larger in zip(
        results[1]["all_elementals"]["p2p"],
        results[2]["all_elementals"]["p2p"],
    ):
        assert smaller.keys() == larger.keys()
        for key, value in smaller.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(value, larger[key])
            else:
                assert value == larger[key]

    for invalid in (-1, 3):
        with pytest.raises(ValueError, match="0 <= usedNe <= available usedNe"):
            build(invalid)
    for invalid in (True, 1.5, "1"):
        with pytest.raises(TypeError, match="usedNe must be an integer"):
            build(invalid)


def test_compute_elemental_strictly_adapts_calc_all_mapping():
    class Generator:
        Ne = usedNe = 1
        Np = usedNp = 1

        def __init__(self):
            self.times = []

        def calc_all(self, t):
            self.times.append(t)
            return {
                "v2v": np.full((1, 1, 1, 1), t),
                "v2p": np.full((1, 1, 1, 3), 2),
                "p2v": np.full((1, 1, 3, 1), 3),
                "p2p": [{"value": 4}],
            }

    generator = Generator()
    result = LocalVectorCurrent().compute_elemental(generator, t=7)
    assert generator.times == [7]
    assert result["time"] == 7
    np.testing.assert_array_equal(result["elemental"], np.full((1, 1, 1, 1), 7))
    assert result["all_elementals"]["p2p"] == [{"value": 4}]
    assert len(result["terms"]) == 4


@pytest.mark.parametrize(
    "generator, message",
    [
        (object(), "calc_all"),
        (
            type(
                "NonMapping",
                (),
                {
                    "Ne": 1,
                    "usedNe": 1,
                    "Np": 1,
                    "usedNp": 1,
                    "calc_all": lambda self, t: [t],
                },
            )(),
            "mapping",
        ),
        (
            type(
                "MissingKey",
                (),
                {
                    "Ne": 1,
                    "usedNe": 1,
                    "Np": 1,
                    "usedNp": 1,
                    "calc_all": lambda self, t: {"p2p": t},
                },
            )(),
            "v2v",
        ),
    ],
)
def test_compute_elemental_rejects_invalid_generator_results(generator, message):
    with pytest.raises(TypeError, match=message):
        LocalVectorCurrent().compute_elemental(generator, t=0)


def test_local_currents_are_local_bilinears():
    expected = (
        (LocalVectorCurrent(), (1, 2, 4, 8)),
        (LocalAxialCurrent(), (14, 13, 11, 7)),
    )
    for current, gamma_indices in expected:
        terms = current.terms
        assert len(terms) == 4
        assert tuple(term.gamma_index for term in terms) == gamma_indices
        assert tuple(term.direction for term in terms) == (0, 1, 2, 3)
        assert all(term.displacement == (0, 0, 0, 0) for term in terms)
        assert all(term.link == "none" for term in terms)


def test_local_axial_terms_match_gamma_mu_gamma5_matrices():
    set_backend("numpy")
    terms = LocalAxialCurrent().terms
    assert tuple(term.coefficient for term in terms) == (1, -1, 1, -1)
    for mu, (term, vector_index) in enumerate(zip(terms, (1, 2, 4, 8))):
        np.testing.assert_allclose(
            term.coefficient * np.asarray(gamma(term.gamma_index)),
            np.asarray(gamma(vector_index)) @ np.asarray(gamma(15)),
        )
        assert term.spin_structure == f"gamma_{mu}gamma_5"


def test_conserved_current_has_wilson_midpoint_terms():
    terms = ConservedVectorCurrent(wilson_r=1.25).terms
    assert len(terms) == 8
    for mu, gamma_index in enumerate((1, 2, 4, 8)):
        forward, backward = terms[2 * mu : 2 * mu + 2]
        positive = tuple(1 if axis == mu else 0 for axis in range(4))
        negative = tuple(-1 if axis == mu else 0 for axis in range(4))
        assert (forward.direction, backward.direction) == (mu, mu)
        assert (forward.gamma_index, backward.gamma_index) == (gamma_index, gamma_index)
        assert (forward.coefficient, backward.coefficient) == (-0.5, 0.5)
        assert (forward.displacement, backward.displacement) == (positive, negative)
        assert (forward.link, backward.link) == ("forward", "backward")
        assert (forward.spin_structure, backward.spin_structure) == (
            f"r-gamma_{mu}",
            f"r+gamma_{mu}",
        )
        assert forward.wilson_r == backward.wilson_r == 1.25


def test_conserved_requires_and_verifies_z_and_wt():
    conserved = ConservedVectorCurrent()
    result = conserved.verify(z=1, divergence=np.zeros(2))
    assert result.keys() == {"Z", "WT"}
    assert result["Z"]["passed"]
    assert result["WT"]["passed"]

    with pytest.raises(ValueError, match="renormalization factor"):
        conserved.verify(divergence=np.zeros(2))
    with pytest.raises(ValueError, match="divergence or current"):
        conserved.verify(z=1)


def test_local_vector_does_not_claim_exact_z_or_wt():
    local = LocalVectorCurrent()
    assert local.verify() == {}
    assert local.verify(z=1, divergence=np.zeros(2)) == {}


def test_axial_and_density_verify_only_pcac():
    inputs = {
        "axial_divergence": np.ones(2),
        "pseudoscalar": np.ones(2),
        "mass": 0.5,
    }
    for operator in (LocalAxialCurrent(), PseudoScalarDensity()):
        result = operator.verify(z=7, divergence=np.ones(2), **inputs)
        assert result.keys() == {"PCAC"}
        assert result["PCAC"]["passed"]

    with pytest.raises(ValueError, match="axial_divergence"):
        LocalAxialCurrent().verify()
    with pytest.raises(ValueError, match="pseudoscalar"):
        PseudoScalarDensity().verify(axial_divergence=np.zeros(2), mass=1)
    with pytest.raises(ValueError, match="mass"):
        LocalAxialCurrent().verify(
            axial_divergence=np.zeros(2),
            pseudoscalar=np.zeros(2),
        )


def test_array_level_wt_and_pcac_results_are_not_false_positives():
    current = np.zeros((4, 2, 2, 2, 2), dtype=np.complex128)
    assert verify_wt(current=current)["passed"]
    current[0, 0, 0, 0, 0] = 1
    assert not verify_wt(current=current)["passed"]

    assert verify_pcac(
        axial_divergence=np.ones(2),
        pseudoscalar=np.ones(2),
        mass=0.5,
    )["passed"]
    assert not verify_pcac(
        axial_divergence=np.ones(2),
        pseudoscalar=np.zeros(2),
        mass=0.5,
    )["passed"]


def test_verification_rejects_empty_and_nonfinite_arrays():
    invalid_values = (
        np.array([]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1 + np.nan * 1j]),
        np.array([1 + np.inf * 1j]),
    )
    for value in invalid_values:
        with pytest.raises(ValueError, match="non-empty|finite"):
            verify_wt(divergence=value)

    valid = {
        "axial_divergence": np.zeros(2),
        "pseudoscalar": np.zeros(2),
        "mass": 1.0,
        "improvement_residual": np.zeros(2),
    }
    for name in valid:
        kwargs = valid.copy()
        kwargs[name] = np.array([]) if name == "mass" else np.array([np.inf])
        with pytest.raises(ValueError, match="non-empty|finite"):
            verify_pcac(**kwargs)


def test_tolerances_are_finite_nonnegative_scalars():
    invalid_tolerances = (-1, np.nan, np.inf, [1e-10], 1 + 0j)
    for value in invalid_tolerances:
        with pytest.raises(ValueError, match="finite real scalar|non-negative"):
            verify_wt(divergence=np.zeros(1), atol=value)
        with pytest.raises(ValueError, match="finite real scalar|non-negative"):
            verify_wt(divergence=np.zeros(1), rtol=value)


def test_zero_target_relative_tolerance_does_not_hide_residuals():
    assert not verify_wt(
        divergence=np.array([1e-6]),
        atol=0,
        rtol=1e6,
    )["passed"]
    assert not ConservedVectorCurrent.verify_z(
        np.array([1 + 1e-6]),
        atol=0,
        rtol=1e6,
    )["passed"]
    assert not verify_pcac(
        axial_divergence=np.array([1e-6]),
        pseudoscalar=np.zeros(1),
        mass=1,
        atol=0,
        rtol=1e6,
    )["passed"]


def test_pcac_distinguishes_divergence_from_current_and_supports_improvement():
    four_component_divergence = np.ones((4, 1, 1, 1, 1))
    result = verify_pcac(
        axial_divergence=four_component_divergence,
        pseudoscalar=four_component_divergence,
        mass=0.5,
    )
    assert result["passed"]
    assert result["lhs"].shape == four_component_divergence.shape

    improved = verify_pcac(
        axial_divergence=np.array([3.0, 5.0]),
        pseudoscalar=np.array([1.0, 2.0]),
        mass=1,
        improvement_residual=np.ones(2),
    )
    assert improved["passed"]
    assert improved["condition"] == "div A = 2 m P + E"

    axial_current = np.zeros((4, 2, 1, 1, 1))
    axial_current[0, 1, 0, 0, 0] = 2
    from_current = verify_pcac(
        axial_current=axial_current,
        pseudoscalar=np.array([0.0, 0.5]).reshape(2, 1, 1, 1),
        mass=1,
        periodic=False,
        lattice_spacing=2,
    )
    assert from_current["passed"]

    with pytest.raises(ValueError, match="exactly one"):
        verify_pcac(
            axial_divergence=np.zeros(1),
            axial_current=axial_current,
            pseudoscalar=np.zeros(1),
            mass=1,
        )
    with pytest.raises(ValueError, match="incompatible shapes"):
        verify_pcac(
            axial_divergence=np.zeros(2),
            pseudoscalar=np.zeros(3),
            mass=1,
        )
    with pytest.raises(ValueError, match="improvement_residual.*incompatible shape"):
        verify_pcac(
            axial_divergence=np.zeros(2),
            pseudoscalar=np.zeros(2),
            mass=1,
            improvement_residual=np.zeros(3),
        )


def test_divergence_periodic_nonperiodic_and_lattice_spacing():
    current = np.zeros((4, 2, 1, 1, 1))
    current[0, 1, 0, 0, 0] = 1

    periodic = lattice_divergence(current, periodic=True)
    nonperiodic = lattice_divergence(current, periodic=False)
    scaled = lattice_divergence(current, periodic=True, lattice_spacing=0.5)
    np.testing.assert_array_equal(periodic[:, 0, 0, 0], [-1, 1])
    np.testing.assert_array_equal(nonperiodic[:, 0, 0, 0], [0, 1])
    np.testing.assert_array_equal(scaled[:, 0, 0, 0], [-2, 2])

    wt = verify_wt(
        current=current,
        periodic=False,
        lattice_spacing=2,
    )
    np.testing.assert_array_equal(wt["array"][:, 0, 0, 0], [0, 0.5])


def test_divergence_rejects_duplicate_axes_and_invalid_spacing():
    current = np.zeros((4, 2, 2, 2, 2))
    with pytest.raises(ValueError, match="unique"):
        lattice_divergence(current, site_axes=(1, 1, 3, 4))
    for spacing in (0, -1, np.nan, np.inf, [1]):
        with pytest.raises(ValueError, match="lattice_spacing"):
            lattice_divergence(current, lattice_spacing=spacing)


def test_current_verify_forwards_pcac_boundary_and_spacing_options():
    axial_current = np.zeros((4, 2, 1, 1, 1))
    axial_current[0, 1, 0, 0, 0] = 2
    operator = LocalAxialCurrent(lattice_spacing=2)
    result = operator.verify(
        axial_current=axial_current,
        pseudoscalar=np.array([0.0, 0.5]).reshape(2, 1, 1, 1),
        mass=1,
        periodic=False,
    )
    assert result["PCAC"]["passed"]

    overridden = operator.verify(
        axial_current=axial_current,
        pseudoscalar=np.array([0.0, 1.0]).reshape(2, 1, 1, 1),
        mass=1,
        periodic=False,
        lattice_spacing=1,
    )
    assert overridden["PCAC"]["passed"]
