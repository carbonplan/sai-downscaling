"""Tests for the store metadata rules in :mod:`saidownscale.store_metadata`.

The planning rules decide what gets written onto published data, so they are tested directly. One
test at the end drives a real icechunk store, because "writes attrs without rewriting data" is the
claim the whole script rests on.
"""

from pathlib import Path

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from saidownscale.store_metadata import (
    gcm_from_store,
    plan_group,
    product_from_path,
    scenario_from_path,
    source_for,
)

OUTPUT_ATTRS = {
    "srm_downscaling:gcm": "CESM2-WACCM6",
    "srm_downscaling:downscaling_method": "BCSD",
    "srm_downscaling:observation_dataset": "ERA5",
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("bcsd/g6_1p5k/tas/001", "g6_1p5k"),
        ("bcsd/debiased_coarse/g6_1p5k/tas/001", "g6_1p5k"),
        ("qdmsd/g6_1p5k_end/pr/002", "g6_1p5k_end"),
        ("bcsd/historical/tas/r1i1p1f1", "historical"),
        ("ssp245", "ssp245"),  # input store: the scenario is the whole path
        ("bcsd/tas/001", None),
    ],
)
def test_scenario_from_path(path: str, expected: str | None) -> None:
    """The scenario is found by name, so output and input nesting both resolve."""
    assert scenario_from_path(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [("bcsd/ssp245/tas/001", "output"), ("qdmsd/ssp245/tas/001", "output"), ("ssp245", "input")],
)
def test_product_from_path(path: str, expected: str) -> None:
    assert product_from_path(path) == expected


def test_plan_sets_missing_attrs() -> None:
    plan = plan_group("bcsd/g6_1p5k/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert plan.to_set["license"] == "CC-BY-4.0"
    assert "Walker Lee" in plan.to_set["references"]
    assert plan.writes


def test_plan_leaves_matching_attrs_alone() -> None:
    existing = {**OUTPUT_ATTRS, "institution": "CarbonPlan"}
    plan = plan_group("bcsd/g6_1p5k/tas/001", existing, "CESM2-WACCM6", "output")
    assert "institution" not in plan.to_set
    assert plan.unchanged["institution"] == "CarbonPlan"


def test_plan_reports_a_differing_attr_instead_of_clobbering() -> None:
    existing = {**OUTPUT_ATTRS, "license": "CC0-1.0"}
    plan = plan_group("bcsd/g6_1p5k/tas/001", existing, "CESM2-WACCM6", "output")
    assert plan.conflicts["license"] == ("CC0-1.0", "CC-BY-4.0")
    assert "license" not in plan.to_set


def test_input_groups_keep_their_own_source() -> None:
    """``source`` on an input group describes the model, so nothing may compose one over it."""
    plan = plan_group("ssp245", {"source": "UM"}, "UKESM1-1-LL", "input")
    assert "source" not in plan.to_set
    assert "source" not in plan.conflicts


def test_input_groups_are_not_attributed_to_carbonplan() -> None:
    """ACDD ``institution`` names who produced the data, and we did not produce the input.

    Several input groups already record their own, such as the National Center for Atmospheric
    Research on CESM historical, which we must not overwrite.
    """
    for scenario in ("historical", "ssp245", "g6_1p5k"):
        plan = plan_group(scenario, {}, "UKESM1-1-LL", "input")
        assert "institution" not in plan.to_set


def test_output_groups_are_attributed_to_carbonplan() -> None:
    """We did produce the downscaled product, so there ``institution`` is ours."""
    plan = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert plan.to_set["institution"] == "CarbonPlan"


def test_ukesm_licenses_differ_within_one_store() -> None:
    """A store-level license attr would be wrong here, which is why it resolves per group."""
    hist = plan_group("historical", {}, "UKESM1-1-LL", "input")
    ssp = plan_group("ssp245", {}, "UKESM1-1-LL", "input")
    assert hist.to_set["license"] == "CC-BY-4.0"
    assert ssp.to_set["license"] == "OGL-UK-3.0"


def test_input_with_no_asserted_license_gets_no_license_attr() -> None:
    """A blank license in the table is a decision not to grant one, so nothing may invent it."""
    plan = plan_group("g6_1p5k", {}, "CESM2-WACCM6", "input")
    assert "license" not in plan.to_set
    assert "license_url" not in plan.to_set
    assert "Walker Lee" in plan.to_set["references"]


def test_output_is_still_licensed_when_its_input_is_not() -> None:
    """Our downscaled product carries its own license regardless of the upstream decision."""
    plan = plan_group("bcsd/g6_1p5k/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert plan.to_set["license"] == "CC-BY-4.0"


def test_path_and_attrs_disagreeing_raises() -> None:
    existing = {**OUTPUT_ATTRS, "srm_downscaling:scenario": "SSP245"}
    with pytest.raises(ValueError, match="Refusing to guess"):
        plan_group("bcsd/g6_1p5k/tas/001", existing, "CESM2-WACCM6", "output")


def test_unattributed_pair_raises_rather_than_defaulting() -> None:
    with pytest.raises(KeyError, match="no license or citation recorded"):
        plan_group("bcsd/esgf_ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")


def test_source_needs_real_provenance() -> None:
    """Without the provenance to describe itself, a group gets no composed ``source``."""
    assert source_for({}, "CESM2-WACCM6") is None
    assert "ERA5" in source_for(OUTPUT_ATTRS, "CESM2-WACCM6")


def test_gcm_inferred_from_either_store_layout() -> None:
    assert gcm_from_store("s3://b/input/processed/CESM2-WACCM6.icechunk") == "CESM2-WACCM6"
    assert gcm_from_store("s3://b/output/UKESM1-1-LL-ERA5-global.icechunk") == "UKESM1-1-LL"
    with pytest.raises(ValueError, match="cannot tell which GCM"):
        gcm_from_store("s3://b/output/MIROC-ES2H-ERA5-global.icechunk")


def _seed_store(path: Path) -> icechunk.Repository:
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(path=str(path)))
    session = repo.writable_session("main")
    ds = xr.Dataset({"tas": (("time",), np.arange(8, dtype="float32"))})
    ds.attrs = {"Conventions": "CF-1.8", **OUTPUT_ATTRS, "srm_downscaling:scenario": "G6-1.5K"}
    ds.to_zarr(session.store, group="bcsd/g6_1p5k/tas/001", mode="w", consolidated=False)
    session.commit("seed")
    return repo


def test_writing_attrs_adds_no_chunks_and_keeps_the_data(tmp_path: Path) -> None:
    """The load-bearing claim: attrs land without a chunk or manifest being rewritten."""
    store_path = tmp_path / "CESM2-WACCM6-ERA5-global.icechunk"
    repo = _seed_store(store_path)

    def census() -> dict[str, int]:
        return {
            kind: len(list((store_path / kind).glob("**/*")))
            for kind in ("chunks", "manifests")
            if (store_path / kind).exists()
        }

    before = census()
    plan = plan_group("bcsd/g6_1p5k/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    session = repo.writable_session("main")
    zarr.open_group(session.store, path="bcsd/g6_1p5k/tas/001", mode="a").attrs.update(plan.to_set)
    session.commit("add attrs")

    assert census() == before, "writing attrs must not add chunks or manifests"

    result = xr.open_zarr(
        repo.readonly_session(branch="main").store,
        group="bcsd/g6_1p5k/tas/001",
        consolidated=False,
    )
    assert np.array_equal(result.tas.values, np.arange(8, dtype="float32"))
    assert result.attrs["license"] == "CC-BY-4.0"
    assert result.attrs["Conventions"] == "CF-1.8", "existing attrs must survive"
    assert result.attrs["srm_downscaling:gcm"] == "CESM2-WACCM6"
