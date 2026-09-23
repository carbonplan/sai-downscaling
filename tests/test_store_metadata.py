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

from saidownscale import licenses
from saidownscale.config import read_attr
from saidownscale.store_metadata import (
    gcm_from_store,
    plan_group,
    product_from_path,
    repair_history,
    scenario_from_path,
    source_for,
)

OUTPUT_ATTRS = {
    "sai_downscaling:gcm": "CESM2-WACCM6",
    "sai_downscaling:downscaling_method": "BCSD",
    "sai_downscaling:observation_dataset": "ERA5",
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
    assert plan.to_set["license"] == "CC BY 4.0"
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
    assert plan.conflicts["license"] == ("CC0-1.0", "CC BY 4.0")
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
    assert hist.to_set["license"] == "CC BY 4.0"
    assert ssp.to_set["license"] == "OGLv3"


def test_input_with_no_asserted_license_gets_no_license_attr() -> None:
    """A blank license in the table is a decision not to grant one, so nothing may invent it."""
    plan = plan_group("g6_1p5k", {}, "CESM2-WACCM6", "input")
    assert "license" not in plan.to_set
    assert "license_url" not in plan.to_set
    assert "Walker Lee" in plan.to_set["references"]


def test_output_is_still_licensed_when_its_input_is_not() -> None:
    """Our downscaled product carries its own license regardless of the upstream decision."""
    plan = plan_group("bcsd/g6_1p5k/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert plan.to_set["license"] == "CC BY 4.0"


def test_path_and_attrs_disagreeing_raises() -> None:
    existing = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "SSP245"}
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
    ds.attrs = {"Conventions": "CF-1.8", **OUTPUT_ATTRS, "sai_downscaling:scenario": "G6-1.5K"}
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
    assert result.attrs["license"] == "CC BY 4.0"
    assert result.attrs["Conventions"] == "CF-1.8", "existing attrs must survive"
    assert result.attrs["sai_downscaling:gcm"] == "CESM2-WACCM6"


def test_output_carries_a_contact_and_input_does_not() -> None:
    """``contact`` points at us, so it belongs only on data we produced."""
    out = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert out.to_set["contact"] == "hello@carbonplan.org"
    assert "contact" not in plan_group("ssp245", {}, "UKESM1-1-LL", "input").to_set


def test_license_matches_the_spelling_the_docs_table_uses() -> None:
    """The published data and the published table must not name the same license two ways."""
    plan = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert plan.to_set["license"] == "CC BY 4.0"
    assert plan.to_set["license_url"] == "https://creativecommons.org/licenses/by/4.0/"


WRONG_HISTORY = "2026-09-09T23:05:14Z: BCSD downscaling by srm v1.0.0"


def test_history_naming_the_wrong_method_is_repaired() -> None:
    """Groups written before the fix claim BCSD whatever ran, contradicting their own attrs."""
    existing = {**OUTPUT_ATTRS, "sai_downscaling:downscaling_method": "QDMSD"}
    corrected = repair_history({**existing, "history": WRONG_HISTORY})
    assert corrected == "2026-09-09T23:05:14Z: QDMSD downscaling by srm v1.0.0"


def test_repair_preserves_the_timestamp_and_the_producer() -> None:
    """Only the method token is wrong. When and what produced the data is the true record.

    ``srm`` really was the package name when v1.0.0 was written, so rewriting it would replace
    history rather than correct it.
    """
    corrected = repair_history(
        {"history": WRONG_HISTORY, "sai_downscaling:downscaling_method": "QDMSD"}
    )
    assert corrected.startswith("2026-09-09T23:05:14Z: ")
    assert corrected.endswith(" downscaling by srm v1.0.0")


def test_correct_history_is_left_alone() -> None:
    assert repair_history({**OUTPUT_ATTRS, "history": WRONG_HISTORY}) is None


@pytest.mark.parametrize(
    "existing",
    [
        {"sai_downscaling:downscaling_method": "QDMSD"},  # no history to repair
        {"history": WRONG_HISTORY},  # no method to repair it against
        {"history": "hand written note", "sai_downscaling:downscaling_method": "QDMSD"},
    ],
    ids=["no-history", "no-method", "unparseable"],
)
def test_repair_declines_without_enough_information(existing: dict[str, str]) -> None:
    """Rather than guess at a history it cannot parse, the repair returns nothing."""
    assert repair_history(existing) is None


def test_repairs_are_kept_out_of_to_set_so_they_need_an_explicit_flag() -> None:
    """A repair rewrites published data, so it must not ride along with the additive writes."""
    existing = {**OUTPUT_ATTRS, "sai_downscaling:downscaling_method": "QDMSD"}
    plan = plan_group(
        "qdmsd/ssp245/tas/001", {**existing, "history": WRONG_HISTORY}, "CESM2-WACCM6", "output"
    )
    assert "history" in plan.repairs
    assert "history" not in plan.to_set
    assert "history" not in plan.conflicts


def test_deprecated_attrs_are_planned_for_removal() -> None:
    """``config_hash`` is no longer written, so published groups still carrying it can shed it."""
    existing = {**OUTPUT_ATTRS, "sai_downscaling:config_hash": "1cc537e7c038"}
    plan = plan_group("bcsd/ssp245/tas/001", existing, "CESM2-WACCM6", "output")
    assert plan.removals == {"sai_downscaling:config_hash": "1cc537e7c038"}


def test_removals_are_kept_out_of_every_other_category() -> None:
    """Deleting cannot be undone by re-running, so it must need its own explicit flag."""
    existing = {**OUTPUT_ATTRS, "sai_downscaling:config_hash": "1cc537e7c038"}
    plan = plan_group("bcsd/ssp245/tas/001", existing, "CESM2-WACCM6", "output")
    for key in plan.removals:
        assert key not in plan.to_set
        assert key not in plan.conflicts
        assert key not in plan.repairs


def test_config_json_is_not_deprecated() -> None:
    """The full config stays: it is what a cache-mismatch check reads, and it is not derivable."""
    existing = {**OUTPUT_ATTRS, "sai_downscaling:config_json": "{}"}
    assert plan_group("bcsd/ssp245/tas/001", existing, "CESM2-WACCM6", "output").removals == {}


def test_doi_and_terms_are_omitted_until_they_have_real_values() -> None:
    """A placeholder in published metadata is worse than an absent attribute."""
    plan = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert "doi" not in plan.to_set
    assert "terms_of_data_access" not in plan.to_set


def test_doi_and_terms_appear_once_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the constants is all it takes; the script needs no further change."""
    monkeypatch.setattr(licenses, "DOI", "10.5281/zenodo.123456")
    monkeypatch.setattr(licenses, "TERMS_OF_DATA_ACCESS", "https://example.org/toda")
    attrs = licenses.metadata_attrs("CESM2-WACCM6", "ssp245", product="output")
    assert attrs["doi"] == "10.5281/zenodo.123456"
    assert attrs["terms_of_data_access"] == "https://example.org/toda"


LEGACY_ONLY = {
    "srm_downscaling:gcm": "CESM2-WACCM6",
    "srm_downscaling:downscaling_method": "QDMSD",
    "srm_downscaling:observation_dataset": "ERA5",
    "srm_downscaling:scenario": "SSP245",
}


def test_readers_accept_the_legacy_namespace() -> None:
    """v1.0.0 is fixed in place, not regenerated, so its provenance must still be readable.

    A reader that saw only the current namespace would find nothing on v1.0.0 and skip the checks
    that depend on provenance rather than fail, which is the more dangerous outcome.
    """
    assert read_attr(LEGACY_ONLY, "gcm") == "CESM2-WACCM6"
    assert source_for(LEGACY_ONLY, "CESM2-WACCM6").endswith("by the QDMSD method")


def test_current_namespace_wins_when_both_are_present() -> None:
    """Mid-migration a group carries both, and the copy we wrote is the authoritative one."""
    both = {"srm_downscaling:gcm": "old", "sai_downscaling:gcm": "new"}
    assert read_attr(both, "gcm") == "new"


def test_legacy_provenance_is_copied_to_the_current_namespace() -> None:
    plan = plan_group("qdmsd/ssp245/tas/001", dict(LEGACY_ONLY), "CESM2-WACCM6", "output")
    assert plan.to_set["sai_downscaling:gcm"] == "CESM2-WACCM6"
    assert plan.to_set["sai_downscaling:downscaling_method"] == "QDMSD"


def test_legacy_attr_is_not_deleted_before_its_replacement_exists() -> None:
    """The safety invariant of the 2-pass move: never remove provenance we have not copied yet."""
    plan = plan_group("qdmsd/ssp245/tas/001", dict(LEGACY_ONLY), "CESM2-WACCM6", "output")
    for key in LEGACY_ONLY:
        assert key not in plan.removals, f"{key} would be deleted before it was copied"


def test_legacy_attr_is_deleted_once_its_replacement_exists() -> None:
    """The second pass, after the copy landed."""
    migrated = {**LEGACY_ONLY, **{k.replace("srm_", "sai_"): v for k, v in LEGACY_ONLY.items()}}
    plan = plan_group("qdmsd/ssp245/tas/001", migrated, "CESM2-WACCM6", "output")
    assert set(plan.removals) == set(LEGACY_ONLY)
    assert not any(k.startswith("srm_downscaling:") for k in plan.to_set)


def test_deprecated_fields_are_never_copied_forward() -> None:
    """``config_hash`` is gone for good, so migrating must not resurrect it under the new name."""
    existing = {**LEGACY_ONLY, "srm_downscaling:config_hash": "1cc537e7c038"}
    plan = plan_group("qdmsd/ssp245/tas/001", existing, "CESM2-WACCM6", "output")
    assert "sai_downscaling:config_hash" not in plan.to_set
    assert plan.removals == {"srm_downscaling:config_hash": "1cc537e7c038"}


def test_nothing_writes_the_legacy_namespace() -> None:
    """The move is one-way: we read the old namespace but never add to it."""
    plan = plan_group("qdmsd/ssp245/tas/001", dict(LEGACY_ONLY), "CESM2-WACCM6", "output")
    assert not any(k.startswith("srm_downscaling:") for k in plan.to_set)


def test_a_hand_edited_copy_is_reported_not_silently_resolved() -> None:
    """If the copy disagrees with its source, one was edited. Deleting either loses evidence."""
    existing = {
        **LEGACY_ONLY,
        "sai_downscaling:gcm": "EDITED-BY-HAND",
    }
    plan = plan_group("qdmsd/ssp245/tas/001", existing, "CESM2-WACCM6", "output")
    assert plan.conflicts["srm_downscaling:gcm"] == ("CESM2-WACCM6", "EDITED-BY-HAND")
    assert "srm_downscaling:gcm" not in plan.removals


def test_pipeline_history_names_the_method_that_ran() -> None:
    """The bug this fixes: every group claimed BCSD because the method was a literal."""
    from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
    from saidownscale.pipeline import DownscalingPipeline

    for method in ("BCSD", "QDMSD"):
        config = DownscalingConfig(
            gcm="CESM2-WACCM6",
            variable="tas",
            ensemble_member="003",
            scenario="SSP245",
            predict_period_start=2015,
            predict_period_end=2099,
            downscaling_method=method,
        )
        attrs = DownscalingPipeline(config, PipelineOptions())._build_output_attrs()
        assert f"{method} downscaling" in attrs["history"]


def test_pipeline_publishes_licensed_data_without_a_second_pass() -> None:
    """A run from scratch must label its own output, or new releases ship unlicensed."""
    from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
    from saidownscale.pipeline import DownscalingPipeline

    config = DownscalingConfig(
        gcm="CESM2-WACCM6",
        variable="tas",
        ensemble_member="003",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2099,
        downscaling_method="QDMSD",
    )
    attrs = DownscalingPipeline(config, PipelineOptions())._build_output_attrs()
    assert attrs["license"] == "CC BY 4.0"
    assert attrs["contact"] == "hello@carbonplan.org"
    assert attrs["institution"] == "CarbonPlan"
    assert "Danabasoglu" in attrs["references"]
    assert attrs["source"].endswith("by the QDMSD method")
    assert not any(k.startswith("srm_downscaling:") for k in attrs)
