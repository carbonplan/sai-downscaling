"""Tests for the store metadata rules in :mod:`saidownscale.store_metadata`."""

from pathlib import Path

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from saidownscale import licenses
from saidownscale.config import read_attr
from saidownscale.licenses import metadata_attrs
from saidownscale.store_metadata import (
    DROPPED_PLAIN_ATTRS,
    describe_data_source,
    gcm_from_store,
    plan_group,
    product_from_path,
    published_name,
    repair_history,
    scenario_from_path,
    source_for,
)

CESM = "CESM2-WACCM6"
UKESM = "UKESM1-1-LL"
CC_BY = "CC BY 4.0"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"
OUTPUT_ATTRS = {
    "sai_downscaling:gcm": CESM,
    "sai_downscaling:downscaling_method": "BCSD",
    "sai_downscaling:observation_dataset": "ERA5",
}
HIST_PATH = "bcsd/historical/tas/r1i1p1f1"
MISLABELED_HISTORICAL = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "G6-1.5K"}
LEGACY_HISTORICAL = {
    "srm_downscaling:gcm": CESM,
    "srm_downscaling:downscaling_method": "BCSD",
    "srm_downscaling:observation_dataset": "ERA5",
    "srm_downscaling:scenario": "G6-1.5K",
}
LEGACY_ONLY = {
    "srm_downscaling:gcm": CESM,
    "srm_downscaling:downscaling_method": "QDMSD",
    "srm_downscaling:observation_dataset": "ERA5",
    "srm_downscaling:scenario": "SSP245",
}
CESM_HISTORICAL_COORD = (
    "'001': corrected NCAR/ESGF historical run (all variables); r1/r2/r3i1p1f1: Pangeo CMIP6 "
    "historical (NaN where a variable is missing)"
)
DERIVATION = "Ensemble member derived from filename case segment or variant_label"
WRONG_HISTORY = "2026-09-09T23:05:14Z: BCSD downscaling by srm v1.0.0"


def _out(path: str = "bcsd/ssp245/tas/001", extra: dict | None = None, gcm: str = CESM):
    return plan_group(path, {**OUTPUT_ATTRS, **(extra or {})}, gcm, "output")


def _in(path: str, existing: dict | None = None, gcm: str = CESM, **kwargs):
    return plan_group(path, dict(existing or {}), gcm, "input", **kwargs)


def _output_attrs(scenario: str, method: str, *, for_historical: bool = False) -> dict:
    from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
    from saidownscale.pipeline import DownscalingPipeline

    config = DownscalingConfig(
        gcm=CESM,
        variable="tas",
        ensemble_member="003",
        scenario=scenario,
        predict_period_start=2035,
        predict_period_end=2084,
        downscaling_method=method,
    )
    pipeline = DownscalingPipeline(config, PipelineOptions())
    return pipeline._build_output_attrs(for_historical=for_historical)


def test_scenario_product_and_gcm_parse_from_paths_and_stores(subtests) -> None:
    scenarios = [
        ("bcsd/g6_1p5k/tas/001", "g6_1p5k"),
        ("bcsd/debiased_coarse/g6_1p5k/tas/001", "g6_1p5k"),
        ("qdmsd/g6_1p5k_end/pr/002", "g6_1p5k_end"),
        ("bcsd/historical/tas/r1i1p1f1", "historical"),
        ("ssp245", "ssp245"),
        ("bcsd/tas/001", None),
    ]
    for path, expected in scenarios:
        with subtests.test(scenario_from_path=path):
            assert scenario_from_path(path) == expected
    products = [
        ("bcsd/ssp245/tas/001", "output"),
        ("qdmsd/ssp245/tas/001", "output"),
        ("ssp245", "input"),
    ]
    for path, expected in products:
        with subtests.test(product_from_path=path):
            assert product_from_path(path) == expected
    with subtests.test("gcm from either store layout, or raises"):
        assert gcm_from_store("s3://b/input/processed/CESM2-WACCM6.icechunk") == CESM
        assert gcm_from_store("s3://b/output/UKESM1-1-LL-ERA5-global.icechunk") == UKESM
        with pytest.raises(ValueError, match="cannot tell which GCM"):
            gcm_from_store("s3://b/output/MIROC-ES2H-ERA5-global.icechunk")


def test_plan_sets_missing_keeps_matching_and_reports_differing(subtests) -> None:
    with subtests.test("sets missing"):
        plan = _out("bcsd/g6_1p5k/tas/001")
        assert plan.to_set["license"] == CC_BY
        assert "Walker Lee" in plan.to_set["references"]
        assert plan.writes
    with subtests.test("leaves matching alone"):
        plan = _out("bcsd/g6_1p5k/tas/001", {"institution": "CarbonPlan"})
        assert "institution" not in plan.to_set
        assert plan.unchanged["institution"] == "CarbonPlan"
    with subtests.test("differing is a conflict, not clobbered"):
        plan = _out("bcsd/g6_1p5k/tas/001", {"license": "CC0-1.0"})
        assert plan.conflicts["license"] == ("CC0-1.0", CC_BY)
        assert "license" not in plan.to_set


def test_plan_refuses_to_guess(subtests) -> None:
    """Only historical paths have a known cause for disagreeing with their scenario attr."""
    disagreeing = [
        ("bcsd/g6_1p5k/tas/001", "SSP245"),
        ("bcsd/ssp245/tas/001", "G6-1.5K"),
    ]
    for path, scenario in disagreeing:
        with subtests.test(path=path, scenario=scenario):
            with pytest.raises(ValueError, match="Refusing to guess"):
                _out(path, {"sai_downscaling:scenario": scenario})
    with subtests.test("unattributed pair"):
        with pytest.raises(KeyError, match="no license or citation recorded"):
            _out("bcsd/esgf_ssp245/tas/001")


def test_input_and_output_groups_get_different_attrs(subtests) -> None:
    out = _out()
    inp = _in("ssp245", gcm=UKESM)
    with subtests.test("input keeps its own source"):
        plan = _in("ssp245", {"source": "UM"}, gcm=UKESM)
        assert "source" not in plan.to_set
        assert "source" not in plan.conflicts
    with subtests.test("institution only on output"):
        assert out.to_set["institution"] == "CarbonPlan"
        for scenario in ("historical", "ssp245", "g6_1p5k"):
            assert "institution" not in _in(scenario, gcm=UKESM).to_set
    with subtests.test("attribution on input, references on output"):
        assert "Andy Jones" in inp.to_set["attribution"]
        assert "references" not in inp.to_set
        assert "Danabasoglu" in out.to_set["references"]
        assert "attribution" not in out.to_set
    with subtests.test("contact only on output"):
        assert out.to_set["contact"] == "hello@carbonplan.org"
        assert "contact" not in inp.to_set
    with subtests.test("doi only on output"):
        assert out.to_set["doi"] == "https://doi.org/10.5281/zenodo.22932138"
        assert "doi" not in inp.to_set
    with subtests.test("data_source only on input"):
        plan = _in("ssp245", {"processing_steps": "lon_to_180"}, gcm=UKESM)
        assert plan.to_set["data_source"].startswith("This icechunk store was created by ingesting")
        assert "`processing_steps` attribute" in plan.to_set["data_source"]
        assert "data_source" not in out.to_set
    with subtests.test("terms_of_data_access on both"):
        assert _out(gcm=UKESM).to_set["terms_of_data_access"] == licenses.TERMS_OF_DATA_ACCESS
        assert inp.to_set["terms_of_data_access"] == licenses.TERMS_OF_DATA_ACCESS


def test_licenses_resolve_per_group_and_withhold_url_on_conflict(subtests) -> None:
    with subtests.test("UKESM licenses differ within one store"):
        assert _in("historical", gcm=UKESM).to_set["license"] == CC_BY
        assert _in("ssp245", gcm=UKESM).to_set["license"] == "OGLv3"
    with subtests.test("unasserted input license is published blank"):
        plan = _in("g6_1p5k")
        assert plan.to_set["license"] == ""
        assert plan.to_set["license_url"] == ""
        assert "Walker Lee" in plan.to_set["attribution"]
    with subtests.test("output licensed when its input is not"):
        assert _out("bcsd/g6_1p5k/tas/001").to_set["license"] == CC_BY
    with subtests.test("output license matches the docs table spelling"):
        plan = _out()
        assert plan.to_set["license"] == CC_BY
        assert plan.to_set["license_url"] == CC_BY_URL
    with subtests.test("clean group gets both halves"):
        plan = _in("historical")
        assert plan.to_set["license"] == CC_BY
        assert plan.to_set["license_url"] == CC_BY_URL
        assert not plan.conflicts
    with subtests.test("license_url withheld while license conflicts"):
        plan = _in("historical", {"license": "CMIP6 model data ... ShareAlike 4.0 ..."})
        assert "license_url" not in plan.to_set
        assert plan.conflicts["license_url"] == (None, CC_BY_URL)
        assert plan.conflicts["license"][1] == CC_BY


def test_mislabeled_historical_scenario_is_repaired(subtests) -> None:
    """Stage 2 is cached without the scenario, so historical groups carry the first run's."""
    plan = plan_group(HIST_PATH, dict(MISLABELED_HISTORICAL), CESM, "output")
    with subtests.test("repair, not set"):
        assert plan.repairs["sai_downscaling:scenario"] == ("G6-1.5K", "historical")
        assert "sai_downscaling:scenario" not in plan.to_set
    with subtests.test("cited as its own simulation"):
        assert "CMIP historical" in plan.to_set["references"]
        assert "Walker Lee" not in plan.to_set["references"]
    with subtests.test("described as historical"):
        assert plan.to_set["source"].startswith("CESM2-WACCM6 historical, downscaled")
    with subtests.test("config_json survives"):
        plan = plan_group(
            HIST_PATH,
            {**MISLABELED_HISTORICAL, "sai_downscaling:config_json": '{"scenario":"G6-1.5K"}'},
            CESM,
            "output",
        )
        assert "sai_downscaling:config_json" not in plan.repairs
        assert "sai_downscaling:config_json" not in plan.removals


def test_legacy_historical_group_migrates_without_copying_stale_scenario(subtests) -> None:
    with subtests.test("stale scenario is not copied forward and legacy stays"):
        plan = plan_group(HIST_PATH, dict(LEGACY_HISTORICAL), CESM, "output")
        assert "sai_downscaling:scenario" not in plan.to_set
        assert plan.repairs["sai_downscaling:scenario"] == ("G6-1.5K", "historical")
        assert "srm_downscaling:scenario" not in plan.removals
        assert "srm_downscaling:scenario" not in plan.conflicts
    with subtests.test("repaired scenario does not block the prune"):
        second_pass = {**LEGACY_HISTORICAL, "sai_downscaling:scenario": "historical"}
        plan = plan_group(HIST_PATH, second_pass, CESM, "output")
        assert plan.removals["srm_downscaling:scenario"] == "G6-1.5K"
        assert "srm_downscaling:scenario" not in plan.conflicts
    with subtests.test("correctly labeled group migrates normally"):
        existing = {**LEGACY_HISTORICAL, "srm_downscaling:scenario": "historical"}
        plan = plan_group(HIST_PATH, existing, CESM, "output")
        assert plan.to_set["sai_downscaling:scenario"] == "historical"
        assert "sai_downscaling:scenario" not in plan.repairs
    with subtests.test("hand edit on another field still conflicts"):
        existing = {
            **LEGACY_HISTORICAL,
            "sai_downscaling:scenario": "historical",
            "sai_downscaling:gcm": "EDITED-BY-HAND",
        }
        plan = plan_group(HIST_PATH, existing, CESM, "output")
        assert plan.conflicts["srm_downscaling:gcm"] == (CESM, "EDITED-BY-HAND")


def test_config_scenario_spellings_are_replaced_by_published_names(subtests) -> None:
    renames = [
        ("bcsd/ssp245/tas/002", "output", "SSP245", "SSP2-4.5"),
        ("bcsd/g6_1p5k/tas/002", "output", "G6-1.5K", "G6-1.5K-SAI"),
        ("bcsd/g6_1p5k_end/tas/002", "output", "G6-1.5K-END", "G6-1.5K-SAI-END"),
        ("ssp245", "input", "SSP245", "SSP2-4.5"),
        ("g6_1p5k", "input", "G6-1.5K", "G6-1.5K-SAI"),
    ]
    for path, product, old, new in renames:
        with subtests.test(path=path, product=product):
            if product == "input":
                plan, key = _in(path, {"scenario": old}), "scenario"
            else:
                key = "sai_downscaling:scenario"
                plan = _out(path, {key: old})
            assert plan.repairs[key] == (old, new)
    with subtests.test("already published name is a no-op"):
        plan = _out("bcsd/ssp245/tas/002", {"sai_downscaling:scenario": "SSP2-4.5"})
        assert "sai_downscaling:scenario" not in plan.repairs
        assert "sai_downscaling:scenario" not in plan.conflicts
    with subtests.test("composed source uses published name"):
        plan = _out("bcsd/g6_1p5k/tas/002", {"sai_downscaling:scenario": "G6-1.5K"})
        assert plan.to_set["source"] == (
            "CESM2-WACCM6 G6-1.5K-SAI, downscaled to the ERA5 grid by the BCSD method"
        )
    with subtests.test("old spelling does not block the legacy prune"):
        second_pass = {
            **LEGACY_ONLY,
            "srm_downscaling:downscaling_method": "BCSD",
            "sai_downscaling:scenario": "SSP2-4.5",
        }
        plan = plan_group("bcsd/ssp245/tas/002", second_pass, CESM, "output")
        assert plan.removals["srm_downscaling:scenario"] == "SSP245"
        assert "srm_downscaling:scenario" not in plan.conflicts
    with subtests.test("sai parent renamed with the group"):
        plan = _out(
            "bcsd/g6_1p5k_end/tas/002",
            {
                "sai_downscaling:scenario": "G6-1.5K-END",
                "sai_downscaling:sai_parent_scenario": "G6-1.5K",
            },
        )
        assert plan.repairs["sai_downscaling:scenario"] == ("G6-1.5K-END", "G6-1.5K-SAI-END")
        assert plan.repairs["sai_downscaling:sai_parent_scenario"] == ("G6-1.5K", "G6-1.5K-SAI")
    with subtests.test("no sai parent is invented"):
        plan = _out("bcsd/ssp245/tas/002")
        assert "sai_downscaling:sai_parent_scenario" not in plan.repairs
        assert "sai_downscaling:sai_parent_scenario" not in plan.to_set
    with subtests.test("unrecognized name handed back unchanged"):
        assert published_name("something-we-have-never-seen") == "something-we-have-never-seen"
        assert published_name("SSP245") == "SSP2-4.5"


def test_input_groups_drop_personal_and_lineage_attrs_but_keep_provenance(subtests) -> None:
    with subtests.test("personal and machine run detail removed"):
        plan = _in("ssp245", {"scenario": "SSP245", "logname": "cmip6", "host": "cheyenne4"})
        assert plan.removals == {"logname": "cmip6", "host": "cheyenne4"}
    with subtests.test("inherited CMIP6 run detail removed"):
        inherited = {"table_id": "day", "grid_label": "gn", "forcing_index": 1, "realm": "atmos"}
        plan = _in("historical", {"scenario": "historical", **inherited})
        assert set(plan.removals) == set(inherited)
    with subtests.test("the creator email goes with it"):
        status = "2019-11-04;created;by someone@example.edu"
        plan = _in("historical", {"scenario": "historical", "status": status})
        assert plan.removals["status"] == status
    with subtests.test("nothing we write is also on the drop list"):
        for product in ("input", "output"):
            written = set(metadata_attrs("CESM2-WACCM6", "historical", product=product))
            assert not written & DROPPED_PLAIN_ATTRS
        assert "contact" not in DROPPED_PLAIN_ATTRS
    with subtests.test("the r/i/p/f decomposition goes as a set"):
        for key in ("forcing_index", "initialization_index", "physics_index"):
            assert key in DROPPED_PLAIN_ATTRS
    with subtests.test("what a reader needs is kept"):
        for key in ("scenario", "source", "model", "Conventions", "case", "model_doi_url"):
            assert key not in DROPPED_PLAIN_ATTRS
    with subtests.test("experiment_lineage is a removal and nothing else"):
        lineage = "unknown_parent -> SSP245"
        plan = _in("ssp245", {"scenario": "SSP245", "experiment_lineage": lineage})
        assert plan.removals == {"experiment_lineage": lineage}
        for key in plan.removals:
            assert key not in plan.to_set
            assert key not in plan.conflicts
            assert key not in plan.repairs
    with subtests.test("parent_experiment_id kept"):
        plan = _in("historical", {"scenario": "historical", "parent_experiment_id": "piControl"})
        assert "parent_experiment_id" not in plan.removals
    with subtests.test("processing_steps kept"):
        assert "processing_steps" not in DROPPED_PLAIN_ATTRS
        plan = _in("ssp245", {"scenario": "SSP245", "processing_steps": "lon_to_180, lat_lon_sort"})
        assert "processing_steps" not in plan.removals
        assert "`processing_steps` attribute" in plan.to_set["data_source"]


def test_derivation_prose_is_removed_from_group_and_matching_coordinate(subtests) -> None:
    existing = {"scenario": "SSP245", "ensemble_derivation_logic": DERIVATION}
    with subtests.test("group attr removed"):
        assert _in("ssp245", existing).removals["ensemble_derivation_logic"] == DERIVATION
    with subtests.test("coordinate duplicate is a coord removal and nothing else"):
        plan = _in(
            "ssp245",
            existing,
            coord_attrs={"ensemble_member": {"long_name": "x", "derivation_method": DERIVATION}},
        )
        assert plan.coord_removals["ensemble_member"] == {"derivation_method": DERIVATION}
        for attrs in plan.coord_removals.values():
            for key in attrs:
                assert key not in plan.to_set
                assert key not in plan.repairs
    with subtests.test("coordinate saying something else kept"):
        plan = _in(
            "historical",
            {"scenario": "historical", "ensemble_derivation_logic": DERIVATION},
            coord_attrs={"ensemble_member": {"derivation_method": CESM_HISTORICAL_COORD}},
        )
        assert plan.coord_removals == {}
        assert plan.removals["ensemble_derivation_logic"] == DERIVATION


def test_source_and_data_source_describe_real_provenance(subtests) -> None:
    with subtests.test("source needs real provenance"):
        assert source_for({}, CESM) is None
        assert "ERA5" in source_for(OUTPUT_ATTRS, CESM)
    with subtests.test("data_source processing_steps clause"):
        without = describe_data_source(has_processing_steps=False)
        assert "processing_steps" not in without
        assert without.endswith(
            "See https://github.com/carbonplan/sai-downscaling for more information."
        )
        assert "processing_steps" in describe_data_source(has_processing_steps=True)


def test_history_naming_the_wrong_method_is_repaired_or_declined(subtests) -> None:
    qdmsd = {**OUTPUT_ATTRS, "sai_downscaling:downscaling_method": "QDMSD"}
    with subtests.test("wrong method repaired, timestamp and producer kept"):
        corrected = repair_history({**qdmsd, "history": WRONG_HISTORY})
        assert corrected == "2026-09-09T23:05:14Z: QDMSD downscaling by srm v1.0.0"
    with subtests.test("correct history left alone"):
        assert repair_history({**OUTPUT_ATTRS, "history": WRONG_HISTORY}) is None
    declines = {
        "no-history": {"sai_downscaling:downscaling_method": "QDMSD"},
        "no-method": {"history": WRONG_HISTORY},
        "unparseable": {
            "history": "hand written note",
            "sai_downscaling:downscaling_method": "QDMSD",
        },
    }
    for case, existing in declines.items():
        with subtests.test(declines=case):
            assert repair_history(existing) is None
    with subtests.test("repair is kept out of to_set and conflicts"):
        plan = plan_group(
            "qdmsd/ssp245/tas/001", {**qdmsd, "history": WRONG_HISTORY}, CESM, "output"
        )
        assert "history" in plan.repairs
        assert "history" not in plan.to_set
        assert "history" not in plan.conflicts


def test_deprecated_output_attrs_are_removed_and_never_copied_forward(subtests) -> None:
    with subtests.test("config_hash is a removal and nothing else"):
        plan = _out(extra={"sai_downscaling:config_hash": "1cc537e7c038"})
        assert plan.removals == {"sai_downscaling:config_hash": "1cc537e7c038"}
        for key in plan.removals:
            assert key not in plan.to_set
            assert key not in plan.conflicts
            assert key not in plan.repairs
    with subtests.test("config_json kept"):
        assert _out(extra={"sai_downscaling:config_json": "{}"}).removals == {}
    with subtests.test("never copied forward from legacy"):
        existing = {**LEGACY_ONLY, "srm_downscaling:config_hash": "1cc537e7c038"}
        plan = plan_group("qdmsd/ssp245/tas/001", existing, CESM, "output")
        assert "sai_downscaling:config_hash" not in plan.to_set
        assert plan.removals == {"srm_downscaling:config_hash": "1cc537e7c038"}


def test_doi_is_a_resolvable_url_that_follows_the_module_setting(
    subtests, monkeypatch: pytest.MonkeyPatch
) -> None:
    with subtests.test("stored as a resolvable URL"):
        assert licenses.DOI is not None
        assert licenses.DOI.startswith("https://doi.org/10.")
    with subtests.test("a changed DOI flows through"):
        with monkeypatch.context() as m:
            m.setattr(licenses, "DOI", "10.5281/zenodo.123456")
            attrs = licenses.metadata_attrs(CESM, "ssp245", product="output")
            assert attrs["doi"] == "10.5281/zenodo.123456"
    with subtests.test("an unset DOI writes nothing"):
        with monkeypatch.context() as m:
            m.setattr(licenses, "DOI", None)
            assert "doi" not in licenses.metadata_attrs(CESM, "ssp245", product="output")


def test_legacy_namespace_is_copied_forward_then_deleted_safely(subtests) -> None:
    path = "qdmsd/ssp245/tas/001"
    first = plan_group(path, dict(LEGACY_ONLY), CESM, "output")
    with subtests.test("readers accept the legacy namespace"):
        assert read_attr(LEGACY_ONLY, "gcm") == CESM
        assert source_for(LEGACY_ONLY, CESM).endswith("by the QDMSD method")
    with subtests.test("current namespace wins"):
        assert read_attr({"srm_downscaling:gcm": "old", "sai_downscaling:gcm": "new"}, "gcm") == (
            "new"
        )
    with subtests.test("legacy copied forward, nothing writes legacy"):
        assert first.to_set["sai_downscaling:gcm"] == CESM
        assert first.to_set["sai_downscaling:downscaling_method"] == "QDMSD"
        assert not any(k.startswith("srm_downscaling:") for k in first.to_set)
    with subtests.test("legacy not deleted before its replacement exists"):
        for key in LEGACY_ONLY:
            assert key not in first.removals, f"{key} would be deleted before it was copied"
    with subtests.test("legacy deleted once its replacement exists"):
        migrated = {**LEGACY_ONLY, **{k.replace("srm_", "sai_"): v for k, v in LEGACY_ONLY.items()}}
        plan = plan_group(path, migrated, CESM, "output")
        assert set(plan.removals) == set(LEGACY_ONLY)
        assert not any(k.startswith("srm_downscaling:") for k in plan.to_set)
    with subtests.test("hand-edited copy reported, not silently resolved"):
        plan = plan_group(
            path, {**LEGACY_ONLY, "sai_downscaling:gcm": "EDITED-BY-HAND"}, CESM, "output"
        )
        assert plan.conflicts["srm_downscaling:gcm"] == (CESM, "EDITED-BY-HAND")
        assert "srm_downscaling:gcm" not in plan.removals


def test_pipeline_writes_labeled_licensed_attrs_in_one_pass(subtests) -> None:
    with subtests.test("historical write labeled historical"):
        hist = _output_attrs("G6-1.5K", "BCSD", for_historical=True)
        assert hist["sai_downscaling:scenario"] == "historical"
        assert "CMIP historical" in hist["references"]
        assert hist["source"].startswith("CESM2-WACCM6 historical, downscaled")
    with subtests.test("scenario write labeled with the published name"):
        scenario_attrs = _output_attrs("G6-1.5K", "BCSD")
        assert scenario_attrs["sai_downscaling:scenario"] == "G6-1.5K-SAI"
        assert "Walker Lee" in scenario_attrs["references"]
    for method in ("BCSD", "QDMSD"):
        with subtests.test(history_names_method=method):
            attrs = _output_attrs("SSP245", method)
            assert f"{method} downscaling" in attrs["history"]
    with subtests.test("licensed without a second pass"):
        attrs = _output_attrs("SSP245", "QDMSD")
        assert attrs["license"] == CC_BY
        assert attrs["contact"] == "hello@carbonplan.org"
        assert attrs["institution"] == "CarbonPlan"
        assert "Danabasoglu" in attrs["references"]
        assert attrs["source"].endswith("by the QDMSD method")
        assert not any(k.startswith("srm_downscaling:") for k in attrs)


def test_writing_attrs_adds_no_chunks_and_keeps_the_data(tmp_path: Path) -> None:
    store_path = tmp_path / "CESM2-WACCM6-ERA5-global.icechunk"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(path=str(store_path)))
    session = repo.writable_session("main")
    ds = xr.Dataset({"tas": (("time",), np.arange(8, dtype="float32"))})
    ds.attrs = {"Conventions": "CF-1.8", **OUTPUT_ATTRS, "sai_downscaling:scenario": "G6-1.5K"}
    ds.to_zarr(session.store, group="bcsd/g6_1p5k/tas/001", mode="w", consolidated=False)
    session.commit("seed")

    def census() -> dict[str, int]:
        return {
            kind: len(list((store_path / kind).glob("**/*")))
            for kind in ("chunks", "manifests")
            if (store_path / kind).exists()
        }

    before = census()
    plan = _out("bcsd/g6_1p5k/tas/001")
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
    assert result.attrs["license"] == CC_BY
    assert result.attrs["Conventions"] == "CF-1.8", "existing attrs must survive"
    assert result.attrs["sai_downscaling:gcm"] == CESM
