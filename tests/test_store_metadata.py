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
    describe_data_source,
    gcm_from_store,
    plan_group,
    product_from_path,
    published_name,
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
    Research on CESM historical. Writing ourselves over that would credit us for a simulation we
    only redistribute.
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


def test_input_with_no_asserted_license_gets_a_blank_one() -> None:
    """A blank license in the table is a decision not to grant one, published as a blank attr.

    Leaving the attr out entirely would read as an oversight, while a blank one shows the question
    was asked. Neither can be mistaken for a grant, and the citation is recorded either way.
    """
    plan = plan_group("g6_1p5k", {}, "CESM2-WACCM6", "input")
    assert plan.to_set["license"] == ""
    assert plan.to_set["license_url"] == ""
    assert "Walker Lee" in plan.to_set["attribution"]


def test_input_publishes_the_citation_as_attribution() -> None:
    """Input groups use the name the licenses table uses, which is what readers of them expect.

    Our own output keeps the CF ``references`` instead, so the 2 products are checked separately
    rather than assumed to agree. Both names have to be pinned, because a rename on one side is
    exactly the kind of change that passes review unnoticed.
    """
    inp = plan_group("ssp245", {}, "UKESM1-1-LL", "input")
    assert "Andy Jones" in inp.to_set["attribution"]
    assert "references" not in inp.to_set
    out = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert "Danabasoglu" in out.to_set["references"]
    assert "attribution" not in out.to_set


def test_output_is_still_licensed_when_its_input_is_not() -> None:
    """Our downscaled product carries its own license regardless of the upstream decision."""
    plan = plan_group("bcsd/g6_1p5k/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert plan.to_set["license"] == "CC BY 4.0"


def test_path_and_attrs_disagreeing_raises() -> None:
    existing = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "SSP245"}
    with pytest.raises(ValueError, match="Refusing to guess"):
        plan_group("bcsd/g6_1p5k/tas/001", existing, "CESM2-WACCM6", "output")


#: What a published historical group really looks like: the path is historical, the recorded
#: scenario is whichever run reached stage 2 first. Not one historical group in either output store
#: records ``historical`` here.
MISLABELED_HISTORICAL = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "G6-1.5K"}


def test_mislabeled_historical_scenario_is_repaired() -> None:
    """Stage 2 is cached without the scenario in its key, so the first run to reach it stamps its
    own config on the historical leg. The path is the reliable witness, and ``config_json`` still
    holds the run's own scenario, so the attr can be corrected without losing anything.
    """
    plan = plan_group(
        "bcsd/historical/tas/r1i1p1f1", dict(MISLABELED_HISTORICAL), "CESM2-WACCM6", "output"
    )
    assert plan.repairs["sai_downscaling:scenario"] == ("G6-1.5K", "historical")
    assert "sai_downscaling:scenario" not in plan.to_set


def test_a_historical_group_is_cited_as_its_own_simulation() -> None:
    """Attribution follows the path, so historical data cites the CMIP6 historical run.

    Following the recorded scenario instead would credit Walker Lee's G6-1.5K simulation on data
    that is not from it, which is a licensing error rather than a cosmetic one. The citation is what
    a downstream paper reproduces, so it has to name the run the numbers actually came from.
    """
    plan = plan_group(
        "bcsd/historical/tas/r1i1p1f1", dict(MISLABELED_HISTORICAL), "CESM2-WACCM6", "output"
    )
    assert "CMIP historical" in plan.to_set["references"]
    assert "Walker Lee" not in plan.to_set["references"]


def test_a_historical_group_is_described_as_historical() -> None:
    """``source`` is composed, so it has to be composed from the corrected scenario.

    Without the override it would read ``CESM2-WACCM6 G6-1.5K, downscaled ...`` on a group whose
    own path says historical, which is the contradiction the repair exists to remove. A reader
    comparing the 2 would have no way to tell which of them to trust.
    """
    plan = plan_group(
        "bcsd/historical/tas/r1i1p1f1", dict(MISLABELED_HISTORICAL), "CESM2-WACCM6", "output"
    )
    assert plan.to_set["source"].startswith("CESM2-WACCM6 historical, downscaled")


def test_only_historical_paths_get_the_benefit_of_the_doubt() -> None:
    """A scenario group disagreeing with its path is unexplained, so it still refuses.

    The historical case has a known cause in the stage 2 cache key. Nothing else does, and guessing
    there would rewrite provenance on the strength of a hunch.
    """
    existing = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "G6-1.5K"}
    with pytest.raises(ValueError, match="Refusing to guess"):
        plan_group("bcsd/ssp245/tas/001", existing, "CESM2-WACCM6", "output")


def test_pipeline_labels_a_historical_write_as_historical() -> None:
    """The pipeline is where this arose, so a from-scratch run must not reintroduce it.

    ``fit_historical`` runs inside a scenario run, so without the flag it stamps that scenario onto
    the historical artifact and cites the wrong simulation with it. Patching the published stores
    without fixing this would leave the next release free to reintroduce it.
    """
    from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
    from saidownscale.pipeline import DownscalingPipeline

    config = DownscalingConfig(
        gcm="CESM2-WACCM6",
        variable="tas",
        ensemble_member="002",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2084,
        downscaling_method="BCSD",
    )
    pipeline = DownscalingPipeline(config, PipelineOptions())
    hist = pipeline._build_output_attrs(for_historical=True)
    assert hist["sai_downscaling:scenario"] == "historical"
    assert "CMIP historical" in hist["references"]
    assert hist["source"].startswith("CESM2-WACCM6 historical, downscaled")
    # The scenario write still records the run it belongs to, under its published name.
    scenario_attrs = pipeline._build_output_attrs()
    assert scenario_attrs["sai_downscaling:scenario"] == "G6-1.5K-SAI"
    assert "Walker Lee" in scenario_attrs["references"]


def test_the_run_config_survives_the_repair() -> None:
    """``config_json`` is what makes correcting ``scenario`` lossless, so it must not be touched."""
    plan = plan_group(
        "bcsd/historical/tas/r1i1p1f1",
        {**MISLABELED_HISTORICAL, "sai_downscaling:config_json": '{"scenario":"G6-1.5K"}'},
        "CESM2-WACCM6",
        "output",
    )
    assert "sai_downscaling:config_json" not in plan.repairs
    assert "sai_downscaling:config_json" not in plan.removals


#: A published historical group as the first pass finds it: provenance under the old namespace
#: only, with the scenario of whichever run wrote the historical leg.
LEGACY_HISTORICAL = {
    "srm_downscaling:gcm": "CESM2-WACCM6",
    "srm_downscaling:downscaling_method": "BCSD",
    "srm_downscaling:observation_dataset": "ERA5",
    "srm_downscaling:scenario": "G6-1.5K",
}
HIST_PATH = "bcsd/historical/tas/r1i1p1f1"


def test_the_stale_scenario_is_not_copied_onto_the_current_namespace() -> None:
    """Copying it forward would only have to be undone, and the repair supplies the right value.

    The legacy attr itself survives this pass untouched, so a run without ``--repair`` leaves the
    group exactly as it found it rather than half converted. Half-converted provenance is the one
    state the 2-pass migration exists to rule out.
    """
    plan = plan_group(HIST_PATH, dict(LEGACY_HISTORICAL), "CESM2-WACCM6", "output")
    assert "sai_downscaling:scenario" not in plan.to_set
    assert plan.repairs["sai_downscaling:scenario"] == ("G6-1.5K", "historical")
    assert "srm_downscaling:scenario" not in plan.removals
    assert "srm_downscaling:scenario" not in plan.conflicts


def test_the_repaired_scenario_does_not_block_the_prune() -> None:
    """A corrected copy differs from its original on purpose, which must not read as a hand edit.

    Without this the legacy attr would be held back as a conflict forever, leaving every historical
    group asserting ``historical`` and ``G6-1.5K`` at once, which is the contradiction the repair
    was meant to remove. The guard is right to distrust an edited copy, so a deliberate correction
    has to announce itself rather than look like one.
    """
    second_pass = {**LEGACY_HISTORICAL, "sai_downscaling:scenario": "historical"}
    plan = plan_group(HIST_PATH, second_pass, "CESM2-WACCM6", "output")
    assert plan.removals["srm_downscaling:scenario"] == "G6-1.5K"
    assert "srm_downscaling:scenario" not in plan.conflicts


def test_a_correctly_labeled_historical_group_is_migrated_normally() -> None:
    """The exemption keys off the value, not the path, so a correct group loses nothing.

    A group already recording ``historical`` has nothing to repair, so its legacy attr is copied
    forward like any other and only then becomes removable. Keying the exemption off the path alone
    would have stranded such a group with no scenario attr at all once the legacy one was pruned.
    """
    existing = {**LEGACY_HISTORICAL, "srm_downscaling:scenario": "historical"}
    plan = plan_group(HIST_PATH, existing, "CESM2-WACCM6", "output")
    assert plan.to_set["sai_downscaling:scenario"] == "historical"
    assert "sai_downscaling:scenario" not in plan.repairs


def test_a_hand_edit_on_another_field_still_conflicts() -> None:
    """The exemption is scoped to the one field we correct, so every other guard still holds."""
    existing = {
        **LEGACY_HISTORICAL,
        "sai_downscaling:scenario": "historical",
        "sai_downscaling:gcm": "EDITED-BY-HAND",
    }
    plan = plan_group(HIST_PATH, existing, "CESM2-WACCM6", "output")
    assert plan.conflicts["srm_downscaling:gcm"] == ("CESM2-WACCM6", "EDITED-BY-HAND")


@pytest.mark.parametrize(
    ("path", "product", "existing", "key", "expected"),
    [
        (
            "bcsd/ssp245/tas/002",
            "output",
            {"sai_downscaling:scenario": "SSP245"},
            "sai_downscaling:scenario",
            "SSP2-4.5",
        ),
        (
            "bcsd/g6_1p5k/tas/002",
            "output",
            {"sai_downscaling:scenario": "G6-1.5K"},
            "sai_downscaling:scenario",
            "G6-1.5K-SAI",
        ),
        (
            "bcsd/g6_1p5k_end/tas/002",
            "output",
            {"sai_downscaling:scenario": "G6-1.5K-END"},
            "sai_downscaling:scenario",
            "G6-1.5K-SAI-END",
        ),
        ("ssp245", "input", {"scenario": "SSP245"}, "scenario", "SSP2-4.5"),
        ("g6_1p5k", "input", {"scenario": "G6-1.5K"}, "scenario", "G6-1.5K-SAI"),
    ],
    ids=["out-ssp245", "out-g6", "out-g6-end", "in-ssp245", "in-g6"],
)
def test_the_published_scenario_name_replaces_our_config_spelling(
    path: str, product: str, existing: dict[str, str], key: str, expected: str
) -> None:
    """Readers outside this project know the CMIP6 and GeoMIP names, not our config spellings.

    The attr differs by product, plain on input and namespaced on output, so both are exercised
    here rather than assuming the 2 stores agree on where the scenario lives. Getting that wrong
    would leave one product renamed and the other silently skipped.
    """
    attrs = existing if product == "input" else {**OUTPUT_ATTRS, **existing}
    plan = plan_group(path, attrs, "CESM2-WACCM6", product)
    assert plan.repairs[key] == (
        existing["scenario"] if product == "input" else existing["sai_downscaling:scenario"],
        expected,
    )


def test_a_group_already_carrying_the_published_name_is_left_alone() -> None:
    """Re-running must be a no-op, or every pass would rewrite the same groups again.

    This is also why the published spellings are accepted by ``SCENARIO_TO_GROUP``: without them
    the group would fail the path check instead of being recognized as already correct. A second
    pass would then report the store we just patched as unresolvable.
    """
    existing = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "SSP2-4.5"}
    plan = plan_group("bcsd/ssp245/tas/002", existing, "CESM2-WACCM6", "output")
    assert "sai_downscaling:scenario" not in plan.repairs
    assert "sai_downscaling:scenario" not in plan.conflicts


def test_the_composed_source_uses_the_published_name() -> None:
    """``source`` is prose a reader parses, so it names the scenario the way the attr does."""
    existing = {**OUTPUT_ATTRS, "sai_downscaling:scenario": "G6-1.5K"}
    plan = plan_group("bcsd/g6_1p5k/tas/002", existing, "CESM2-WACCM6", "output")
    assert plan.to_set["source"] == (
        "CESM2-WACCM6 G6-1.5K-SAI, downscaled to the ERA5 grid by the BCSD method"
    )


def test_the_old_spelling_does_not_block_the_legacy_prune() -> None:
    """The rename makes the copy differ from its original on every group, not just historical.

    The prune guard would otherwise read that as a hand edit and hold the legacy attr back on all
    458 published groups rather than the 68 historical ones. Every group would then keep both
    spellings side by side, which is the state the rename is meant to end.
    """
    second_pass = {
        "srm_downscaling:gcm": "CESM2-WACCM6",
        "srm_downscaling:downscaling_method": "BCSD",
        "srm_downscaling:observation_dataset": "ERA5",
        "srm_downscaling:scenario": "SSP245",
        "sai_downscaling:scenario": "SSP2-4.5",
    }
    plan = plan_group("bcsd/ssp245/tas/002", second_pass, "CESM2-WACCM6", "output")
    assert plan.removals["srm_downscaling:scenario"] == "SSP245"
    assert "srm_downscaling:scenario" not in plan.conflicts


def test_the_sai_parent_scenario_is_renamed_with_the_group() -> None:
    """The termination scenario records the run it continues, and both names are published.

    Leaving the parent alone would put ``G6-1.5K-SAI-END`` and ``G6-1.5K`` on one group, which
    reads as 2 different naming conventions rather than as a parent and a child. The parent names a
    different scenario than the group, so the path cannot settle this one for us.
    """
    existing = {
        **OUTPUT_ATTRS,
        "sai_downscaling:scenario": "G6-1.5K-END",
        "sai_downscaling:sai_parent_scenario": "G6-1.5K",
    }
    plan = plan_group("bcsd/g6_1p5k_end/tas/002", existing, "CESM2-WACCM6", "output")
    assert plan.repairs["sai_downscaling:scenario"] == ("G6-1.5K-END", "G6-1.5K-SAI-END")
    assert plan.repairs["sai_downscaling:sai_parent_scenario"] == ("G6-1.5K", "G6-1.5K-SAI")


def test_a_group_with_no_sai_parent_gains_none() -> None:
    """Only a scenario continuing an earlier run has a parent, so nothing may invent one."""
    plan = plan_group("bcsd/ssp245/tas/002", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert "sai_downscaling:sai_parent_scenario" not in plan.repairs
    assert "sai_downscaling:sai_parent_scenario" not in plan.to_set


def test_an_unrecognized_scenario_name_is_handed_back_unchanged() -> None:
    """Rewriting a name we cannot place would invent provenance, so the helper declines to.

    The disagreement then surfaces through the path check, which refuses the group and names it,
    rather than being quietly normalized into something plausible. A plausible wrong answer in
    published metadata is the one outcome nobody downstream can detect.
    """
    assert published_name("something-we-have-never-seen") == "something-we-have-never-seen"
    assert published_name("SSP245") == "SSP2-4.5"


def test_the_researchers_account_name_is_not_republished() -> None:
    """``logname`` is a personal account name that came in with the source netCDF.

    Everyone behind each run is credited by name in ``attribution``, which is where a reader should
    look, so a username adds nothing and is not ours to publish.
    """
    existing = {"scenario": "SSP245", "logname": "cmip6", "host": "cheyenne4"}
    plan = plan_group("ssp245", existing, "CESM2-WACCM6", "input")
    assert plan.removals["logname"] == "cmip6"
    # ``host`` names a machine rather than a person, so it is left where it is.
    assert "host" not in plan.removals


def test_experiment_lineage_is_planned_for_removal() -> None:
    """It restated the scenario behind a literal ``unknown_parent`` on every raw CAM delivery.

    The one group with a real parent records it verbatim in ``parent_experiment_id`` anyway, so
    nothing is lost by dropping the derived attr. No UKESM group ever had it, so it was never a
    field a reader could rely on across the 2 stores.
    """
    existing = {"scenario": "SSP245", "experiment_lineage": "unknown_parent -> SSP245"}
    plan = plan_group("ssp245", existing, "CESM2-WACCM6", "input")
    assert plan.removals == {"experiment_lineage": "unknown_parent -> SSP245"}


def test_a_plain_deprecated_attr_needs_the_prune_flag_like_any_other() -> None:
    """Removals cannot be undone by re-running, so a plain attr gets no shortcut past the flag."""
    existing = {"scenario": "SSP245", "experiment_lineage": "unknown_parent -> SSP245"}
    plan = plan_group("ssp245", existing, "CESM2-WACCM6", "input")
    for key in plan.removals:
        assert key not in plan.to_set
        assert key not in plan.conflicts
        assert key not in plan.repairs


def test_the_real_parent_attr_is_left_alone() -> None:
    """``parent_experiment_id`` comes from the source file and is the fact worth keeping."""
    existing = {"scenario": "historical", "parent_experiment_id": "piControl"}
    plan = plan_group("historical", existing, "CESM2-WACCM6", "input")
    assert "parent_experiment_id" not in plan.removals


#: The published CESM coordinate sentence, which is not derivation prose at all. It names which
#: historical members are complete and which hold NaN, so it must survive the prune.
CESM_HISTORICAL_COORD = (
    "'001': corrected NCAR/ESGF historical run (all variables); r1/r2/r3i1p1f1: Pangeo CMIP6 "
    "historical (NaN where a variable is missing)"
)
DERIVATION = "Ensemble member derived from filename case segment or variant_label"


def test_the_derivation_prose_is_planned_for_removal() -> None:
    """It was rendered at ingest from the suite-to-member lookups, which are the real record.

    A prose copy in a published store can fall behind the tables it was rendered from, so the code
    is the reference and the attr goes.
    """
    existing = {"scenario": "SSP245", "ensemble_derivation_logic": DERIVATION}
    plan = plan_group("ssp245", existing, "CESM2-WACCM6", "input")
    assert plan.removals["ensemble_derivation_logic"] == DERIVATION


def test_the_duplicate_on_the_coordinate_goes_with_it() -> None:
    """The same sentence was written twice under 2 different names, so both copies have to go.

    Dropping only the group attr would leave the text published on the ``ensemble_member``
    coordinate as ``derivation_method``, which defeats the point of removing it.
    """
    existing = {"scenario": "SSP245", "ensemble_derivation_logic": DERIVATION}
    plan = plan_group(
        "ssp245",
        existing,
        "CESM2-WACCM6",
        "input",
        coord_attrs={"ensemble_member": {"long_name": "x", "derivation_method": DERIVATION}},
    )
    assert plan.coord_removals["ensemble_member"] == {"derivation_method": DERIVATION}


def test_a_coordinate_saying_something_else_is_left_alone() -> None:
    """One published coordinate carries a data-quality fact rather than derivation prose.

    It names which CESM historical members are complete and which hold NaN, which has no other
    home, so the rule keys on the value matching the group attr rather than on the attr name.
    """
    existing = {"scenario": "historical", "ensemble_derivation_logic": DERIVATION}
    plan = plan_group(
        "historical",
        existing,
        "CESM2-WACCM6",
        "input",
        coord_attrs={"ensemble_member": {"derivation_method": CESM_HISTORICAL_COORD}},
    )
    assert plan.coord_removals == {}
    assert plan.removals["ensemble_derivation_logic"] == DERIVATION


def test_processing_steps_is_deliberately_kept() -> None:
    """It is the only published record of the transformations applied to the values.

    Longitude rotation, dropped duplicate timestamps, trimmed negative precipitation and the
    calendar conversion all change how a reader interprets the data, and ``data_source`` points at
    this attr by name, so a future sweep must not take it.
    """
    from saidownscale.store_metadata import DROPPED_PLAIN_ATTRS

    assert "processing_steps" not in DROPPED_PLAIN_ATTRS
    existing = {"scenario": "SSP245", "processing_steps": "lon_to_180, lat_lon_sort"}
    plan = plan_group("ssp245", existing, "CESM2-WACCM6", "input")
    assert "processing_steps" not in plan.removals
    assert "`processing_steps` attribute" in plan.to_set["data_source"]


def test_coord_removals_still_need_the_prune_flag() -> None:
    """A coordinate attr is as irreversible as a group attr, so it gets no shortcut."""
    existing = {"scenario": "SSP245", "ensemble_derivation_logic": DERIVATION}
    plan = plan_group(
        "ssp245",
        existing,
        "CESM2-WACCM6",
        "input",
        coord_attrs={"ensemble_member": {"derivation_method": DERIVATION}},
    )
    for attrs in plan.coord_removals.values():
        for key in attrs:
            assert key not in plan.to_set
            assert key not in plan.repairs


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
    """The published data and the published table must not name the same license 2 ways."""
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
    history rather than correct it. The timestamp is a fact of the same kind, which is why only the
    method token moves.
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


def test_doi_is_omitted_until_it_has_a_real_value() -> None:
    """A placeholder in published metadata is worse than an absent attribute."""
    plan = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert "doi" not in plan.to_set


def test_doi_appears_once_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the constant is all it takes; the script needs no further change."""
    monkeypatch.setattr(licenses, "DOI", "10.5281/zenodo.123456")
    attrs = licenses.metadata_attrs("CESM2-WACCM6", "ssp245", product="output")
    assert attrs["doi"] == "10.5281/zenodo.123456"


@pytest.mark.parametrize(
    ("path", "product"), [("bcsd/ssp245/tas/001", "output"), ("ssp245", "input")]
)
def test_terms_of_data_access_is_set_on_both_products(path: str, product: str) -> None:
    """The terms govern a simulation we redistribute as much as output we derived ourselves."""
    plan = plan_group(
        path, dict(OUTPUT_ATTRS) if product == "output" else {}, "UKESM1-1-LL", product
    )
    assert plan.to_set["terms_of_data_access"] == licenses.TERMS_OF_DATA_ACCESS


def test_input_groups_describe_how_the_store_was_built() -> None:
    """``source`` on an input group is the model, so our own provenance needs its own key."""
    plan = plan_group("ssp245", {"processing_steps": "lon_to_180"}, "UKESM1-1-LL", "input")
    assert plan.to_set["data_source"].startswith("This icechunk store was created by ingesting")
    assert "`processing_steps` attribute" in plan.to_set["data_source"]


def test_data_source_drops_the_processing_steps_clause_when_there_are_none() -> None:
    """No UKESM input group records processing steps, so naming that attr there would dangle."""
    without = describe_data_source(has_processing_steps=False)
    assert "processing_steps" not in without
    assert without.endswith(
        "See https://github.com/carbonplan/sai-downscaling for more information."
    )
    assert "processing_steps" in describe_data_source(has_processing_steps=True)


def test_output_groups_get_no_data_source() -> None:
    """Output provenance is already composed into CF ``source``, so a second key would duplicate."""
    plan = plan_group("bcsd/ssp245/tas/001", dict(OUTPUT_ATTRS), "CESM2-WACCM6", "output")
    assert "data_source" not in plan.to_set


def test_license_url_is_withheld_while_the_license_conflicts() -> None:
    """Published CESM historical input inherits the CMIP6 CC BY-SA statement from its netCDF.

    Adding our ``license_url`` while leaving that text alone would leave the group naming 2
    different licenses, so the pair moves together or not at all. Metadata that is incomplete can
    still be completed later, while metadata that contradicts itself misleads every reader of it.
    """
    existing = {"license": "CMIP6 model data ... ShareAlike 4.0 ..."}
    plan = plan_group("historical", existing, "CESM2-WACCM6", "input")
    assert "license_url" not in plan.to_set
    assert plan.conflicts["license_url"] == (None, "https://creativecommons.org/licenses/by/4.0/")
    assert plan.conflicts["license"][1] == "CC BY 4.0"


def test_a_clean_group_still_gets_both_halves_of_the_license() -> None:
    """The pairing rule must only fire on a conflict, never hold back an ordinary write."""
    plan = plan_group("historical", {}, "CESM2-WACCM6", "input")
    assert plan.to_set["license"] == "CC BY 4.0"
    assert plan.to_set["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert not plan.conflicts


LEGACY_ONLY = {
    "srm_downscaling:gcm": "CESM2-WACCM6",
    "srm_downscaling:downscaling_method": "QDMSD",
    "srm_downscaling:observation_dataset": "ERA5",
    "srm_downscaling:scenario": "SSP245",
}


def test_readers_accept_the_legacy_namespace() -> None:
    """v1.0.0 is fixed in place, not regenerated, so its provenance must still be readable.

    A reader that saw only the current namespace would find nothing on v1.0.0 and skip the checks
    that depend on provenance rather than fail. A check that quietly passes is the more dangerous
    outcome, because nothing downstream reports that it never ran.
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
