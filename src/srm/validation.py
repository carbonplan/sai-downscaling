"""
Input data validation check for BCSD pipeline datasets.

Checks are organized into the following groups:
- integrity: checks for data completeness and correctness, such as missing values, duplicates
- cross-scenario: member consistency across historical, SSP245, and G6 scenarios

Each check is a standalone function returning a CheckResult.
"""

import enum
import traceback
from collections.abc import Callable

import pydantic
import xarray as xr

from srm.datasets import catalog

# Blocking: crash or silent wrong output — abort the pipeline run.
# Warning:  wrong data ingested — emit a warning but continue.
# Info:     incomplete provenance — informational only.
BLOCKING_CHECKS = {"ensemble_member_dim", "ssp245_hist_member_pairing", "g6_ssp245_member_pairing"}
WARNING_CHECKS = {"temporal_coverage"}
INFO_CHECKS = {"branch_time_attr"}


GCM_OPTIONS = ("CESM2-WACCM", "MIROC-ES2H", "UKESM")
SCENARIO_OPTIONS = ("historical", "SSP245", "G6-1.5K")


def _parse_catalog_value(label: str, value: str, options: tuple[str, ...]) -> str:
    if value in options:
        return value
    raise ValueError(f"Unknown {label} '{value}'. Valid options: {list(options)}")


def parse_gcm(value: str) -> str:
    """Parse a user-provided GCM string to a canonical catalog value."""
    return _parse_catalog_value("GCM", value, GCM_OPTIONS)


def parse_scenario(value: str) -> str:
    """Parse a user-provided scenario string to a canonical catalog value."""
    return _parse_catalog_value("scenario", value, SCENARIO_OPTIONS)


class CheckStatus(enum.StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # not applicable for this (gcm, scenario) pair
    UNKNOWN = "unknown"  # check could not be determined


class CheckResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    check_id: str = ""
    gcm: str
    scenario: str
    status: CheckStatus
    message: str = ""
    detail: dict = pydantic.Field(default_factory=dict)


def check_ensemble_member_dim(gcm: str, scenario: str) -> CheckResult:
    """
    Check that the ensemble member dimension is present and correctly named in the dataset.
    """
    try:
        parsed_gcm = parse_gcm(gcm)
        parsed_scenario = parse_scenario(scenario)
    except ValueError as exc:
        return CheckResult(
            gcm=str(gcm),
            scenario=str(scenario),
            status=CheckStatus.FAIL,
            message=str(exc),
            detail={"valid_gcms": list(GCM_OPTIONS), "valid_scenarios": list(SCENARIO_OPTIONS)},
        )

    key = f"{parsed_gcm}-{parsed_scenario}-icechunk"
    dataset = catalog.datasets.get(key)

    if dataset is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Dataset for {parsed_gcm} {parsed_scenario} not found in catalog using key {key}",
            detail={"valid_keys": sorted(catalog.datasets.keys())},
        )

    try:
        ds = dataset.to_xarray()

    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Error loading dataset for {parsed_gcm} {parsed_scenario} using key {key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )
    detail = {"dims": dict(ds.sizes), "coords": list(ds.coords), "data_vars": list(ds.data_vars)}
    if "ensemble_member" in ds.dims and ds.sizes["ensemble_member"] >= 1:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.PASS,
            message="Ensemble member dimension is present and correctly named.",
            detail=detail,
        )

    return CheckResult(
        gcm=parsed_gcm,
        scenario=parsed_scenario,
        status=CheckStatus.FAIL,
        message="Ensemble member dimension is missing or incorrectly named.",
        detail=detail,
    )


def _get_ensemble_members(ds: xr.Dataset) -> list[str] | None:
    """
    Return ensemble_member coordinate values as strings from an already-opened dataset.

    Returns None if the dataset has no ensemble_member dimension or coordinate.
    """
    if "ensemble_member" in ds.dims or "ensemble_member" in ds.coords:
        return [str(m) for m in ds["ensemble_member"].values]
    return None


def check_ssp245_hist_member_pairing(gcm: str, scenario: str) -> CheckResult:
    """
    D1: Every SSP245 ensemble member must have a matching member in the historical store.

    Skipped when scenario is not 'SSP245'. Passes automatically when the historical
    dataset has no ensemble_member dimension (treated as a single shared run).
    """
    try:
        parsed_gcm = parse_gcm(gcm)
        parsed_scenario = parse_scenario(scenario)
    except ValueError as exc:
        return CheckResult(
            gcm=str(gcm), scenario=str(scenario), status=CheckStatus.FAIL, message=str(exc)
        )

    if parsed_scenario != "SSP245":
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message="Only applicable to SSP245 scenario.",
        )

    ssp245_key = f"{parsed_gcm}-SSP245-icechunk"
    hist_key = f"{parsed_gcm}-historical-icechunk"

    ssp245_ds = catalog.datasets.get(ssp245_key)
    hist_ds = catalog.datasets.get(hist_key)

    if ssp245_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message=f"SSP245 dataset not found in catalog: {ssp245_key}",
        )
    if hist_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Historical dataset not found in catalog: {hist_key}",
            detail={"available_keys": sorted(catalog.datasets.keys())},
        )

    try:
        ssp245_members = _get_ensemble_members(ssp245_ds.to_xarray())
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load SSP245 dataset {ssp245_key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    if ssp245_members is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message="SSP245 dataset has no ensemble_member dimension; nothing to check.",
        )

    try:
        hist_members = _get_ensemble_members(hist_ds.to_xarray())
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load historical dataset {hist_key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    if hist_members is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.PASS,
            message="Historical has no ensemble_member dim; treated as single shared run matching all SSP245 members.",
            detail={"ssp245_members": sorted(ssp245_members)},
        )

    unmatched = sorted(set(ssp245_members) - set(hist_members))
    if unmatched:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"{len(unmatched)} SSP245 member(s) not found in historical.",
            detail={
                "unmatched": unmatched,
                "ssp245_members": sorted(ssp245_members),
                "historical_members": sorted(hist_members),
            },
        )

    return CheckResult(
        gcm=parsed_gcm,
        scenario=parsed_scenario,
        status=CheckStatus.PASS,
        message=f"All {len(ssp245_members)} SSP245 members present in historical.",
        detail={"members": sorted(ssp245_members)},
    )


def check_g6_ssp245_member_pairing(gcm: str, scenario: str) -> CheckResult:
    """
    D2: Every G6-1.5K ensemble member must have a matching member in the SSP245 store.

    The SSP245 bridge run is required for SAI scenario detrending. Skipped when
    scenario is not 'G6-1.5K' or when the GCM has no G6-1.5K dataset in the catalog.
    """
    try:
        parsed_gcm = parse_gcm(gcm)
        parsed_scenario = parse_scenario(scenario)
    except ValueError as exc:
        return CheckResult(
            gcm=str(gcm), scenario=str(scenario), status=CheckStatus.FAIL, message=str(exc)
        )

    if parsed_scenario != "G6-1.5K":
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message="Only applicable to G6-1.5K scenario.",
        )

    g6_key = f"{parsed_gcm}-G6-1.5K-icechunk"
    ssp245_key = f"{parsed_gcm}-SSP245-icechunk"

    g6_ds = catalog.datasets.get(g6_key)
    ssp245_ds = catalog.datasets.get(ssp245_key)

    if g6_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message=f"G6-1.5K dataset not found in catalog: {g6_key}",
        )
    if ssp245_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"SSP245 dataset not found in catalog: {ssp245_key} (required for G6 detrending bridge).",
            detail={"available_keys": sorted(catalog.datasets.keys())},
        )

    try:
        g6_members = _get_ensemble_members(g6_ds.to_xarray())
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load G6-1.5K dataset {g6_key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    if g6_members is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message="G6-1.5K dataset has no ensemble_member dimension; nothing to check.",
        )

    try:
        ssp245_members = _get_ensemble_members(ssp245_ds.to_xarray())
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load SSP245 dataset {ssp245_key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    if ssp245_members is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message="SSP245 dataset has no ensemble_member dimension; cannot verify G6 member pairing.",
        )

    unmatched = sorted(set(g6_members) - set(ssp245_members))
    if unmatched:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"{len(unmatched)} G6 member(s) not found in SSP245 (required for SAI detrending bridge).",
            detail={
                "unmatched": unmatched,
                "g6_members": sorted(g6_members),
                "ssp245_members": sorted(ssp245_members),
            },
        )

    return CheckResult(
        gcm=parsed_gcm,
        scenario=parsed_scenario,
        status=CheckStatus.PASS,
        message=f"All {len(g6_members)} G6 members present in SSP245.",
        detail={"members": sorted(g6_members)},
    )


_GROUP_CHECKS: dict[str, list[Callable[[str, str], CheckResult]]] = {
    "integrity": [check_ensemble_member_dim],
    "cross-scenario": [check_ssp245_hist_member_pairing, check_g6_ssp245_member_pairing],
}


def validate_dataset(gcm: str, scenario: str) -> list[CheckResult]:
    """
    Run all applicable checks for the given GCM and scenario and return CheckResults.
    """
    try:
        parsed_gcm = parse_gcm(gcm)
        parsed_scenario = parse_scenario(scenario)
    except ValueError as exc:
        return [
            CheckResult(
                gcm=str(gcm),
                scenario=str(scenario),
                status=CheckStatus.FAIL,
                message=str(exc),
                detail={"valid_gcms": list(GCM_OPTIONS), "valid_scenarios": list(SCENARIO_OPTIONS)},
            )
        ]

    results = []
    for _group_name, checks in _GROUP_CHECKS.items():
        for check in checks:
            result = check(parsed_gcm, parsed_scenario)
            check_id = check.__name__.removeprefix("check_")
            results.append(result.model_copy(update={"check_id": check_id}))
    return results
