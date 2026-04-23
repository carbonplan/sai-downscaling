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


_GROUP_CHECKS: dict[str, list[Callable[[str, str], CheckResult]]] = {
    "integrity": [check_ensemble_member_dim],
    "cross-scenario": [],
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
