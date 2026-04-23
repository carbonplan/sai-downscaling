"""
Input data validation check for BCSD pipeline datasets.

Checks are organized into the following groups:
- integrity: checks for data completeness and correctness, such as missing values, duplicates
- cross-scenario: member consistency across historical, SSP245, and G6 scenarios

Each check is a standalone function returning a CheckResult.
"""

import enum
import hashlib
import traceback
from collections.abc import Callable

import pydantic
import xarray as xr

from srm.datasets import catalog

# Blocking: crash or silent wrong output — abort the pipeline run.
# Warning:  wrong data ingested — emit a warning but continue.
# Info:     incomplete provenance — informational only.
BLOCKING_CHECKS = {
    "ensemble_member_dim",
    "g6_not_identical_to_ssp245",
    "temporal_coverage",
    "ssp245_hist_member_pairing",
    "g6_ssp245_member_pairing",
}
WARNING_CHECKS: set[str] = set()
INFO_CHECKS = {"branch_time_attr"}


GCM_OPTIONS = ("CESM2-WACCM", "MIROC-ES2H", "UKESM")
SCENARIO_OPTIONS = ("historical", "SSP245", "G6-1.5K")

# Expected inclusive daily time bounds per scenario (CMIP6 conventions).
_SCENARIO_TIME_BOUNDS: dict[str, tuple[str, str]] = {
    "historical": ("1850-01-01", "2014-12-31"),
    "SSP245": ("2015-01-01", "2100-12-31"),
    "G6-1.5K": ("2015-01-01", "2100-12-31"),
}


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


def check_g6_not_identical_to_ssp245(gcm: str, scenario: str) -> CheckResult:
    """
    E1: G6-1.5K data must not be identical to SSP245 for the same ensemble member.

    Detects copy-paste errors where G6 data was accidentally duplicated from SSP245.
    Samples a small corner slice (first 5 time steps, 10×10 spatial patch) of the
    first common variable and compares SHA-256 checksums per shared member.
    Skipped when scenario is not 'G6-1.5K'.
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

    g6_catalog_ds = catalog.datasets.get(g6_key)
    ssp245_catalog_ds = catalog.datasets.get(ssp245_key)

    if g6_catalog_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message=f"G6-1.5K dataset not found in catalog: {g6_key}",
        )
    if ssp245_catalog_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message=f"SSP245 dataset not found in catalog: {ssp245_key}",
        )

    try:
        g6_ds = g6_catalog_ds.to_xarray()
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load G6-1.5K dataset {g6_key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    try:
        ssp245_ds = ssp245_catalog_ds.to_xarray()
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load SSP245 dataset {ssp245_key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    common_vars = sorted(set(g6_ds.data_vars) & set(ssp245_ds.data_vars))
    if not common_vars:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message="No common variables between G6-1.5K and SSP245; cannot compare.",
            detail={
                "g6_vars": sorted(g6_ds.data_vars),
                "ssp245_vars": sorted(ssp245_ds.data_vars),
            },
        )

    var = common_vars[0]
    sample_kwargs = {"time": slice(0, 5), "lat": slice(0, 10), "lon": slice(0, 10)}

    def _checksum(arr: xr.DataArray) -> str:
        return hashlib.sha256(arr.isel(**sample_kwargs).values.tobytes()).hexdigest()

    g6_members = _get_ensemble_members(g6_ds)
    ssp245_members = _get_ensemble_members(ssp245_ds)

    identical_members: list[str] = []
    checked_members: list[str] = []

    if g6_members is not None and ssp245_members is not None:
        shared = sorted(set(g6_members) & set(ssp245_members))
        if not shared:
            return CheckResult(
                gcm=parsed_gcm,
                scenario=parsed_scenario,
                status=CheckStatus.SKIP,
                message="No shared ensemble members between G6-1.5K and SSP245.",
            )
        for member in shared:
            g6_hash = _checksum(g6_ds[var].sel(ensemble_member=member))
            ssp245_hash = _checksum(ssp245_ds[var].sel(ensemble_member=member))
            checked_members.append(member)
            if g6_hash == ssp245_hash:
                identical_members.append(member)
    else:
        g6_hash = _checksum(g6_ds[var])
        ssp245_hash = _checksum(ssp245_ds[var])
        checked_members = ["(all)"]
        if g6_hash == ssp245_hash:
            identical_members = ["(all)"]

    if identical_members:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=(
                f"G6-1.5K data is identical to SSP245 for {len(identical_members)} member(s) "
                f"(var={var!r}, first 5 time steps, 10×10 corner patch)."
            ),
            detail={
                "identical_members": identical_members,
                "checked_members": checked_members,
                "variable_sampled": var,
            },
        )

    return CheckResult(
        gcm=parsed_gcm,
        scenario=parsed_scenario,
        status=CheckStatus.PASS,
        message=(
            f"G6-1.5K and SSP245 data differ for all {len(checked_members)} "
            f"checked member(s) (var={var!r})."
        ),
        detail={"checked_members": checked_members, "variable_sampled": var},
    )


def check_temporal_coverage(gcm: str, scenario: str) -> CheckResult:
    """
    E2: Time axis must be gapless with correct first and last dates.

    Calendar-aware: uses ``xr.cftime_range`` with the dataset's own calendar so
    that 360-day (UKESM), noleap (CESM2), and Gregorian (MIROC) datasets are
    handled correctly without unsafe coercion to numpy datetime64.
    """
    try:
        parsed_gcm = parse_gcm(gcm)
        parsed_scenario = parse_scenario(scenario)
    except ValueError as exc:
        return CheckResult(
            gcm=str(gcm), scenario=str(scenario), status=CheckStatus.FAIL, message=str(exc)
        )

    key = f"{parsed_gcm}-{parsed_scenario}-icechunk"
    catalog_ds = catalog.datasets.get(key)

    if catalog_ds is None:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message=f"Dataset not found in catalog: {key}",
        )

    try:
        ds = catalog_ds.to_xarray()
    except Exception as exc:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message=f"Failed to load dataset {key}: {exc}",
            detail={"traceback": traceback.format_exc()},
        )

    if "time" not in ds.dims:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.SKIP,
            message="Dataset has no time dimension.",
        )

    time_index = ds.indexes["time"]
    # ds.time.dt.calendar works for both CFTime and numpy datetime64 (returns
    # "proleptic_gregorian" for the latter), so xr.cftime_range always gets the
    # right calendar — no unsafe astype("datetime64") cast needed.
    calendar = ds.time.dt.calendar
    n_times = len(time_index)

    expected_start, expected_end = _SCENARIO_TIME_BOUNDS[parsed_scenario]

    # Build the expected daily range using the dataset's own calendar so that
    # 360-day (UKESM), noleap (CESM2), and Gregorian (MIROC) datasets are all
    # handled correctly.
    expected_range = xr.date_range(
        start=expected_start, end=expected_end, freq="D", calendar=calendar, use_cftime=True
    )
    n_expected = len(expected_range)

    # strftime works on both cftime.datetime and pandas.Timestamp.
    actual_start_str = time_index[0].strftime("%Y-%m-%d")
    actual_end_str = time_index[-1].strftime("%Y-%m-%d")

    detail: dict = {
        "calendar": calendar,
        "actual_start": actual_start_str,
        "actual_end": actual_end_str,
        "n_times": n_times,
        "n_expected": n_expected,
        "expected_start": expected_start,
        "expected_end": expected_end,
    }

    issues: list[str] = []

    if actual_start_str != expected_start:
        issues.append(f"start date {actual_start_str} != expected {expected_start}")
    if actual_end_str != expected_end:
        issues.append(f"end date {actual_end_str} != expected {expected_end}")

    if n_times != n_expected:
        delta = n_expected - n_times
        label = "missing" if delta > 0 else "extra"
        detail["n_missing_or_extra"] = delta
        issues.append(f"{abs(delta)} {label} time steps (expected {n_expected}, got {n_times})")
    elif n_times > 1:
        # Counts match; verify regularity (gaps / duplicates within the range).
        inferred_freq = xr.infer_freq(ds.time)
        if inferred_freq != "D":
            detail["inferred_freq"] = inferred_freq
            issues.append(
                f"time axis is not uniformly daily (inferred freq: {inferred_freq!r}); "
                "possible gaps or duplicates within range"
            )

    if issues:
        return CheckResult(
            gcm=parsed_gcm,
            scenario=parsed_scenario,
            status=CheckStatus.FAIL,
            message="; ".join(issues),
            detail=detail,
        )

    return CheckResult(
        gcm=parsed_gcm,
        scenario=parsed_scenario,
        status=CheckStatus.PASS,
        message=f"Temporal coverage complete: {expected_start} to {expected_end} ({n_times} daily steps, no gaps).",
        detail=detail,
    )


_GROUP_CHECKS: dict[str, list[Callable[[str, str], CheckResult]]] = {
    "integrity": [
        check_ensemble_member_dim,
        check_g6_not_identical_to_ssp245,
        check_temporal_coverage,
    ],
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
