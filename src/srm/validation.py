"""
Input data validation checks for BCSD pipeline datasets.

Checks are organized into the following groups:
- integrity: checks for data completeness and correctness, such as missing values, duplicates
- lineage: verifies that all lineage-resolved parent members exist in their stores

Use :class:`DatasetValidator` to run checks for a given (gcm, scenario) pair.
"""

import enum
import hashlib
import traceback
from typing import ClassVar

import pandas as pd
import pydantic
import xarray as xr

from srm.datasets import catalog

# Blocking: crash or silent wrong output — abort the pipeline run.
# Warning:  wrong data ingested — emit a warning but continue.
# Info:     incomplete provenance — informational only.
BLOCKING_CHECKS = {
    "ensemble_member_dim",
    "g6_not_identical_to_ssp245",
    "lineage_member_availability",
    "temporal_coverage",
}
WARNING_CHECKS: set[str] = set()
INFO_CHECKS: set[str] = set()


GCM_OPTIONS = ("CESM2-WACCM", "MIROC-ES2H", "UKESM")
SCENARIO_OPTIONS = ("historical", "SSP245", "G6-1.5K")

# Expected inclusive daily time bounds per scenario (CMIP6 conventions).
# to_xarray() normalizes all GCM calendars to proleptic_gregorian (numpy datetime64),
# so end dates are fixed Gregorian strings regardless of original GCM calendar.
_SCENARIO_TIME_BOUNDS: dict[str, tuple[str, str]] = {
    "historical": ("1850-01-01", "2014-12-31"),
    "SSP245": ("2015-01-01", "2100-12-31"),
    "G6-1.5K": ("2035-01-01", "2085-12-31"),
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


def _get_ensemble_members(ds: xr.Dataset) -> list[str] | None:
    """
    Return ensemble_member coordinate values as strings from an already-opened dataset.

    Returns None if the dataset has no ensemble_member dimension or coordinate.
    """
    if "ensemble_member" in ds.dims or "ensemble_member" in ds.coords:
        return [str(m) for m in ds["ensemble_member"].values]
    return None


class DatasetValidator(pydantic.BaseModel):
    """
    Validates a (gcm, scenario) pair against the BCSD pipeline dataset catalog.

    Raises ``pydantic.ValidationError`` on construction if the gcm or scenario is
    not a recognized option.

    Parameters
    ----------
    gcm : str
        GCM name, must be one of :data:`GCM_OPTIONS`.
    scenario : str
        Scenario name, must be one of :data:`SCENARIO_OPTIONS`.

    Examples
    --------
    >>> validator = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245")
    >>> results = validator.run_checks()  # all checks
    >>> result = validator.check_temporal_coverage()  # single check
    """

    model_config = pydantic.ConfigDict(frozen=True)

    gcm: str
    scenario: str

    _dataset_cache: dict[str, xr.Dataset | None] = pydantic.PrivateAttr(default_factory=dict)
    _load_errors: dict[str, tuple[str, str]] = pydantic.PrivateAttr(default_factory=dict)

    _CHECKS: ClassVar[list[str]] = [
        "check_ensemble_member_dim",
        "check_g6_not_identical_to_ssp245",
        "check_temporal_coverage",
        "check_lineage_member_availability",
    ]

    @pydantic.field_validator("gcm")
    @classmethod
    def _validate_gcm(cls, v: str) -> str:
        return parse_gcm(v)

    @pydantic.field_validator("scenario")
    @classmethod
    def _validate_scenario(cls, v: str) -> str:
        return parse_scenario(v)

    # ── private helpers ──────────────────────────────────────────────────────────

    def _result(
        self,
        status: CheckStatus,
        message: str = "",
        detail: dict | None = None,
    ) -> CheckResult:
        """Pre-filled CheckResult factory — eliminates repeated gcm=/scenario= kwargs."""
        return CheckResult(
            gcm=self.gcm,
            scenario=self.scenario,
            status=status,
            message=message,
            detail=detail or {},
        )

    def _open_dataset(
        self, key: str, *, on_missing: CheckStatus = CheckStatus.FAIL
    ) -> tuple[xr.Dataset | None, CheckResult | None]:
        """
        Catalog lookup + ``to_xarray()`` with unified error handling.

        Returns ``(dataset, None)`` on success or ``(None, CheckResult)`` on any failure.
        The ``on_missing`` status controls what is returned when the key is absent from
        the catalog (typically ``SKIP`` for optional datasets, ``FAIL`` for required ones).
        Load errors always produce a ``FAIL`` result regardless of ``on_missing``.
        """
        if key in self._load_errors:
            msg, tb = self._load_errors[key]
            return None, self._result(CheckStatus.FAIL, msg, {"traceback": tb})

        if key not in self._dataset_cache:
            catalog_ds = catalog.datasets.get(key)
            if catalog_ds is None:
                self._dataset_cache[key] = None
            else:
                try:
                    self._dataset_cache[key] = catalog_ds.to_xarray()
                except Exception as exc:
                    tb = traceback.format_exc()
                    self._load_errors[key] = (f"Failed to load dataset {key}: {exc}", tb)
                    return None, self._result(
                        CheckStatus.FAIL,
                        f"Failed to load dataset {key}: {exc}",
                        {"traceback": tb},
                    )

        ds = self._dataset_cache[key]
        if ds is None:
            detail = (
                {"available_keys": sorted(catalog.datasets.keys())}
                if on_missing == CheckStatus.FAIL
                else {}
            )
            return None, self._result(on_missing, f"Dataset not found in catalog: {key}", detail)
        return ds, None

    # ── public check methods ─────────────────────────────────────────────────────

    def check_ensemble_member_dim(self) -> CheckResult:
        """
        Check that the ensemble member dimension is present and correctly named in the dataset.
        """
        key = f"{self.gcm}-{self.scenario}-icechunk"
        ds, err = self._open_dataset(key, on_missing=CheckStatus.FAIL)
        if err is not None:
            return err
        assert ds is not None

        detail = {
            "dims": dict(ds.sizes),
            "coords": list(ds.coords),
            "data_vars": list(ds.data_vars),
        }
        if "ensemble_member" in ds.dims and ds.sizes["ensemble_member"] >= 1:
            return self._result(
                CheckStatus.PASS,
                "Ensemble member dimension is present and correctly named.",
                detail,
            )
        return self._result(
            CheckStatus.FAIL,
            "Ensemble member dimension is missing or incorrectly named.",
            detail,
        )

    def check_lineage_member_availability(self) -> CheckResult:
        """
        D1/D2: All lineage-resolved parent members must exist in their stores.

        For each (member, variable) pair registered in the lineage table, resolves the
        historical_member and (for G6-1.5K) the ssp245_member, then checks those exist
        in the respective icechunk stores.

        Skipped when no lineage is registered for this (gcm, scenario).
        """
        from srm.lineage import get_lineage_entries

        entries = get_lineage_entries(self.gcm, self.scenario)
        if not entries:
            return self._result(
                CheckStatus.SKIP,
                f"No lineage registered for {self.gcm} {self.scenario}; skipping.",
            )

        hist_members_needed: set[str] = set()
        ssp245_members_needed: set[str] = set()
        for hist, ssp245 in entries.values():
            hist_members_needed.add(hist)
            if ssp245 is not None:
                ssp245_members_needed.add(ssp245)

        hist_ds, err = self._open_dataset(
            f"{self.gcm}-historical-icechunk", on_missing=CheckStatus.FAIL
        )
        if err is not None:
            return err
        assert hist_ds is not None
        hist_available = set(_get_ensemble_members(hist_ds) or [])

        issues: list[str] = []
        detail: dict = {
            "historical_needed": sorted(hist_members_needed),
            "historical_available": sorted(hist_available),
        }

        missing_hist = sorted(hist_members_needed - hist_available)
        if missing_hist:
            issues.append(f"{len(missing_hist)} resolved historical member(s) missing")
            detail["missing_historical"] = missing_hist

        if ssp245_members_needed:
            ssp245_ds, err = self._open_dataset(
                f"{self.gcm}-SSP245-icechunk", on_missing=CheckStatus.FAIL
            )
            if err is not None:
                return err
            assert ssp245_ds is not None
            ssp245_available = set(_get_ensemble_members(ssp245_ds) or [])
            detail["ssp245_needed"] = sorted(ssp245_members_needed)
            detail["ssp245_available"] = sorted(ssp245_available)
            missing_ssp245 = sorted(ssp245_members_needed - ssp245_available)
            if missing_ssp245:
                issues.append(f"{len(missing_ssp245)} resolved SSP245 bridge member(s) missing")
                detail["missing_ssp245"] = missing_ssp245

        if issues:
            return self._result(CheckStatus.FAIL, "; ".join(issues), detail)

        store_labels = "historical" + (" and SSP245" if ssp245_members_needed else "")
        return self._result(
            CheckStatus.PASS,
            f"All {len(entries)} lineage entries resolve to available members "
            f"in {store_labels} store(s).",
            detail,
        )

    def check_g6_not_identical_to_ssp245(self) -> CheckResult:
        """
        E1: G6-1.5K data must not be identical to SSP245 for the same ensemble member.

        Detects copy-paste errors where G6 data was accidentally duplicated from SSP245.
        Samples a small corner slice (first 5 time steps, 10×10 spatial patch) of the
        first common variable and compares SHA-256 checksums per shared member.
        Skipped when scenario is not ``'G6-1.5K'``.
        """
        if self.scenario != "G6-1.5K":
            return self._result(CheckStatus.SKIP, "Only applicable to G6-1.5K scenario.")

        g6_key = f"{self.gcm}-G6-1.5K-icechunk"
        ssp245_key = f"{self.gcm}-SSP245-icechunk"

        g6_ds, err = self._open_dataset(g6_key, on_missing=CheckStatus.SKIP)
        if err is not None:
            return err
        assert g6_ds is not None

        ssp245_ds, err = self._open_dataset(ssp245_key, on_missing=CheckStatus.SKIP)
        if err is not None:
            return err
        assert ssp245_ds is not None

        g6_vars = {str(v) for v in g6_ds.data_vars}
        ssp245_vars = {str(v) for v in ssp245_ds.data_vars}
        common_vars = sorted(g6_vars & ssp245_vars)
        if not common_vars:
            return self._result(
                CheckStatus.SKIP,
                "No common variables between G6-1.5K and SSP245; cannot compare.",
                {"g6_vars": sorted(g6_vars), "ssp245_vars": sorted(ssp245_vars)},
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
                return self._result(
                    CheckStatus.SKIP,
                    "No shared ensemble members between G6-1.5K and SSP245.",
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
            return self._result(
                CheckStatus.FAIL,
                (
                    f"G6-1.5K data is identical to SSP245 for {len(identical_members)} member(s) "
                    f"(var={var!r}, first 5 time steps, 10×10 corner patch)."
                ),
                {
                    "identical_members": identical_members,
                    "checked_members": checked_members,
                    "variable_sampled": var,
                },
            )

        return self._result(
            CheckStatus.PASS,
            (
                f"G6-1.5K and SSP245 data differ for all {len(checked_members)} "
                f"checked member(s) (var={var!r})."
            ),
            {"checked_members": checked_members, "variable_sampled": var},
        )

    def check_temporal_coverage(self) -> CheckResult:
        """
        E2: Time axis must be gapless with correct first and last dates.

        Time axis must span the expected date range with no gaps.

        to_xarray() normalizes all GCM calendars to proleptic_gregorian (numpy
        datetime64), so expected bounds are fixed Gregorian strings.
        """
        key = f"{self.gcm}-{self.scenario}-icechunk"
        ds, err = self._open_dataset(key, on_missing=CheckStatus.SKIP)
        if err is not None:
            return err
        assert ds is not None

        if "time" not in ds.dims:
            return self._result(CheckStatus.SKIP, "Dataset has no time dimension.")

        time_index = ds.indexes["time"]
        n_times = len(time_index)

        expected_start, expected_end = _SCENARIO_TIME_BOUNDS[self.scenario]
        n_expected = len(pd.date_range(start=expected_start, end=expected_end, freq="D"))

        actual_start_str = time_index[0].strftime("%Y-%m-%d")
        actual_end_str = time_index[-1].strftime("%Y-%m-%d")

        detail: dict = {
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
            return self._result(CheckStatus.FAIL, "; ".join(issues), detail)

        return self._result(
            CheckStatus.PASS,
            f"Temporal coverage complete: {expected_start} to {expected_end} ({n_times} daily steps, no gaps).",
            detail,
        )

    # ── orchestration ────────────────────────────────────────────────────────────

    def run_checks(self) -> list[CheckResult]:
        """Run all applicable checks and return results stamped with ``check_id``."""
        results = []
        for check_name in self._CHECKS:
            result = getattr(self, check_name)()
            check_id = check_name.removeprefix("check_")
            results.append(result.model_copy(update={"check_id": check_id}))
        return results
